import torch
from torch.nn import functional as F
from functools import partial

from open_lm.attention import flex_attn, causal_mask, create_block_mask_cached
from open_lm.model import SwiGLUTorch
from open_lm.precision import get_autocast
from xformers.ops import SwiGLU


def get_rectangular_causal_mask(shape, q_seq_len, k_seq_len, device, dtype):
    """Create a rectangular causal mask.

    This is especially useful when query length < key length, and ensures that the attention tensor comes from a tensor
    that initially has dimensions that are a multiple of 8, as required by xformers.

    >>> get_rectangular_causal_mask((1, 1), 2, 2, "cpu", torch.float32)
    tensor([[[[0., -inf],
              [0., 0.]]]])
    >>> get_rectangular_causal_mask((1, 1), 3, 5, "cpu", torch.float32)
    tensor([[[[0., 0., 0., -inf, -inf],
              [0., 0., 0., 0., -inf],
              [0., 0., 0., 0., 0.]]]])
    >>> get_rectangular_causal_mask((1, 1), 5, 5, "cpu", torch.float32)
    tensor([[[[0., -inf, -inf, -inf, -inf],
              [0., 0., -inf, -inf, -inf],
              [0., 0., 0., -inf, -inf],
              [0., 0., 0., 0., -inf],
              [0., 0., 0., 0., 0.]]]])
    """
    # xformers requires the mask to be built with a shape that is a multiple of 8
    next_multiple_8 = (k_seq_len + 7) // 8 * 8  #

    mask = torch.ones((q_seq_len, k_seq_len), device=device, dtype=bool)
    mask[:, -q_seq_len:] = torch.tril(mask[:, -q_seq_len:], diagonal=0)

    output_mask = torch.zeros((*shape, q_seq_len, next_multiple_8), device=device, dtype=dtype)
    output_mask[:, :, :, :k_seq_len].masked_fill_(~mask, torch.finfo(dtype).min)
    return output_mask[:, :, :, :k_seq_len]


def apply_attention_mask_(bias, attention_mask, queries_dtype):
    """Applies attention mask (e.g., from HuggingFace generate) to an attention bias mask in-place.

    Args:
        bias (torch.Tensor, shape (batch_size, num_heads, q_seq_len, k_seq_len))
        attention_mask (torch.Tensor, shape (batch_size, sequence_len))
        queries_dtype: queries.dtype; used to get minimum value for masked indices.

    Returns:
        bias_with_mask (torch.Tensor, shape (batch_size, num_heads, q_seq_len, k_seq_len))
    """
    # Update mask to remove attention based on attention_mask that's passed in.
    assert attention_mask.dim() == 2
    # From https://github.com/huggingface/transformers/blob/f738ab3b5d30e30c43a4c3d00ca8939f8a4d4427/src/transformers/models/llama/modeling_llama.py#L1089C1-L1091C117
    mask_length = attention_mask.shape[-1]
    # Set parts of bias that are zero (i.e., where attention is allowed) _and_ attention_mask is False (i.e.,
    # where we should not attend) with min_dtype.
    padding_mask = bias[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
    min_dtype = torch.finfo(queries_dtype).min
    bias[..., :mask_length] = bias[..., :mask_length].masked_fill(padding_mask, min_dtype)
    # Disable masking for sequence indices where all attention weights are -inf
    # We won't use these anyway, and keeping them as -inf leads to nans.
    # See https://github.com/huggingface/transformers/blob/f738ab3b5d30e30c43a4c3d00ca8939f8a4d4427/src/transformers/modeling_attn_mask_utils.py#L189
    # for details.
    bias.mul_(~torch.all(bias == min_dtype, dim=-1, keepdim=True))


def torch_attn(queries, keys, values, is_causal, attention_mask=None):
    # Need to call contiguous in torch >=2.1, otherwise later calls to .view() fail.
    # Possibly related: https://github.com/pytorch/pytorch/issues/110213 - behavior of scaled_dot_product_attention
    # changed between 2.0 and 2.1
    if is_causal and keys.shape[1] > queries.shape[1] > 1:
        q_seq_len = queries.shape[1]
        k_seq_len = keys.shape[1]
        # Same as above, we would like to use:
        # mask = xops.fmha.attn_bias.LowerTriangularFromBottomRightMask().materialize((1, 1, q_seq_len, k_seq_len), queries.dtype, queries.device)
        mask = get_rectangular_causal_mask((1, 1), q_seq_len, k_seq_len, queries.device, queries.dtype)
        if attention_mask is not None:
            apply_attention_mask_(mask, attention_mask, queries_dtype=queries.dtype)
        return (
            F.scaled_dot_product_attention(
                queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=mask
            )
            .transpose(1, 2)
            .contiguous()
        )
    else:
        if attention_mask is None:
            bias = None
            # If we only have one query, assume we don't need to be in causal mode (can attend to all keys).
            if queries.shape[1] == 1:
                is_causal = False
        else:
            if not is_causal:
                raise NotImplementedError("attention_mask with is_causal=False is not yet implemented.")
            # Build causal mask that assumes queries are in the end of the sequence.
            batch, q_seq_len, heads, _ = queries.shape
            k_seq_len = keys.shape[1]
            bias = get_rectangular_causal_mask((batch, heads), q_seq_len, k_seq_len, queries.device, queries.dtype)
            if attention_mask is not None:
                apply_attention_mask_(bias, attention_mask, queries_dtype=queries.dtype)
            # We apply causal mask in attention instead of using is_causal=True.
            is_causal = False
        return (
            F.scaled_dot_product_attention(
                queries.transpose(1, 2),
                keys.transpose(1, 2),
                values.transpose(1, 2),
                attn_mask=bias,
                is_causal=is_causal,
            )
            .transpose(1, 2)
            .contiguous()
        )


ATTN_ACTIVATIONS = {
    "relu": F.relu,
    "relu_squared": lambda x: torch.pow(F.relu(x), 2),
    # "gelu": F.gelu, # goes to NaN with bais so comment out for now
    "softplus": F.softplus,
    "identity": lambda x: x,
    "relu6": F.relu6,
    "sigmoid": F.sigmoid,
    "softmax": partial(F.softmax, dim=-1),
}

ATTN_SEQ_SCALARS = {
    "max": lambda x: x,
    # "seq": lambda x: torch.arange(x) + 1,  # comment out for now more involved
    "avg": lambda x: (x - 1) / 2 + 1,
    "none": lambda _: 1,
}


def custom_attn(
    queries,
    keys,
    values,
    attn_activation,
    attn_seq_scalar,
    alpha,
    is_causal=False,
    attention_mask=None,
) -> torch.Tensor:
    # naive reference implementation for relu-attention following: https://arxiv.org/pdf/2309.08586.pdf
    # code modifies: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    if attention_mask is not None:
        raise NotImplementedError("attention_mask not yet implemented for custom_attn.")

    batch, q_seq_len, heads, embed_dim = queries.shape
    _, k_seq_len, _, _ = keys.shape

    attn_bias = torch.zeros(batch, heads, q_seq_len, k_seq_len, device=queries.device, dtype=queries.dtype)
    if is_causal and queries.shape[1] > 1:
        attn_bias = get_rectangular_causal_mask((batch, heads), q_seq_len, k_seq_len, queries.device, queries.dtype)

    inner_scale = embed_dim**-0.5
    attn_weight = torch.einsum("bqhd,bkhd->bhqk", inner_scale * queries, keys)
    attn_weight += attn_bias

    # scaling by: 1/L^{-\alpha}
    outter_scale = ATTN_SEQ_SCALARS[attn_seq_scalar](k_seq_len) ** -alpha
    attn_weight = outter_scale * ATTN_ACTIVATIONS[attn_activation](attn_weight)

    return torch.einsum("bhqk,bkhd->bqhd", attn_weight, values)


def test_custom_attn_matches_softmax_attn(threshold=1e-1):
    for bs, q_seq_len, k_seq_len, h, d in [
        [10, 1024, 2048, 8, 128],
        [10, 2048, 1024, 8, 128],
        [10, 2048, 2048, 8, 128],
        [1, 1024, 2048, 8, 128],
    ]:
        queries = torch.rand(bs, q_seq_len, h, d)
        keys = torch.rand(bs, k_seq_len, h, d)
        values = torch.rand(bs, k_seq_len, h, d)

        for is_causal in [True, False]:
            torch_out = torch_attn(queries.cpu(), keys.cpu(), values.cpu(), is_causal=is_causal)
            if is_causal:
                block_causal_mask = create_block_mask_cached(causal_mask, bs, h, q_seq_len, k_seq_len)
            else:
                block_causal_mask = None
            my_out = flex_attn(queries.cpu(), keys.cpu(), values.cpu(), block_causal_mask)

            if torch.cuda.is_available():
                torch_out = torch_attn(queries.cuda(), keys.cuda(), values.cuda(), is_causal=is_causal)
                my_out = flex_attn(queries.cuda(), keys.cuda(), values.cuda(), block_causal_mask)

                assert torch.allclose(
                    torch_out, my_out, atol=threshold, rtol=threshold
                ), "custom_attn incorrectly implements softmax attention"

            assert torch.allclose(
                torch_out, my_out, atol=threshold, rtol=threshold
            ), "custom_attn incorrectly implements softmax attention"


def test_no_failure():
    for nl in ATTN_ACTIVATIONS:
        for os in ATTN_SEQ_SCALARS:
            for bs, q_seq_len, k_seq_len, h, d in [
                [2, 64, 64, 1, 32],
                [2, 64, 16, 1, 32],
                [2, 16, 64, 1, 32],
            ]:
                queries = torch.rand(bs, q_seq_len, h, d)
                keys = torch.rand(bs, k_seq_len, h, d)
                values = torch.rand(bs, k_seq_len, h, d)

                for is_causal in [True, False]:
                    if is_causal:
                        block_causal_mask = create_block_mask_cached(causal_mask, bs, h, q_seq_len, k_seq_len)
                    else:
                        block_causal_mask = None
                    my_out = flex_attn(queries, keys, values, block_causal_mask)

    assert True


def test_swiglu_torch(threshold=1e-7):
    bsz = 5
    in_feats = 10
    hidden_feats = 30
    out_feats = 10
    num_tries = 5

    xops_swiglu = SwiGLU(in_features=in_feats, hidden_features=hidden_feats, out_features=out_feats)
    torch_swiglu = SwiGLUTorch(in_dim=in_feats, hidden_dim=hidden_feats, out_dim=out_feats)

    # Copy state dict from one swiglu to the other so that they have the same weights

    state_dict = xops_swiglu.state_dict()
    new_state_dict = {
        "w12.weight": state_dict["w12.weight"],
        "w3.weight": state_dict["w3.weight"],
        "w12.bias": state_dict["w12.bias"],
        "w3.bias": state_dict["w3.bias"],
    }
    torch_swiglu.load_state_dict(new_state_dict)

    with torch.no_grad():
        for _ in range(num_tries):
            random_in = torch.rand((bsz, in_feats))
            torch_out = torch_swiglu(random_in)
            xops_out = xops_swiglu(random_in)
            assert torch.allclose(torch_out, xops_out, atol=threshold)


if __name__ == "__main__":
    test_custom_attn_matches_softmax_attn()
    test_no_failure()
    test_swiglu_torch()
    print("All tests passed.")