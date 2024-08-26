import torch
import pytest
from functools import partial

from open_lm.attention import flex_attn, no_mask, mask_all, prefix_mask, prefix_causal_mask, causal_mask, no_prefix_causal_mask, create_block_mask_cached, no_prefix_mask, offset_causal_mask

# fex_attn = torch.compile(flex_attn, dynamic=False)
BLOCK_SIZE = 16

###### flex debug ######
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)
# flex_attention = torch.compile(flex_attention, dynamic=False)

from tabulate import tabulate
import random
from functools import lru_cache, partial

import torch
import torch.nn.functional as F

from triton.testing import do_bench

def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12


def _test_mask(
    score_mod=None,
    mask_mod=None,
    B=16,
    H=16,
    S=8192,
    D=64,
    skip_correctness=False,
    print_mask=True,
):
    assert (
        score_mod is not None or mask_mod is not None
    ), "Must provide a score_mod or mask_mod"
    query = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    key = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    value = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    gradOut = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    if mask_mod is not None:
        block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=query.device, BLOCK_SIZE=BLOCK_SIZE)
    else:
        block_mask = None
    sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
    mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=query.device)

    causal_fa2 = lambda: F.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    xformers_mask = lambda: F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    )
    flex_attention_call = lambda: flex_attention(
        query, key, value, score_mod=score_mod, block_mask=block_mask
    )

    results = []
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    causal_fav2_flops = 0.5 * B * H * D * S * S
    flops = density * B * H * D * S * S

    # Forward pass
    causal_fa2_time = do_bench(causal_fa2)
    xformers_mask_time = do_bench(xformers_mask)
    flex_ms = do_bench(flex_attention_call)

    # Backward pass
    causal_fa2_out = causal_fa2()
    xformers_out = xformers_mask()
    flex_out = flex_attention_call()

    causal_fa2_bw_time = do_bench(
        lambda: causal_fa2_out.backward(gradOut, retain_graph=True)
    )
    xformers_mask_bw_time = do_bench(
        lambda: xformers_out.backward(gradOut, retain_graph=True)
    )
    flex_bw_ms = do_bench(lambda: flex_out.backward(gradOut, retain_graph=True))

    # Inline correctness check
    if not skip_correctness:
        xformers_outs = []
        flex_outs = []

        query.grad = None
        key.grad = None
        value.grad = None

        out1 = xformers_mask()
        xformers_outs.append(out1)
        out1.backward(gradOut)
        xformers_outs += [query.grad, key.grad, value.grad]

        query.grad = None
        key.grad = None
        value.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [query.grad, key.grad, value.grad]
        for flex, xformer in zip(flex_outs, xformers_outs):
            torch.testing.assert_close(flex, xformer, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")
    # Usage in your results formatting:
    results = [
        [
            "causal FA2",
            f"{causal_fa2_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_time, 4):.2f}",
            f"{causal_fa2_bw_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10):.2f}",
        ],
        [
            "F.sdpa + mask",
            f"{xformers_mask_time:.4f}",
            f"{calculate_tflops(flops, xformers_mask_time, 4):.2f}",
            f"{xformers_mask_bw_time:.4f}",
            f"{calculate_tflops(flops, xformers_mask_bw_time, 10):.2f}",
        ],
        [
            "flexattention",
            f"{flex_ms:.4f}",
            f"{calculate_tflops(flops, flex_ms, 4):.2f}",
            f"{flex_bw_ms:.4f}",
            f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
        ],
    ]
    # print(
    #     f"\nResults for {score_mod.__name__ if score_mod is not None else mask_mod.__name__}:"
    # )
    print(
        tabulate(
            results,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ],
            tablefmt="grid",
        )
    )
    if print_mask:
        print(f"\nBlock Mask:\n{block_mask}")

    # Clean up to save memory
    del query, key, value, gradOut, causal_fa2_out, xformers_out, flex_out
    torch.cuda.empty_cache()

###### end flex debug ######




@pytest.mark.gpu
def test_attention_masking():
    # Use a multiple of the block size to ensure that the block mask is created correctly
    b, n, h, d = 1, BLOCK_SIZE*8, 1, 32
    prefix_length = BLOCK_SIZE*4
    queries = torch.rand((b, n, h, d)).cuda()
    keys = torch.rand((b, n, h, d)).cuda()
    values = torch.rand((b, n, h, d)).cuda()

    # Test masking all elements
    block_mask_all = create_block_mask_cached(mask_all, b, h, n, n, device=queries.device, BLOCK_SIZE=BLOCK_SIZE)
    output_mask_all = flex_attn(queries, keys, values, attention_mask=block_mask_all)
    assert (output_mask_all==0).all()
    
    # Test removing the first elements
    no_prefix_mask_instance = partial(no_prefix_mask, prefix_length=prefix_length)
    no_prefix_block_mask = create_block_mask_cached(no_prefix_mask_instance, b, h, n, n, device=queries.device, BLOCK_SIZE=BLOCK_SIZE)
    block_mask_reduced = create_block_mask_cached(no_mask, b, h, n-prefix_length, n-prefix_length, device=queries.device, BLOCK_SIZE=BLOCK_SIZE)
    
    output_no_prefix = flex_attn(queries, keys, values, attention_mask=no_prefix_block_mask)
    output_reduced = flex_attn(
        queries[:, prefix_length:],
        keys[:, prefix_length:],
        values[:, prefix_length:],
        attention_mask=block_mask_reduced
    )
    
    assert torch.allclose(output_reduced, output_no_prefix[:, prefix_length:], atol=1e-1)
    
    
    # There is a bug with flex attention when the whole line is masked... it returns values that are not 0 (except for the first element)
    
    # # Test causal masking and removing the first elements
    # no_prefix_causal_mask_instance = partial(no_prefix_causal_mask, prefix_length=prefix_length)
    causal_block_mask = create_block_mask_cached(causal_mask, b, h, n, n, device=queries.device, BLOCK_SIZE=BLOCK_SIZE)
    output_causal = flex_attn(queries, keys, values, attention_mask=causal_block_mask)
    # no_prefix_causal_block_mask = create_block_mask_cached(no_prefix_causal_mask_instance, b, h, n, n, device=queries.device, BLOCK_SIZE=BLOCK_SIZE)
    # causal_block_mask_reduced = create_block_mask_cached(causal_mask, b, h, n-prefix_length, n-prefix_length, device=queries.device, BLOCK_SIZE=BLOCK_SIZE)

    # # Run with all elements but mask the first elements
    # output_causal_no_prefix = flex_attn(queries, keys, values, attention_mask=no_prefix_causal_block_mask)
    
    # # Run without the first elements of the sequence
    # output_causal_reduced = flex_attn(
    #     queries[:, prefix_length:],
    #     keys[:, prefix_length:],
    #     values[:, prefix_length:],
    #     attention_mask=causal_block_mask_reduced,
    # )

    # print(f"diff = {torch.abs(output_causal_reduced - output_causal_no_prefix[:, prefix_length:]).sum(-1)}")
    # print(f"output_causal_no_prefix: {output_causal_no_prefix[:, prefix_length:].sum(-1)}")
    
    # _test_mask(mask_mod=no_mask, print_mask=True, B=b, H=h, S=n, D=d)
    # _test_mask(mask_mod=mask_all, print_mask=True, B=b, H=h, S=n, D=d)
    # _test_mask(mask_mod=causal_mask, print_mask=True, B=b, H=h, S=n, D=d)
    # _test_mask(mask_mod=no_prefix_mask_instance, print_mask=True, B=b, H=h, S=n, D=d)
    # _test_mask(mask_mod=no_prefix_causal_mask_instance, print_mask=True, B=b, H=h, S=n, D=d)
    
    # assert torch.allclose(output_causal_reduced, output_causal_no_prefix[:, prefix_length:], atol=1e-1)

    # Run with the output of attention again and ensure it looks good (e.g., we don't run into NaNs. This happened in
    # initial implementations where the output had NaNs for certain types of masks.
    output3 = flex_attn(
        output_causal, output_causal, output_causal, attention_mask=causal_block_mask
    )
    assert not output3.isnan().any()
    
    # Test prefix mask by computing the full attention on the first elements and causal attention on the rest
    prefix_causal_mask_instance = partial(prefix_causal_mask, prefix_length=prefix_length)
    _test_mask(mask_mod=prefix_causal_mask_instance, print_mask=True, B=b, H=h, S=n, D=d)
    prefix_causal_block_mask = create_block_mask_cached(prefix_causal_mask_instance, b, h, n, n, device=queries.device, BLOCK_SIZE=BLOCK_SIZE)
    output_prefix_attention = flex_attn(queries, keys, values, attention_mask=prefix_causal_block_mask)
   
    output_full_attention_prefix = flex_attn(queries[:, :prefix_length], keys[:, :prefix_length], values[:, :prefix_length], attention_mask=None)
    
    offset_causal_mask_instance = partial(offset_causal_mask, offset=prefix_length)
    rectangulal_causal_block_mask = create_block_mask_cached(offset_causal_mask_instance, b, h, n-prefix_length, n, device=queries.device, BLOCK_SIZE=BLOCK_SIZE)
    output_causal_attention_suffix = flex_attn(queries[:, prefix_length:], keys, values, attention_mask=rectangulal_causal_block_mask)
    
    assert torch.allclose(output_prefix_attention[:, :prefix_length], output_full_attention_prefix, atol=1e-1)    
    assert torch.allclose(output_prefix_attention[:, prefix_length:], output_causal_attention_suffix, atol=1e-1)

    
if __name__ == "__main__":
    test_attention_masking()