from functools import partial
from typing import Tuple, Union

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, or_masks, and_masks, _DEFAULT_SPARSE_BLOCK_SIZE
from functools import lru_cache

@lru_cache()
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", BLOCK_SIZE: Union[int, Tuple[int, int]] = _DEFAULT_SPARSE_BLOCK_SIZE, _compile=False, **kwargs):
    if kwargs:
        score_mod = partial(score_mod, **kwargs)
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, BLOCK_SIZE=BLOCK_SIZE, _compile=_compile)
    return block_mask


def no_mask(b, h, q_idx, kv_idx):
    return torch.tensor(True, device=q_idx.device)


def mask_all(b, h, q_idx, kv_idx):
    return torch.tensor(False, device=q_idx.device)


def causal_mask(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


def offset_causal_mask(b, h, q_idx, kv_idx, offset=0):
    return kv_idx <= q_idx + offset


def prefix_mask(b, h, q_idx, kv_idx, prefix_length):
    return kv_idx <= prefix_length


def prefix_causal_mask(b, h, q_idx, kv_idx, prefix_length):
    return causal_mask(b, h, q_idx, kv_idx) | prefix_mask(b, h, q_idx, kv_idx, prefix_length)


def no_prefix_mask(b, h, q_idx, kv_idx, prefix_length):
    return kv_idx > prefix_length


def no_prefix_causal_mask(b, h, q_idx, kv_idx, prefix_length):
    return causal_mask(b, h, q_idx, kv_idx) & no_prefix_mask(b, h, q_idx, kv_idx, prefix_length)        


def flex_attn(queries, keys, values, attention_mask):
    # Perform validation outside the traced function
    # if queries.dtype != keys.dtype or queries.dtype != values.dtype:
    #     raise ValueError("Query, Key, and Value tensors must have the same dtype.")

    # Now call the flex_attention function
    result = flex_attention(queries.transpose(1, 2).contiguous(),
                            keys.transpose(1, 2).contiguous(),
                            values.transpose(1, 2).contiguous(),
                            block_mask=attention_mask)

    return result.transpose(1, 2).contiguous()
