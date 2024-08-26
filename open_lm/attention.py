from functools import partial, lru_cache
from typing import Tuple, Union

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, or_masks, and_masks, _DEFAULT_SPARSE_BLOCK_SIZE


@lru_cache()
def create_block_mask_cached(mask_function, B, H, M, N, device="cuda", BLOCK_SIZE: Union[int, Tuple[int, int]] = _DEFAULT_SPARSE_BLOCK_SIZE, _compile=False, offset=0, prefix_length=0):
    if offset > 0:
        mask_function = offset_wrapper(mask_function, offset)
    if prefix_length > 0:
        mask_function = prefix_wrapper(mask_function, prefix_length)

    block_mask = create_block_mask(mask_function, B, H, M, N, device=device, BLOCK_SIZE=BLOCK_SIZE, _compile=_compile)
    return block_mask


def offset_wrapper(mask_function, offset):
    def wrapped(b, h, q_idx, kv_idx):
        return mask_function(b, h, q_idx + offset, kv_idx)
    return wrapped


def prefix_wrapper(mask_function, prefix_length):
    def wrapped(b, h, q_idx, kv_idx):
        return mask_function(b, h, q_idx, kv_idx) | prefix_mask(b, h, q_idx, kv_idx, prefix_length=prefix_length)
    return wrapped


def no_prefix_wrapper(mask_function, prefix_length):
    def wrapped(b, h, q_idx, kv_idx):
        return mask_function(b, h, q_idx, kv_idx) & no_prefix_mask(b, h, q_idx, kv_idx, prefix_length=prefix_length)
    return wrapped


def no_mask(b, h, q_idx, kv_idx):
    return torch.tensor(True, device=q_idx.device)


def mask_all(b, h, q_idx, kv_idx):
    return torch.tensor(False, device=q_idx.device)


def causal_mask(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx

def prefix_causal_mask(b, h, q_idx, kv_idx, *, prefix_length):
    return torch.logical_or(kv_idx <= q_idx, kv_idx <= prefix_length)

def prefix_mask(b, h, q_idx, kv_idx, *, prefix_length):
    return kv_idx <= prefix_length


def no_prefix_mask(b, h, q_idx, kv_idx, *, prefix_length):
    return kv_idx > prefix_length  


def flex_attn(queries, keys, values, block_mask, BLOCK_SIZE: Union[int, Tuple[int, int]] = _DEFAULT_SPARSE_BLOCK_SIZE, _compile=False):
    # Build the block mask
    # if mask_function is not None:
    #     BLOCK_SIZE = min(BLOCK_SIZE, queries.size(1))
    #     BLOCK_SIZE = min(BLOCK_SIZE, keys.size(1))
    #     query_offset = max(keys.size(1) - queries.size(1), 0)
    #     block_mask = create_block_mask_cached(mask_function, queries.size(0), queries.size(2), queries.size(1), keys.size(1), BLOCK_SIZE=BLOCK_SIZE, _compile=_compile, offset=query_offset)
    # else:
    #     block_mask = None

    # Run flex_attention function
    result = flex_attention(queries.transpose(1, 2).contiguous(),
                            keys.transpose(1, 2).contiguous(),
                            values.transpose(1, 2).contiguous(),
                            block_mask=block_mask)

    return result.transpose(1, 2).contiguous()
