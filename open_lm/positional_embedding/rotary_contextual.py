# NOTE: 08/31/23, this class is copied from xformers as there is currently a bug related to which channel dim the rotary embedding is applied to.
# when the upstream issue is fixed, this file should be deleted. To track progress, see this issue: https://github.com/facebookresearch/xformers/issues/841

# taken from: https://github.com/facebookresearch/xformers/blob/748c159096d4f9fcfe3eaf22801e5aed4777210b/xformers/components/positional_embedding/rotary.py
from typing import Tuple

import torch
from einops import rearrange
from fla.ops.linear_attn import fused_chunk_linear_attn

from open_lm.positional_embedding.rotary import apply_rotary_pos_emb


class RotaryEmbeddingContextual(torch.nn.Module):
    """
    Variation of the rotary position embeddings from RoFormer_ (Su et. al).

    This version differs from the original in that it uses a custom positon vector 
    instead of assuming that the position is the same as the index of the token.

    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim_model: int, seq_len: int, layer_id: int = 0, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.dim_model = dim_model
        self.register_buffer("inv_freq", torch.zeros(self.dim_model // 2))

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        self.seq_len = seq_len
        self.encoding_step_size = 32
        self.layer_id = layer_id
        # self.embed_v = torch.nn.Linear(dim_model, 16)
        self.reset_parameters()

    def reset_parameters(self):
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim_model, 2).float() / self.dim_model))
        self._update_cos_sin_tables(self.seq_len)

    def _update_cos_sin_tables(self, seq_len: int = None, device: torch.device = None, dtype: torch.dtype = None):
        # If no seq_len is provided, use the cached one
        # If the seq_len is smaller than the cached one it is included in the cached one so no need to update
        if seq_len is None or seq_len < self._seq_len_cached:
            seq_len = self._seq_len_cached

        # Reset the tables if the sequence length has increased,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len//self.encoding_step_size, device=device, dtype=torch.float32)*self.encoding_step_size
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            self._cos_cached_step = emb.cos()[None, :, None, :].to(dtype)
            self._sin_cached_step = emb.sin()[None, :, None, :].to(dtype)

            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            self._cos_cached = emb.cos()[None, :, None, :].to(dtype)
            self._sin_cached = emb.sin()[None, :, None, :].to(dtype)

    
    def contextual_positions(self, q, k, v):
        q = rearrange(q, "b n h d -> b h n d")
        k = rearrange(k, "b m h d -> b h m d")
        v = rearrange(v, "b m h d -> b h m d")
        # We actually want last dim=1 but linear attn wants last dim>=16
        # sum_v = torch.sigmoid(self.embed_v(v))
        sum_v = torch.ones_like(k[..., :16])
        q = torch.relu(q)
        k = torch.relu(k)
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-6)
        k = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
        
        positions, _ = fused_chunk_linear_attn(q.to(sum_v.dtype), k.to(sum_v.dtype), sum_v, normalize=False)
        positions = rearrange(positions.mean(-1, keepdim=True), "b h n d -> b n h d")
        positions = (positions / self.encoding_step_size).clamp(0, self._seq_len_cached//self.encoding_step_size-1)
        return positions

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v:  torch.Tensor, offset: int, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the rotary embedding to the queries and keys
        Args:
            q: The queries tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k: The keys tensor of shape [batch_size, seq_len, num_heads, head_dim]
            v: The values tensor of shape [batch_size, seq_len, num_heads, head_dim]
            positions: The positions tensor of shape [batch_size, seq_len, num_heads, 1]
        """
        self._update_cos_sin_tables(k.shape[1] + offset, device=k.device, dtype=k.dtype)
        
        if self.layer_id % 2 == 0:
            return (
                apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, offset),
                apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, offset),
            )
        else:
            if positions is None:
                positions = self.contextual_positions(q, k, v)
            else:
                positions = (positions/self.encoding_step_size).clamp(0, k.shape[1]-1)

            positions_c = torch.ceil(positions).to(torch.long)
            positions_f = torch.floor(positions).to(torch.long)
            # Gather the cos and sin values for the positions and interpolate for the float positions
            sin_positions_c = torch.gather(self._sin_cached_step.repeat(positions_c.shape[0], 1, positions_c.shape[2], 1), 1, positions_c)
            cos_positions_f = torch.gather(self._cos_cached_step.repeat(positions_c.shape[0], 1, positions_c.shape[2], 1), 1, positions_f)
            cos_positions_c = torch.gather(self._cos_cached_step.repeat(positions_c.shape[0], 1, positions_c.shape[2], 1), 1, positions_c)
            sin_positions_f = torch.gather(self._sin_cached_step.repeat(positions_c.shape[0], 1, positions_c.shape[2], 1), 1, positions_f)
            cos_positions = cos_positions_f + (positions - positions_f) * (cos_positions_c - cos_positions_f)
            sin_positions = sin_positions_f + (positions - positions_f) * (sin_positions_c - sin_positions_f)
            return (
                apply_rotary_pos_emb(q, cos_positions, sin_positions, offset),
                apply_rotary_pos_emb(k, cos_positions, sin_positions, offset),
            )


class RotaryContextualWithCast(RotaryEmbeddingContextual):
    def forward(self, q, k, v, offset: int=0, positions: torch.Tensor=None):
        q, k = super().forward(q, k, v, offset, positions)
        return q.to(v.dtype), k.to(v.dtype), v