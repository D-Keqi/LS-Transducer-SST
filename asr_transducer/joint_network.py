"""CAAT joint network implementation."""

import torch
from espnet2.asr_transducer.activation import get_activation
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import torch.nn.functional as F
import math
from fairseq.modules import FairseqDropout


class JointNetwork(torch.nn.Module):
    def __init__(
        self,
        output_size: int,
        encoder_size: int,
        decoder_size: int,
        joint_space_size: int = 256,
        joint_activation_type: str = "tanh",
        downsample: int = 32,
        nlayers: int = 6,
        **activation_parameters,
    ):
        super().__init__()
        self.downsample = downsample
        self.lin_out = torch.nn.Linear(1024, output_size)
        self.layers = nn.ModuleList([])
        self.layers.extend([TransformerJointerLayer() for _ in range(nlayers)])

    def _gen_group_mask(self, encoder_out, encoder_padding_mask):
        B, T = encoder_padding_mask.shape
        with torch.no_grad():
            group_num = math.ceil(T / self.downsample)
            group_pos = torch.arange(1, group_num + 1) * self.downsample
            tidx = torch.arange(T)
            group_mask = group_pos.unsqueeze(1) <= tidx.unsqueeze(0)
            group_mask = group_mask.to(encoder_padding_mask.device)
            group_mask_float = encoder_out.new(group_mask.shape).fill_(0)
            group_mask_float = group_mask_float.masked_fill(group_mask, float("-inf"))
            group_mask_float = group_mask_float.unsqueeze(0).repeat(B, 1, 1)
            encout_lengths = (~encoder_padding_mask).sum(1).float()
            group_lengths = (encout_lengths / self.downsample).ceil().long()
        return group_mask_float, group_lengths

    def forward(
        self,
        encoder_out,
        encoder_padding_mask,
        decoder_state: Tensor,
        incremental_state=None,
    ):
        encoder_state = encoder_out
        encoder_padding_mask = encoder_padding_mask
        if self.downsample > 0:
            group_mask, group_lengths = self._gen_group_mask(
                encoder_state, encoder_padding_mask
            )
        else:
            group_mask = None
            group_lengths = decoder_state.new(decoder_state.shape[0]).long().fill_(1)
        x = decoder_state.transpose(0, 1)
        for layer in self.layers:
            x = layer(
                x,
                encoder_state,
                encoder_padding_mask,
                group_attn_mask=group_mask,
                incremental_state=incremental_state,
            )
        # gxtxbxd->bxgxtxd
        x = x.permute(2, 0, 1, 3)
        return self.lin_out(x), group_lengths


class TransformerJointerLayer(nn.Module):
    def __init__(
        self,
        joint_activation_type: str = "tanh",
        num_heads: int = 8,
        hid_dim: int = 2048,
    ):
        super().__init__()
        self.embed_dim = 1024
        self.enc_attn = ExpandMultiheadAttention(self.embed_dim, num_heads, dropout=0)
        self.dropout_module = FairseqDropout(0.3, module_name=self.__class__.__name__)
        self.activation_fn = get_activation(joint_activation_type)
        activation_dropout_p = 0.1
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = False
        self.fc1 = nn.Linear(self.embed_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, self.embed_dim)
        self.attn_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        group_attn_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        residual = x
        if self.normalize_before:
            x = self.attn_layer_norm(x)
        x, _ = self.enc_attn(
            x,
            encoder_out,
            encoder_padding_mask,
            group_attn_mask=group_attn_mask,
            incremental_state=incremental_state,
        )
        x = self.dropout_module(x)
        x = x + residual
        if not self.normalize_before:
            x = self.attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = x + residual
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class ExpandMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.3):
        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def forward(
        self,
        query,
        key,
        key_padding_mask=None,
        group_attn_mask=None,
        incremental_state=None,
    ):
        """
        Args:
            query: TxBxD
            key:   SxBxD
            key_padding_mask: BxS, bool
            group_attn_mask: BxGxS, mask for each group, used to expand attention energy
            incremental_state: cache k,v for next step, need not rollback,reorder only for beam size
        """
        if query.dim() == 3:
            tgt_len, bsz, embed_dim = query.size()
            pre_group_num = 1
            query = query.unsqueeze(0)
        else:
            pre_group_num, tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        src_len = key.shape[0]
        key_processed = False
        k, v = None, None

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                prev_len = saved_state["prev_key"].shape[2]
                if prev_len == src_len:
                    k = saved_state["prev_key"].view(-1, src_len, self.head_dim)
                    v = saved_state["prev_value"].view(-1, src_len, self.head_dim)
                    key_processed = True
        q = (
            self.q_proj(query)
            .view(pre_group_num * tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        q = q * self.scaling
        if not key_processed:
            k = (
                self.k_proj(key)
                .view(src_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            v = (
                self.v_proj(key)
                .view(src_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            if incremental_state is not None:
                saved_state = {}
                saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_value"] = v.view(
                    bsz, self.num_heads, -1, self.head_dim
                )
                incremental_state = self._set_input_buffer(
                    incremental_state, saved_state
                )

        # (b*numheads)*(groupnum*tgtlen)*srclen
        attn_weight = torch.bmm(q, k.transpose(1, 2))
        if key_padding_mask is not None:
            attn_weight = attn_weight.view(
                bsz, self.num_heads, pre_group_num, tgt_len, src_len
            )
            attn_weight = attn_weight.masked_fill(
                key_padding_mask.view(bsz, 1, 1, 1, src_len).to(torch.bool),
                float("-inf"),
            )

        group_num = 1
        if group_attn_mask is not None:
            #  BxGxS, (B*numheads)x(G*T)xS->(B*numheads)xGxTxS
            group_num = group_attn_mask.shape[1]
            assert group_num == pre_group_num or pre_group_num == 1
            attn_weight = attn_weight.view(
                bsz, self.num_heads, pre_group_num, tgt_len, src_len
            )

            group_attn_mask = group_attn_mask.view(bsz, 1, group_num, 1, src_len)
            attn_weight = (
                (attn_weight + group_attn_mask)
                .view(bsz * self.num_heads, group_num, tgt_len, src_len)
                .contiguous()
            )
        else:
            assert pre_group_num == 1
            attn_weight = attn_weight.view(
                bsz * self.num_heads, pre_group_num, tgt_len, src_len
            )

        attn_prob = F.softmax(attn_weight.float(), dim=-1).to(attn_weight)
        attn_prob_drop = self.dropout_module(attn_prob)

        attn_out = torch.einsum("bgts,bsd->bgtd", attn_prob_drop, v)
        attn_out = (
            attn_out.view(bsz, self.num_heads, group_num, tgt_len, self.head_dim)
            .permute(2, 3, 0, 1, 4)
            .contiguous()
            .view(group_num, tgt_len, bsz, embed_dim)
        )
        out = self.out_proj(attn_out)
        return out, attn_prob
