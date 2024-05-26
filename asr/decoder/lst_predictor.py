from typing import Any, List, Sequence, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.scorer_interface import BatchScorerInterface
import torch.nn as nn
from espnet2.tasks.lm import LMTask
import math


class MyMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MyMultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = key.view(n_batch, -1, self.h, self.d_k)
        v = value.view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return x

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class BaseTransformerDecoder(AbsDecoder, BatchScorerInterface):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        self.linear_k = nn.Linear(attention_dim - 1, attention_dim)
        self.linear_v = nn.Linear(attention_dim - 1, attention_dim)

        self.multi_att = MyMultiHeadedAttention(8, 1024, 0.0)

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        bsz = ys_in_pad.shape[0]
        text_length = ys_in_pad.shape[1]
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = make_pad_mask(hlens, maxlen=memory.size(1)).to(memory.device)

        # AIF----
        alphas = get_alphas(memory, memory_mask)
        decode_length = ys_in_lens - 1
        _, num_output = resize(alphas, decode_length)
        aif_mask = tgt.new_zeros(bsz, text_length, memory.size(1))

        accum_alphas = alphas.clone().unsqueeze(-2).repeat(1, memory.size(1), 1)
        ret = torch.ones(
            memory.size(1), memory.size(1), device=memory.device, dtype=torch.bool
        )
        sub_mask = torch.tril(ret, out=ret).unsqueeze(0)
        accum_alphas = (sub_mask * accum_alphas).sum(-1)
        accum_alphas2 = accum_alphas.clone()

        # full chunk: fully utilise the chunk range
        src_length = memory.size(1)
        assert memory.size(1) == accum_alphas.size(1)
        chunk_num = int(src_length / self.chunk_size)
        if src_length % self.chunk_size != 0:
            chunk_num = chunk_num + 1
        for ch in range(chunk_num):
            accum_alphas2[
                :, self.chunk_size * ch : self.chunk_size * ch + self.chunk_size
            ] = accum_alphas[:, self.chunk_size * ch].unsqueeze(-1)

        # Prepare AIF mask
        ones = tgt.new_ones(alphas.shape)
        zeros = tgt.new_zeros(alphas.shape)
        for j in range(text_length):
            aif_mask[:, j] = torch.where(
                accum_alphas2 <= (j + 1 + self.beta), ones, zeros
            )

        padding_mask = ~memory_mask.unsqueeze(1).repeat(1, text_length, 1)
        aif_mask = aif_mask * padding_mask

        diff = torch.sqrt(torch.pow(num_output - decode_length, 2) + 1e-6).sum()
        diff = diff * text_length / bsz

        x = self.lm.embed(tgt)
        mask = self.lm._target_mask(tgt)
        h, _, inter_out = self.lm.encoder(x, mask)
        y = self.lm.decoder(h)
        queries = inter_out[0]

        keys = self.linear_k(memory[:, :, :-1])
        values = self.linear_v(memory[:, :, :-1])
        a_hidden = self.multi_att(queries, keys, values, aif_mask)

        a_logits = self.output_layer(a_hidden)

        olens = tgt_mask.sum(1)
        return y + a_logits, olens, diff

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache=None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        if cache == None:
            self.cache_idx = 1

        # batch decoding
        h, _, states = self.lm.encoder.forward_one_step(
            self.lm.embed(tgt), self.lm._target_mask(tgt), cache=cache
        )
        assert len(states) == 6
        if self.lm.encoder.normalize_before:
            h_tmp = self.lm.encoder.after_norm(states[2])[:, -1:]
        else:
            h_tmp = states[2][:, -1:]
        h = self.lm.decoder(h[:, -1])

        now_length = tgt.size(1)
        alphas = get_alphas(memory, None)
        for i in range(self.cache_idx, memory.size(1) + 1):
            if alphas[0, :i].sum(-1) > (now_length + self.beta):
                tmp = i
                if tmp % self.chunk_size == 0:
                    self.cache_idx = tmp
                else:
                    self.cache_idx = min(
                        self.chunk_size * (int(tmp / self.chunk_size) + 1),
                        memory.size(1),
                    )
                break
            if i == memory.size(1):
                self.cache_idx = memory.size(1)

        a_hidden = self.multi_att(
            h_tmp,
            self.linear_k(memory[:, :, :-1])[:, : self.cache_idx],
            self.linear_v(memory[:, :, :-1])[:, : self.cache_idx],
            None,
        )
        a_logits = self.output_layer(a_hidden)

        y = a_logits[:, 0] + h
        y = torch.log_softmax(y, dim=-1)

        return y, states

    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states, xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        n_batch = len(ys)
        n_layers = len(self.lm.encoder.encoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list


class LSTPredictor(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        chunk_size: int = 64,
        beta: int = 0,
        lm_path: str = "exp/lm_train_lm_en_de_bpe4000/",
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        self.chunk_size = chunk_size
        # adjust AIF threshold
        self.beta = beta
        lm_train_config = lm_path + "config.yaml"
        lm_file = lm_path + "valid.loss.ave.pth"

        lm, lm_train_args = LMTask.build_model_from_file(
            lm_train_config, lm_file, "cpu"
        )
        self.lm = lm.lm
        self.lm.encoder.intermediate_layers = [3]


def get_alphas(encoder_output, memory_mask):
    if memory_mask is not None:
        padding_mask = memory_mask
        alphas = encoder_output[:, :, -1]
        alphas = 0.95 * torch.sigmoid(alphas) + 0.05
        alphas = alphas * (~padding_mask).float()
    else:
        alphas = encoder_output[:, :, -1]
        alphas = 0.95 * torch.sigmoid(alphas) + 0.05

    return alphas


def resize(alphas, target_lengths, threshold=0.999):
    """
    alpha in thresh=1.0 | (0.0, +0.21)
    """
    # sum
    _num = alphas.sum(-1)
    num = target_lengths.float()
    _alphas = alphas * ((num / _num)[:, None].repeat(1, alphas.size(1)))

    return _alphas, _num
