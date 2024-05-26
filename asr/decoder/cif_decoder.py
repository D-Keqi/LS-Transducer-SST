from typing import Any, List, Sequence, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface
import torch.nn as nn
import numbers


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

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        self.map = torch.nn.Linear(attention_dim - 1, attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        ys_out_pad: torch.Tensor,
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

        alphas = get_alphas(memory, memory_mask)

        delays = get_delay(alphas, ys_in_lens)
        target_padding_mask = ys_out_pad == -1
        expected_latency = DifferentiableAverageLagging(
            delays, hlens, ys_in_lens, target_padding_mask=target_padding_mask
        )
        latency_loss = expected_latency.clip(min=0).sum()
        latency_loss = latency_loss / bsz
        if self.training:
            assert latency_loss.requires_grad

        decode_length = ys_in_lens
        _alphas, num_output = resize(alphas, decode_length)
        cif_out = cif(memory[:, :, :-1], _alphas)
        cif_out = self.map(cif_out)
        if self.normalize_before:
            cif_out = self.after_norm(cif_out)

        diff = torch.sqrt(torch.pow(num_output - decode_length, 2) + 1e-6).sum()
        diff = diff * text_length / bsz

        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(x, tgt_mask, cif_out, None)

        cif_logits = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return cif_logits, olens, diff, latency_loss

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

        if cache is None:
            _alphas = get_alphas(memory, None)
            cif_out = cif(memory[:, :, :-1], _alphas, infer=True)
            cif_out = self.map(cif_out)
            if self.normalize_before:
                cif_out = self.after_norm(cif_out)

            self.output_layer2 = cif_out.clone()
        else:
            cif_out = self.output_layer2

        now_length = tgt.size(1)
        if now_length <= cif_out.size(1):
            x = self.embed(tgt)
            if cache is None:
                cache = [None] * len(self.decoders)
            new_cache = []
            for c, decoder in zip(cache, self.decoders):
                x, tgt_mask, memory, memory_mask = decoder(
                    x,
                    tgt_mask,
                    cif_out[:, :now_length].repeat(tgt.size(0), 1, 1),
                    None,
                    cache=c,
                )
                new_cache.append(x)

            if self.output_layer is not None:
                y = torch.log_softmax(self.output_layer(x[:, -1]), dim=-1)

            self.cache = new_cache

            return y, new_cache
        else:
            x = self.embed(tgt)
            if cache is None:
                cache = [None] * len(self.decoders)
            new_cache = []
            for c, decoder in zip(cache, self.decoders):
                x, tgt_mask, memory, memory_mask = decoder(
                    x,
                    tgt_mask,
                    cif_out[:, :now_length].repeat(tgt.size(0), 1, 1),
                    None,
                    cache=c,
                    infer=True,
                )
                new_cache.append(x)

            if self.output_layer is not None:
                y = torch.log_softmax(self.output_layer(x[:, -1]), dim=-1)

            self.cache = new_cache

            return y, new_cache

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
        n_layers = len(self.decoders)
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
        # states = states_tmp[1]
        # transpose state of [layer, batch] into [batch, layer]
        if states is not None:
            state_list = [
                [states[i][b] for i in range(n_layers)] for b in range(n_batch)
            ]
        else:
            state_list = None
        # return logp, [states_tmp[0], state_list]
        return logp, state_list


class CIFILDecoder(BaseTransformerDecoder):
    """CIF-IL Decoder"""

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

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: ILDecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


def cif(encoder_output, alphas, threshold=0.999, log=False, infer=False):

    hidden = encoder_output

    device = hidden.device
    B, T, H = hidden.size()

    # loop varss
    integrate = torch.zeros([B], device=device)
    frame = torch.zeros([B, H], device=device)
    # intermediate vars along time
    list_fires = []
    list_frames = []

    for t in range(T):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([B], device=device) - integrate

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place, integrate - torch.ones([B], device=device), integrate
        )
        cur = torch.where(fire_place, distribution_completion, alpha)
        remainds = alpha - cur

        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(
            fire_place[:, None].repeat(1, H), remainds[:, None] * hidden[:, t, :], frame
        )

        if log:
            print(
                "t: {}\t{:.3f} -> {:.3f}|{:.3f} fire: {}".format(
                    t, integrate[log], cur[log], remainds[log], fire_place[log]
                )
            )

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    list_ls = []
    if infer:
        len_labels = torch.round(alphas.sum(-1)).int() + 1
    else:
        len_labels = torch.round(alphas.sum(-1)).int()
    max_label_len = len_labels.max()
    for b in range(B):
        fire = fires[b, :]
        l = torch.index_select(frames[b, :, :], 0, torch.where(fire >= threshold)[0])
        if infer:
            l = torch.cat([l, frames[b, -1:, :]], dim=0)
        pad_l = torch.zeros([max_label_len - l.size(0), H], device=device)
        list_ls.append(torch.cat([l, pad_l], 0))

        if log:
            print(b, l.size(0))

    if log:
        print("fire:\n", fires[log])
        print("fire place:\n", torch.where(fires[log] >= threshold))

    return torch.stack(list_ls, 0)


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
    # scaling
    _alphas = alphas * ((num / _num)[:, None].repeat(1, alphas.size(1)))

    return _alphas, _num


class ILDecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)


    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(ILDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None, infer=False):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            if not infer:
                x_concat = torch.cat(
                    (x, self.src_attn(x, memory, memory, tgt_q_mask)), dim=-1
                )
            else:
                x_concat = torch.cat(
                    (x, self.src_attn(x, memory, memory, None)), dim=-1
                )
            x = residual + self.concat_linear2(x_concat)
        else:
            if not infer:
                x = residual + self.dropout(
                    self.src_attn(x, memory, memory, tgt_q_mask)
                )
            else:
                x = residual + self.dropout(self.src_attn(x, memory, memory, None))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask


def get_delay(alpha, target_lengths):

    B, S = alpha.shape
    feat_lengths = target_lengths.long()
    T = feat_lengths.max()
    beta = 1

    csum = alpha.cumsum(-1)
    with torch.no_grad():
        # indices used for scattering
        right_idx = csum.floor().long().clip(max=T)
        left_idx = right_idx.roll(1, dims=1)
        left_idx[:, 0] = 0

        # count # of fires from each source
        fire_num = right_idx - left_idx
        extra_weights = (fire_num - 1).clip(min=0)

    delay = alpha.new_zeros((B, T + 1))
    source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(alpha)
    zero = alpha.new_zeros((1,))

    # right scatter
    fire_mask = fire_num > 0
    right_weight = torch.where(
        fire_mask, csum - right_idx.type_as(alpha) * beta, zero
    ).type_as(alpha)
    delay.scatter_add_(1, right_idx, right_weight * source_range / beta)

    # left scatter
    left_weight = (alpha - right_weight - extra_weights.type_as(alpha) * beta).type_as(
        alpha
    )

    delay.scatter_add_(1, left_idx, left_weight * source_range / beta)
    # extra scatters
    if extra_weights.ge(0).any():
        extra_steps = extra_weights.max().item()
        tgt_idx = left_idx
        # src_feats = input * beta
        for _ in range(extra_steps):
            tgt_idx = (tgt_idx + 1).clip(max=T)
            # (B, S, 1)
            src_mask = extra_weights > 0
            delay.scatter_add_(1, tgt_idx, source_range * src_mask)
            extra_weights -= 1

    return delay[:, :T]


def latency_metric(func):
    def prepare_latency_metric(
        delays,
        src_lens,
        ref_lens=None,
        target_padding_mask=None,
    ):
        """
        delays: bsz, tgt_len
        src_lens: bsz
        target_padding_mask: bsz, tgt_len
        """
        if isinstance(delays, list):
            delays = torch.FloatTensor(delays).unsqueeze(0)

        if len(delays.size()) == 1:
            delays = delays.view(1, -1)

        if isinstance(src_lens, list):
            src_lens = torch.FloatTensor(src_lens)
        if isinstance(src_lens, numbers.Number):
            src_lens = torch.FloatTensor([src_lens])
        if len(src_lens.size()) == 1:
            src_lens = src_lens.view(-1, 1)
        src_lens = src_lens.type_as(delays)

        if ref_lens is not None:
            if isinstance(ref_lens, list):
                ref_lens = torch.FloatTensor(ref_lens)
            if isinstance(ref_lens, numbers.Number):
                ref_lens = torch.FloatTensor([ref_lens])
            if len(ref_lens.size()) == 1:
                ref_lens = ref_lens.view(-1, 1)
            ref_lens = ref_lens.type_as(delays)

        if target_padding_mask is not None:
            tgt_lens = delays.size(-1) - target_padding_mask.sum(dim=1)
            delays = delays.masked_fill(target_padding_mask, 0)
        else:
            tgt_lens = torch.ones_like(src_lens) * delays.size(1)

        tgt_lens = tgt_lens.view(-1, 1)

        return delays, src_lens, tgt_lens, ref_lens, target_padding_mask

    def latency_wrapper(delays, src_lens, ref_lens=None, target_padding_mask=None):
        delays, src_lens, tgt_lens, ref_lens, target_padding_mask = (
            prepare_latency_metric(delays, src_lens, ref_lens, target_padding_mask)
        )
        return func(delays, src_lens, tgt_lens, ref_lens, target_padding_mask)

    return latency_wrapper


@latency_metric
def AverageProportion(
    delays, src_lens, tgt_lens, ref_lens=None, target_padding_mask=None
):
    """
    Function to calculate Average Proportion from
    Can neural machine translation do simultaneous translation?
    (https://arxiv.org/abs/1606.02012)
    Delays are monotonic steps, range from 1 to src_len.
    Give src x tgt y, AP is calculated as:
    AP = 1 / (|x||y]) sum_i^|Y| delays_i
    """
    if target_padding_mask is not None:
        AP = torch.sum(delays.masked_fill(target_padding_mask, 0), dim=1, keepdim=True)
    else:
        AP = torch.sum(delays, dim=1, keepdim=True)

    AP = AP / (src_lens * tgt_lens)
    return AP.squeeze(1)


@latency_metric
def AverageLagging(delays, src_lens, tgt_lens, ref_lens=None, target_padding_mask=None):
    """
    Function to calculate Average Lagging from
    STACL: Simultaneous Translation with Implicit Anticipation
    and Controllable Latency using Prefix-to-Prefix Framework
    (https://arxiv.org/abs/1810.08398)
    Delays are monotonic steps, range from 1 to src_len.
    Give src x tgt y, AP is calculated as:
    AL = 1 / tau sum_i^tau delays_i - (i - 1) / gamma
    Where
    gamma = |y| / |x|
    tau = argmin_i(delays_i = |x|)

    When reference was given, |y| would be the reference length
    """
    bsz, max_tgt_len = delays.size()
    if ref_lens is not None:
        max_tgt_len = ref_lens.max().long()
        tgt_lens = ref_lens

    # tau = argmin_i(delays_i = |x|)
    # Only consider the delays that has already larger than src_lens
    lagging_padding_mask = delays >= src_lens
    # Padding one token at beginning to consider at least one delays that
    # larget than src_lens
    lagging_padding_mask = torch.nn.functional.pad(lagging_padding_mask, (1, 0))[:, :-1]

    if target_padding_mask is not None:
        lagging_padding_mask = lagging_padding_mask.masked_fill(
            target_padding_mask, True
        )

    # oracle delays are the delay for the oracle system which goes diagonally
    oracle_delays = (
        (
            torch.arange(max_tgt_len)
            .unsqueeze(0)
            .type_as(delays)
            .expand([bsz, max_tgt_len])
        )
        * src_lens
        / tgt_lens
    )

    if delays.size(1) < max_tgt_len:
        oracle_delays = oracle_delays[:, : delays.size(1)]

    if delays.size(1) > max_tgt_len:
        oracle_delays = torch.cat(
            [
                oracle_delays,
                oracle_delays[:, -1]
                * oracle_delays.new_ones(
                    [delays.size(0), delays.size(1) - max_tgt_len]
                ),
            ],
            dim=1,
        )

    lagging = delays - oracle_delays
    lagging = lagging.masked_fill(lagging_padding_mask, 0)

    # tau is the cut-off step
    tau = (1 - lagging_padding_mask.type_as(lagging)).sum(dim=1)
    AL = lagging.sum(dim=1) / tau

    return AL


@latency_metric
def DifferentiableAverageLagging(
    delays, src_lens, tgt_lens, ref_len=None, target_padding_mask=None
):
    """
    Function to calculate Differentiable Average Lagging from
    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation
    (https://arxiv.org/abs/1906.05218)
    """

    _, max_tgt_len = delays.size()

    gamma = tgt_lens / src_lens
    new_delays = torch.zeros_like(delays)

    for i in range(max_tgt_len):
        if i == 0:
            new_delays[:, i] = delays[:, i]
        else:
            new_delays[:, i] = (
                torch.cat(
                    [
                        new_delays[:, i - 1].unsqueeze(1) + 1 / gamma,
                        delays[:, i].unsqueeze(1),
                    ],
                    dim=1,
                )
                .max(dim=1)
                .values
            )

    DAL = (
        new_delays
        - torch.arange(max_tgt_len).unsqueeze(0).type_as(delays).expand_as(delays)
        / gamma
    )
    if target_padding_mask is not None:
        DAL = DAL.masked_fill(target_padding_mask, 0)

    DAL = DAL.sum(dim=1, keepdim=True) / tgt_lens

    return DAL.squeeze(1)
