from typing import Any, List, Sequence, Tuple
import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.scorer_interface import BatchScorerInterface
from fairseq import (
    options,
)


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
        self.decoders = None
        self.stop_decode = False
        self.chunk_prune = False
        self.is_final = True

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
        masks = make_pad_mask(hlens).to(hlens.device)
        enc_out = {
            "encoder_out": [hs_pad.transpose(0, 1)],
            "encoder_padding_mask": [masks] if masks is not None else [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }

        decoder_out = self.decoders(
            prev_output_tokens=ys_in_pad, encoder_out=enc_out  # hs_pad
        )
        decoder_out = decoder_out[0]
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        return decoder_out, tgt_mask.sum(1)

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
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

        enc_out = {
            "encoder_out": [memory.transpose(0, 1)],
            "encoder_padding_mask": [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }
        decoder_out = self.decoders(
            prev_output_tokens=tgt, encoder_out=enc_out  # hs_pad
        )

        alpha_list = [item["beta"] for item in decoder_out[1].attn_list]
        bsz, num_heads, tgt_len, src_len = alpha_list[0].size()
        # assert bsz == 1
        assert tgt_len == tgt.size(1)

        alpha_all = torch.cat(alpha_list, dim=1).view(-1, tgt_len, src_len)
        alpha_all = alpha_all != 0
        alpha_length = alpha_all[:, -1, :].sum(-1)
        assert alpha_length.max() == alpha_length.min()
        assert src_len == memory.size(1)

        assert alpha_length.max().item() == self.prefix * (
            tgt.size(1) + self.waitk_lagging - 1
        ) or alpha_length.max().item() == memory.size(1)
        if self.prefix * (tgt.size(1) + self.waitk_lagging - 1) > memory.size(1):
            self.stop_decode = True
        else:
            self.stop_decode = False

        num_chunk = (
            int(
                (self.prefix * (tgt.size(1) + self.waitk_lagging - 1) - 1)
                / self.chunk_size
            )
            + 1
        )
        next_num_chunk = (
            int(
                (self.prefix * (tgt.size(1) + self.waitk_lagging) - 1) / self.chunk_size
            )
            + 1
        )

        if self.is_final and self.chunk_size * num_chunk >= memory.size(1):
            self.chunk_prune = False
        elif next_num_chunk > num_chunk:
            self.chunk_prune = True
        else:
            self.chunk_prune = False

        decoder_out = decoder_out[0]

        y = torch.log_softmax(decoder_out[:, -1], dim=-1)
        return y, y

    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
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
        if states[0] is None:
            batch_state = None
        else:
            batch_state = states

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        return logp, states


class WaitkDecoder(BaseTransformerDecoder):
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

        parser = options.get_training_parser()
        add_parse(parser)
        args, _ = parser.parse_known_args(None)

        base_architecture(args)
        args.simul_type = "waitk_fixed_pre_decision"

        # import from fairseq.examples
        from examples.simultaneous_translation.models.transformer_monotonic_attention import (
            TransformerMonotonicDecoder,
        )

        self.decoders = TransformerMonotonicDecoder(
            args,
            [None] * vocab_size,
            torch.nn.Embedding(vocab_size, 1024, padding_idx=0),
        )
        self.waitk_lagging = args.waitk_lagging
        self.prefix = args.fixed_pre_decision_ratio
        self.chunk_size = chunk_size


def base_architecture(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.max_source_positions = getattr(args, "max_source_positions", 3000)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.conv_out_channels = getattr(args, "conv_out_channels", args.encoder_embed_dim)
    args.simul_type = getattr(args, "simul_type", "waitk_fixed_pre_decision")
    args.waitk_lagging = getattr(args, "waitk_lagging", 5)
    args.fixed_pre_decision_ratio = getattr(args, "fixed_pre_decision_ratio", 18)


def add_parse(parser):
    parser.add_argument(
        "--no-mass-preservation",
        action="store_false",
        dest="mass_preservation",
        help="Do not stay on the last token when decoding",
    )
    parser.add_argument(
        "--mass-preservation",
        action="store_true",
        dest="mass_preservation",
        help="Stay on the last token when decoding",
    )
    parser.set_defaults(mass_preservation=True)
    parser.add_argument(
        "--noise-var", type=float, default=1.0, help="Variance of discretness noise"
    )
    parser.add_argument(
        "--noise-mean", type=float, default=0.0, help="Mean of discretness noise"
    )
    parser.add_argument(
        "--noise-type", type=str, default="flat", help="Type of discretness noise"
    )
    parser.add_argument(
        "--energy-bias", action="store_true", default=False, help="Bias for energy"
    )
    parser.add_argument(
        "--energy-bias-init",
        type=float,
        default=-2.0,
        help="Initial value of the bias for energy",
    )
    parser.add_argument(
        "--attention-eps",
        type=float,
        default=1e-6,
        help="Epsilon when calculating expected attention",
    )
    parser.add_argument(
        "--fixed-pre-decision-ratio",
        default=18,
        type=int,
        help=(
            "Ratio for the fixed pre-decision,"
            "indicating how many encoder steps will start"
            "simultaneous decision making process."
        ),
    )
    parser.add_argument(
        "--fixed-pre-decision-type",
        default="average",
        choices=["average", "last"],
        help="Pooling type",
    )
    parser.add_argument(
        "--fixed-pre-decision-pad-threshold",
        type=float,
        default=0.3,
        help="If a part of the sequence has pad"
        ",the threshold the pooled part is a pad.",
    )
