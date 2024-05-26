from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.transducer.beam_search_transducer import ExtendedHypothesis, Hypothesis
from espnet2.lm.transformer_lm import TransformerLM


class CAATPredictor(AbsDecoder):
    """(RNN-)Transducer decoder module.

    Args:
        vocab_size: Output dimension.
        layers_type: (RNN-)Decoder layers type.
        num_layers: Number of decoder layers.
        hidden_size: Number of decoder units per layer.
        dropout: Dropout rate for decoder layers.
        dropout_embed: Dropout rate for embedding layer.
        embed_pad: Embed/Blank symbol ID.

    """

    def __init__(
        self,
        vocab_size: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int = 320,
        dropout: float = 0.0,
        dropout_embed: float = 0.0,
        embed_pad: int = 0,
    ):
        assert check_argument_types()

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported: rnn_type={rnn_type}")

        super().__init__()

        self.lm = TransformerLM(
            vocab_size=vocab_size,
            pos_enc=None,
            embed_unit=1024,
            att_unit=1024,
            head=8,
            unit=2048,
            layer=6,
            dropout_rate=0.0,
        )
        del self.lm.decoder

        self.dlayers = num_layers
        self.dunits = hidden_size
        self.dtype = rnn_type
        self.odim = vocab_size

        self.ignore_id = -1
        self.blank_id = embed_pad

        self.device = next(self.parameters()).device

    def set_device(self, device: torch.device):
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def init_state(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, Optional[torch.tensor]]:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Initial decoder hidden states. ((N, B, D_dec), (N, B, D_dec))

        """
        return [None] * len(self.lm.encoder.encoders)
        h_n = torch.zeros(
            self.dlayers,
            batch_size,
            self.dunits,
            device=self.device,
        )

        if self.dtype == "lstm":
            c_n = torch.zeros(
                self.dlayers,
                batch_size,
                self.dunits,
                device=self.device,
            )

            return (h_n, c_n)

        return (h_n, None)

    def rnn_forward(
        self,
        sequence: torch.Tensor,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Encode source label sequences.

        Args:
            sequence: RNN input sequences. (B, D_emb)
            state: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))

        Returns:
            sequence: RNN output sequences. (B, D_dec)
            (h_next, c_next): Decoder hidden states. (N, B, D_dec), (N, B, D_dec))

        """
        h_prev, c_prev = state
        h_next, c_next = self.init_state(sequence.size(0))

        for layer in range(self.dlayers):
            if self.dtype == "lstm":
                sequence, (
                    h_next[layer : layer + 1],
                    c_next[layer : layer + 1],
                ) = self.decoder[layer](
                    sequence, hx=(h_prev[layer : layer + 1], c_prev[layer : layer + 1])
                )
            else:
                sequence, h_next[layer : layer + 1] = self.decoder[layer](
                    sequence, hx=h_prev[layer : layer + 1]
                )

            sequence = self.dropout_dec[layer](sequence)

        return sequence, (h_next, c_next)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences. (B, L)

        Returns:
            dec_out: Decoder output sequences. (B, T, U, D_dec)

        """

        tgt = labels.clone()
        x = self.lm.embed(tgt)
        mask = self.lm._target_mask(tgt)
        y, _ = self.lm.encoder(x, mask)
        return y

    def score(
        self, hyp: Hypothesis, cache: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """One-step forward hypothesis.

        Args:
            hyp: Hypothesis.
            cache: Pairs of (dec_out, state) for each label sequence. (key)

        Returns:
            dec_out: Decoder output sequence. (1, D_dec)
            new_state: Decoder hidden states. ((N, 1, D_dec), (N, 1, D_dec))
            label: Label ID for LM. (1,)

        """
        tgt = torch.LongTensor(hyp.yseq, device="cpu").unsqueeze(0)
        tgt = tgt.to(self.device)

        label = torch.full((1, 1), hyp.yseq[-1], dtype=torch.long, device=self.device)

        str_labels = "_".join(list(map(str, hyp.yseq)))

        if str_labels in cache:
            dec_out, dec_state = cache[str_labels]
        else:
            h, _, dec_state = self.lm.encoder.forward_one_step(
                self.lm.embed(tgt), self.lm._target_mask(tgt), cache=hyp.dec_state
            )
            cache[str_labels] = (dec_out, dec_state)

        return dec_out, dec_state, label[0]

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
        dec_states: Tuple[torch.Tensor, Optional[torch.Tensor]],
        cache: Dict[str, Any],
        use_lm: bool,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
            cache: Pairs of (dec_out, dec_states) for each label sequences. (keys)
            use_lm: Whether to compute label ID sequences for LM.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec)
            dec_states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
            lm_labels: Label ID sequences for LM. (B,)

        """
        final_batch = len(hyps)

        process = []
        done = [None] * final_batch

        for i, hyp in enumerate(hyps):
            str_labels = "_".join(list(map(str, hyp.yseq)))

            if str_labels in cache:
                done[i] = cache[str_labels]
            else:
                process.append((str_labels, hyp.yseq, hyp.dec_state))

        if process:

            tmp = []
            tmp_length = []
            for p in process:
                tmp.append(p[1])
                tmp_length.append(len(p[1]))
            max_length = max(tmp_length)
            ys = torch.zeros(len(process), max_length, device="cpu")
            ys_prior = torch.zeros(len(process), max_length, device="cpu")
            for i in range(ys.shape[0]):
                ys[i][: tmp_length[i]] = torch.Tensor(tmp[i])
                ys_prior[i][-tmp_length[i] :] = torch.Tensor(tmp[i])
            ys = ys.long().to(self.device)
            ys_prior = ys_prior.long().to(self.device)

            ys_emb = self.lm.embed(ys)
            ys_emb_real = ys_emb.new_zeros(ys_emb.shape)
            tmp_pad = ys_emb.new_zeros(1, 1024)

            # states = []
            cha = []
            for i in range(ys.shape[0]):
                ys_emb_real[i][-tmp_length[i] :] = ys_emb[i][: tmp_length[i]]
                cha.append(max_length - tmp_length[i])

            mask = self.lm._target_mask(ys_prior)

            states = [p[2] for p in process]

            n_batch = len(ys)
            n_layers = len(self.lm.encoder.encoders)
            if states[0] is None:
                batch_state = None
            else:
                # transpose state of [batch, layer] into [layer, batch]
                # batch_state = None
                batch_state = [
                    torch.stack(
                        [
                            (
                                states[b][i]
                                if cha[b] == 0
                                else torch.cat(
                                    [tmp_pad.repeat(cha[b], 1), states[b][i]], dim=0
                                )
                            )
                            for b in range(n_batch)
                        ]
                    )
                    for i in range(n_layers)
                ]

            dec_out_tmp, _, states = self.lm.encoder.forward_one_step(
                ys_emb_real, mask, cache=batch_state
            )
            dec_out = dec_out_tmp[:, -1]

            # transpose state of [layer, batch] into [batch, layer]
            dec_states = [
                [states[i][b][-tmp_length[b] :] for i in range(n_layers)]
                for b in range(n_batch)
            ]

        if use_lm:
            lm_labels = torch.LongTensor(
                [h.yseq[-1] for h in hyps], device=self.device
            ).view(final_batch, 1)

            return dec_out, dec_states, lm_labels

        return dec_out, dec_states, None
