"""Search algorithms for CAAT models."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


@dataclass
class Hypothesis:
    """Default hypothesis definition for Transducer search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        torch.Tensor,
    ]
    lm_state: Union[Dict[str, Any], List[Any]] = None


@dataclass
class ExtendedHypothesis(Hypothesis):
    """Extended hypothesis definition for NSC beam search and mAES."""

    dec_out: List[torch.Tensor] = None
    lm_scores: torch.Tensor = None


class BeamSearchTransducer:
    """Beam search implementation for Transducer."""

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        beam_size: int,
        lm: torch.nn.Module = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        max_sym_exp: int = 2,
        u_max: int = 50,
        nstep: int = 1,
        prefix_alpha: int = 1,
        expansion_gamma: int = 2.3,
        expansion_beta: int = 2,
        score_norm: bool = True,
        nbest: int = 1,
        token_list: Optional[List[str]] = None,
    ):
        """Initialize Transducer search module.

        Args:
            decoder: Decoder module.
            joint_network: Joint network module.
            beam_size: Beam size.
            lm: LM class.
            lm_weight: LM weight for soft fusion.
            search_type: Search algorithm to use during inference.
            max_sym_exp: Number of maximum symbol expansions at each time step. (TSD)
            u_max: Maximum output sequence length. (ALSD)
            nstep: Number of maximum expansion steps at each time step. (NSC/mAES)
            prefix_alpha: Maximum prefix length in prefix search. (NSC/mAES)
            expansion_beta:
              Number of additional candidates for expanded hypotheses selection. (mAES)
            expansion_gamma: Allowed logp difference for prune-by-value method. (mAES)
            score_norm: Normalize final scores by length. ("default")
            nbest: Number of final hypothesis.

        """
        self.decoder = decoder
        self.joint_network = joint_network

        self.beam_size = beam_size
        self.hidden_size = decoder.dunits
        self.vocab_size = decoder.odim

        self.sos = self.vocab_size - 1
        self.token_list = token_list

        self.blank_id = decoder.blank_id

        assert search_type == "default"
        self.search_algorithm = self.default_beam_search

        self.use_lm = lm is not None
        self.lm = lm
        self.lm_weight = lm_weight

        self.score_norm = score_norm
        self.nbest = nbest

    def __call__(
        self, enc_out: torch.Tensor
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis]]:
        """Perform beam search.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(enc_out.device)

        nbest_hyps = self.search_algorithm(enc_out)

        return nbest_hyps

    def sort_nbest(
        self, hyps: Union[List[Hypothesis], List[ExtendedHypothesis]]
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis]]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: Hypothesis.

        Return:
            hyps: Sorted hypothesis.

        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[: self.nbest]

    def default_beam_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam_10 = min(self.beam_size, self.vocab_size)
        beam_k_10 = min(beam_10, (self.vocab_size - 1))

        beam_1 = min(1, self.vocab_size)
        beam_k_1 = min(beam_1, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)]
        cache = {}
        cache_lm = {}
        chunk_size = 64

        for nn, enc_out_t in enumerate(enc_out):

            if (nn + 1) % chunk_size == 0:
                beam = beam_1
                beam_k = beam_k_1
                hyps = [max(kept_hyps, key=lambda x: x.score)]
                kept_hyps = []
            else:
                beam = beam_10
                beam_k = beam_k_10
                hyps = kept_hyps
                kept_hyps = []

            # hyps = kept_hyps
            # kept_hyps = []

            if self.token_list is not None:
                logging.debug(
                    "\n"
                    + "\n".join(
                        [
                            "hypo: "
                            + "".join([self.token_list[x] for x in hyp.yseq[1:]])
                            + f", score: {round(float(hyp.score), 2)}"
                            for hyp in sorted(hyps, key=lambda x: x.score, reverse=True)
                        ]
                    )
                )

            while True:
                # for n in range(10):
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                dec_outt, state, lm_tokens = self.decoder.score(max_hyp, cache)

                lm_logits = None
                if isinstance(dec_outt, tuple):
                    dec_out = dec_outt[0]
                    lm_logits = dec_outt[1]
                    # logging.warning(f"lm_logits shape: {lm_logits.shape}")
                    # logging.warning(f"dec_out shape: {dec_out.shape}")
                    # assert lm_logits.shape[0]==1000
                else:
                    dec_out = dec_outt

                if lm_logits is not None:
                    tmp = self.joint_network(enc_out_t, dec_out).clone()
                    tmp[0] = tmp[0] + tmp[0]
                    tt = lm_logits[1:].clone()
                    tmp[1:] = tmp[1:] + tt
                    logp = torch.log_softmax(
                        tmp,
                        dim=-1,
                    )
                else:
                    enc_out_lens = torch.LongTensor([nn + 1]).to(enc_out.device)
                    memory_mask = (make_pad_mask(enc_out_lens, maxlen=nn + 1)).to(
                        enc_out.device
                    )

                    l_out, _ = self.joint_network(
                        enc_out[: nn + 1].unsqueeze(1), memory_mask, dec_out
                    )
                    logp = torch.log_softmax(
                        l_out[0, -1, -1],
                        dim=-1,
                    )
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )

                if self.use_lm:
                    if tuple(max_hyp.yseq) not in cache_lm:
                        lm_scores, lm_state = self.lm.score(
                            torch.LongTensor(
                                [self.sos] + max_hyp.yseq[1:],
                                device=self.decoder.device,
                            ),
                            max_hyp.lm_state,
                            None,
                        )
                        cache_lm[tuple(max_hyp.yseq)] = (lm_scores, lm_state)
                    else:
                        lm_scores, lm_state = cache_lm[tuple(max_hyp.yseq)]
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break
                if len(max_hyp.yseq) > 0.5 * len(enc_out):
                    break

        return self.sort_nbest(kept_hyps)
