# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from mosestokenizer import MosesDetokenizer
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.utils import entrypoint

from espnet2.bin.st_inference import Speech2Text
from espnet2.bin.caat_st_inference import CAATSpeech2Text
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
import pickle
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
import numpy as np
from espnet.nets.beam_search import Hypothesis
from typeguard import check_argument_types, check_return_type
from espnet2.torch_utils.device_funcs import to_device
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis
from espnet2.asr.transducer.beam_search_transducer import (
    ExtendedHypothesis as ExtTransHypothesis,
)


@entrypoint
class DummyAgent(SpeechToTextAgent):
    """
    DummyAgent operates in an offline mode.
    Waits until all source is read to run inference.
    """

    def __init__(self, args):
        super().__init__(args)
        kwargs = vars(args)

        logging.basicConfig(
            level=kwargs["log_level"],
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

        if kwargs["ngpu"] >= 1:
            device = "cuda"
        else:
            device = "cpu"

        # 1. Set random-seed
        set_all_random_seed(kwargs["seed"])

        speech2text_kwargs = dict(
            st_train_config=kwargs["st_train_config"],
            st_model_file=kwargs["st_model_file"],
            lm_train_config=kwargs["lm_train_config"],
            lm_file=kwargs["lm_file"],
            ngram_file=kwargs["ngram_file"],
            token_type=kwargs["token_type"],
            bpemodel=kwargs["bpemodel"],
            device=device,
            maxlenratio=kwargs["maxlenratio"],
            minlenratio=kwargs["minlenratio"],
            batch_size=kwargs["batch_size"],
            dtype=kwargs["dtype"],
            beam_size=kwargs["beam_size"],
            lm_weight=kwargs["lm_weight"],
            ngram_weight=kwargs["ngram_weight"],
            penalty=kwargs["penalty"],
            nbest=kwargs["nbest"],
            enh_s2t_task=kwargs["enh_s2t_task"],
        )

        if kwargs["rnnt"]:
            transducer_conf = {
                "search_type": "default",
                "score_norm": True,
            }
            speech2text_kwargs["transducer_conf"] = transducer_conf
        else:
            transducer_conf = None
            if kwargs["aed_type"] == "lst":
                speech2text_kwargs["stctc_weight"] = kwargs["ctc_weight"]

        if kwargs["rnnt"]:
            self.speech2text = CAATSpeech2Text.from_pretrained(
                model_tag=kwargs["model_tag"],
                **speech2text_kwargs,
            )
        else:
            self.speech2text = Speech2Text.from_pretrained(
                model_tag=kwargs["model_tag"],
                **speech2text_kwargs,
            )

        self.sim_chunk_length = kwargs["sim_chunk_length"]
        self.backend = kwargs["backend"]
        self.rnnt = kwargs["rnnt"]
        self.aed_type = kwargs["aed_type"]
        self.token_delay = kwargs["token_delay"]
        self.lang = kwargs["lang"]
        self.recompute = kwargs["recompute"]
        self.chunk_decay = kwargs["chunk_decay"]
        self.n_chunks = 0
        self.clean()
        self.word_list = (
            pickle.load(open("german_dict.obj", "rb"))
            if kwargs["use_word_list"]
            else None
        )

    @staticmethod
    def add_args(parser):
        # Note(kamo): Use '_' instead of '-' as separator.
        # '-' is confusing if written in yaml.
        parser.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )

        parser.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--dtype",
            default="float32",
            choices=["float16", "float32", "float64"],
            help="Data type",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )

        group = parser.add_argument_group("The model configuration related")
        group.add_argument(
            "--st_train_config",
            type=str,
            help="ST training configuration",
        )
        group.add_argument(
            "--st_model_file",
            type=str,
            help="ST model parameter file",
        )
        group.add_argument(
            "--lm_train_config",
            type=str,
            help="LM training configuration",
        )
        group.add_argument(
            "--src_lm_train_config",
            type=str,
            help="LM training configuration",
        )
        group.add_argument(
            "--lm_file",
            type=str,
            help="LM parameter file",
        )
        group.add_argument(
            "--src_lm_file",
            type=str,
            help="LM parameter file",
        )
        group.add_argument(
            "--word_lm_train_config",
            type=str,
            help="Word LM training configuration",
        )
        group.add_argument(
            "--src_word_lm_train_config",
            type=str,
            help="Word LM training configuration",
        )
        group.add_argument(
            "--word_lm_file",
            type=str,
            help="Word LM parameter file",
        )
        group.add_argument(
            "--src_word_lm_file",
            type=str,
            help="Word LM parameter file",
        )
        group.add_argument(
            "--ngram_file",
            type=str,
            help="N-gram parameter file",
        )
        group.add_argument(
            "--src_ngram_file",
            type=str,
            help="N-gram parameter file",
        )
        group.add_argument(
            "--model_tag",
            type=str,
            help="Pretrained model tag. If specify this option, *_train_config and "
            "*_file will be overwritten",
        )
        group.add_argument(
            "--enh_s2t_task",
            type=str2bool,
            default=False,
            help="enhancement and asr joint model",
        )

        group = parser.add_argument_group("Beam-search related")
        group.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="The batch size for inference",
        )
        group.add_argument(
            "--nbest", type=int, default=1, help="Output N-best hypotheses"
        )
        group.add_argument(
            "--asr_nbest", type=int, default=1, help="Output N-best hypotheses"
        )
        group.add_argument("--beam_size", type=int, default=20, help="Beam size")
        group.add_argument("--asr_beam_size", type=int, default=20, help="Beam size")
        group.add_argument(
            "--penalty", type=float, default=0.0, help="Insertion penalty"
        )
        group.add_argument(
            "--asr_penalty", type=float, default=0.0, help="Insertion penalty"
        )
        group.add_argument(
            "--maxlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain max output length. "
            "If maxlenratio=0.0 (default), it uses a end-detect "
            "function "
            "to automatically find maximum hypothesis lengths."
            "If maxlenratio<0.0, its absolute value is interpreted"
            "as a constant max output length",
        )
        group.add_argument(
            "--asr_maxlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain max output length. "
            "If maxlenratio=0.0 (default), it uses a end-detect "
            "function "
            "to automatically find maximum hypothesis lengths."
            "If maxlenratio<0.0, its absolute value is interpreted"
            "as a constant max output length",
        )
        group.add_argument(
            "--minlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain min output length",
        )
        group.add_argument(
            "--asr_minlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain min output length",
        )
        group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
        group.add_argument(
            "--asr_lm_weight", type=float, default=1.0, help="RNNLM weight"
        )
        group.add_argument(
            "--ngram_weight", type=float, default=0.9, help="ngram weight"
        )
        group.add_argument(
            "--asr_ngram_weight", type=float, default=0.9, help="ngram weight"
        )
        group.add_argument(
            "--ctc_weight", type=float, default=0.0, help="ST CTC weight"
        )
        group.add_argument(
            "--asr_ctc_weight", type=float, default=0.3, help="ASR CTC weight"
        )

        group.add_argument(
            "--transducer_conf",
            default=None,
            help="The keyword arguments for transducer beam search.",
        )

        group = parser.add_argument_group("Text converter related")
        group.add_argument(
            "--token_type",
            type=str_or_none,
            default=None,
            choices=["char", "bpe", None],
            help="The token type for ST model. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--src_token_type",
            type=str_or_none,
            default=None,
            choices=["char", "bpe", None],
            help="The token type for ST model. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model path of sentencepiece. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--src_bpemodel",
            type=str_or_none,
            default=None,
            help="The model path of sentencepiece. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--ctc_greedy",
            type=str2bool,
            default=False,
        )

        group.add_argument(
            "--sim_chunk_length",
            type=int,
            default=0,
            help="The length of one chunk, to which speech will be "
            "divided for evalution of streaming processing.",
        )
        group.add_argument(
            "--disable_repetition_detection", type=str2bool, default=False
        )
        group.add_argument(
            "--encoded_feat_length_limit",
            type=int,
            default=0,
            help="Limit the lengths of the encoded feature" "to input to the decoder.",
        )
        group.add_argument(
            "--decoder_text_length_limit",
            type=int,
            default=0,
            help="Limit the lengths of the text" "to input to the decoder.",
        )

        group.add_argument(
            "--backend",
            type=str,
            default="offline",
            help="Limit the lengths of the text" "to input to the decoder.",
        )
        group.add_argument(
            "--time_sync",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--incremental_decode",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--blank_penalty",
            type=float,
            default=1.0,
        )
        group.add_argument(
            "--hold_n",
            type=int,
            default=0,
        )
        group.add_argument(
            "--token_delay",
            type=str2bool,
            default=True,
        )
        group.add_argument(
            "--rnnt",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--aed_type",
            type=str,
            default="lst",
        )
        group.add_argument(
            "--lang",
            type=str,
            default="de",
        )
        group.add_argument("--hugging_face_decoder", type=str2bool, default=False)
        group.add_argument(
            "--recompute",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--chunk_decay",
            type=float,
            default=1.0,
        )
        group.add_argument(
            "--use_word_list",
            type=str2bool,
            default=False,
        )

        return parser

    def clean(self):
        self.processed_index = -1
        self.maxlen = 0
        self.prev_prediction = ""
        self.prev_token_prediction = []
        self.n_chunks = 0
        self.speech2text.maxlenratio = 0

    @torch.no_grad()
    def lst_decode(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        prev_token_prediction=[],
        is_final: bool = False,
        beta: int = 0,
        chunk_size: int = 64,
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input speech data
            beta: Adjust AIF threshold
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.speech2text.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.speech2text.device)

        # b. Forward Encoder
        enc, _ = self.speech2text.st_model.encode(**batch)
        if not is_final:
            if enc.shape[1] != chunk_size * self.n_chunks:
                logging.warning(
                    f"\n enc and chunk are: {enc.shape} !!{enc.shape[1]} and {64*self.n_chunks} \n"
                )
            assert enc.shape[1] == chunk_size * self.n_chunks
        assert len(enc) == 1, len(enc)

        alpha_sum = (0.95 * torch.sigmoid(enc[0, :, -1]) + 0.05).sum(-1).item() - beta

        if is_final:
            self.speech2text.maxlenratio = 0
            self.speech2text.beam_search.part_scorers["stctc"].is_final = True
        else:
            if alpha_sum < len(prev_token_prediction) + 1:
                return []
            self.speech2text.maxlenratio = int(alpha_sum) / enc[0].size(0) + 1e-5
            self.speech2text.beam_search.part_scorers["stctc"].is_final = False

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.speech2text.beam_search(
            x=enc[0],
            maxlenratio=self.speech2text.maxlenratio,
            minlenratio=self.speech2text.minlenratio,
        )
        nbest_hyps = nbest_hyps[: self.speech2text.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            token_int = hyp.yseq[1:-1].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.speech2text.converter.ids2tokens(token_int)

            right_token = token
            if not is_final:
                right_token = token[:0]
                for i in range(1, len(token) + 1):
                    if token[-i].endswith("▁"):
                        if i != 1:
                            if token[-i + 1] != "&":
                                right_token = token[:-i] + [token[-i]]
                                break
                        else:
                            right_token = token[:-i] + [token[-i]]
                            break

            if self.speech2text.tokenizer is not None:
                text = self.speech2text.tokenizer.tokens2text(right_token)  # token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results

    @torch.no_grad()
    def waitk_decode(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        prev_token_prediction=[],
        is_final: bool = False,
        chunk_size: int = 64,
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.speech2text.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.speech2text.device)

        # b. Forward Encoder
        enc, _ = self.speech2text.st_model.encode(**batch)
        if not is_final:
            if enc.shape[1] != chunk_size * self.n_chunks:
                logging.warning(
                    f"\n enc and chunk are: {enc.shape} !!{enc.shape[1]} and {64*self.n_chunks} \n"
                )
            assert enc.shape[1] == chunk_size * self.n_chunks
        assert len(enc) == 1, len(enc)

        if is_final:
            self.speech2text.beam_search.full_scorers["decoder"].is_final = True
        else:
            self.speech2text.beam_search.full_scorers["decoder"].is_final = False

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.speech2text.beam_search(
            x=enc[0],
            maxlenratio=self.speech2text.maxlenratio,
            minlenratio=self.speech2text.minlenratio,
            is_final=is_final,
        )
        if nbest_hyps == None:
            return None
        nbest_hyps = nbest_hyps[: self.speech2text.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            token_int = hyp.yseq[1:-1].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.speech2text.converter.ids2tokens(token_int)

            right_token = token
            if not is_final:
                right_token = token[:0]
                for i in range(1, len(token) + 1):
                    if token[-i].endswith("▁"):
                        if i != 1:
                            if token[-i + 1] != "&":
                                right_token = token[:-i] + [token[-i]]
                                break
                        else:
                            right_token = token[:-i] + [token[-i]]
                            break

            if self.speech2text.tokenizer is not None:
                text = self.speech2text.tokenizer.tokens2text(right_token)  # token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results

    @torch.no_grad()
    def rnnt_decode(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        is_final: bool = False,
        chunk_size: int = 64,
    ) -> List[
        Tuple[
            Optional[str],
            List[str],
            List[int],
            Union[Hypothesis, ExtTransHypothesis, TransHypothesis],
        ]
    ]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.speech2text.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.speech2text.device)

        # b. Forward Encoder
        enc, _ = self.speech2text.st_model.encode(**batch)
        if not is_final:
            if enc.shape[1] != chunk_size * self.n_chunks:
                logging.warning(
                    f"\n enc and chunk are: {enc.shape} !!{enc.shape[1]} and {64*self.n_chunks} \n"
                )
            assert enc.shape[1] == chunk_size * self.n_chunks
        assert len(enc) == 1, len(enc)

        results = self._decode_single_sample(enc[0], is_final)
        assert check_return_type(results)
        return results

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=enc[0], maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
        )
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            token_int = hyp.yseq[1:-1].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results

    def _decode_single_sample(self, enc: torch.Tensor, is_final):

        assert self.speech2text.beam_search_transducer
        if self.speech2text.beam_search_transducer:
            logging.info("encoder output length: " + str(enc.shape[0]))
            nbest_hyps = self.speech2text.beam_search_transducer(enc)

            best = nbest_hyps[0]
            logging.info(f"total log probability: {best.score:.2f}")
            logging.info(
                f"normalized log probability: {best.score / len(best.yseq):.2f}"
            )
            logging.info(
                "best hypo: "
                + "".join(self.speech2text.converter.ids2tokens(best.yseq[1:]))
                + "\n"
            )
        else:
            nbest_hyps = self.speech2text.beam_search(
                x=enc, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            )

        nbest_hyps = nbest_hyps[: self.speech2text.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

            # remove sos/eos and get results
            last_pos = None if self.speech2text.st_model.use_transducer_decoder else -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.speech2text.converter.ids2tokens(token_int)
            right_token = token
            if not is_final:
                for i in range(1, len(token) + 1):
                    if token[-i].endswith("▁"):
                        if i != 1:
                            if token[-i + 1] != "&":
                                right_token = token[:-i] + [token[-i]]
                                break
                        else:
                            right_token = token[:-i] + [token[-i]]
                            break

            if self.speech2text.tokenizer is not None:
                text = self.speech2text.tokenizer.tokens2text(right_token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        return results

    def policy(self):
        # dummy offline policy
        if self.backend == "offline":
            if self.states.source_finished:
                results = self.speech2text(torch.tensor(self.states.source))
                if self.speech2text.st_model.use_multidecoder:
                    prediction = results[0][0][
                        0
                    ]  # multidecoder result is in this format
                else:
                    prediction = results[0][0]
                return WriteAction(prediction, finished=True)
            else:
                return ReadAction()

        # streaming policy. takes running beam search hyp as an incremental output
        else:
            unread_length = len(self.states.source) - self.processed_index - 1
            if self.n_chunks > 0:
                chunk_length = self.sim_chunk_length // self.chunk_decay
            else:
                chunk_length = self.sim_chunk_length

            if self.processed_index == -1:
                decide_chunk_length = chunk_length + 80
            else:
                decide_chunk_length = chunk_length

            if unread_length >= decide_chunk_length or self.states.source_finished:
                # 80 = (0.025-0.02)*16000
                print("chunk_length:", str(chunk_length))
                # logging.warning(f"chunk_length: {str(chunk_length)}")
                self.n_chunks += 1

                assert self.recompute
                if self.recompute:
                    speech = torch.tensor(self.states.source)
                else:
                    speech = torch.tensor(
                        self.states.source[self.processed_index + 1 :]
                    )
                try:
                    if self.rnnt:
                        results = self.rnnt_decode(
                            speech=speech, is_final=self.states.source_finished
                        )
                    else:
                        if self.aed_type == "lst":
                            results = self.lst_decode(
                                speech=speech,
                                prev_token_prediction=self.prev_token_prediction,
                                is_final=self.states.source_finished,
                            )
                        elif self.aed_type == "waitk":
                            results = self.waitk_decode(
                                speech=speech,
                                prev_token_prediction=self.prev_token_prediction,
                                is_final=self.states.source_finished,
                            )
                except TooShortUttError:
                    print("skipping inference for too short input")
                    results = [[""]]

                self.processed_index = len(self.states.source) - 1
                if results == None:
                    return ReadAction()

                if not self.states.source_finished:
                    if len(results) > 0:
                        prediction = results[0][0]
                        token_prediction = results[0][1]
                    else:
                        return ReadAction()
                else:
                    if len(results) > 0:
                        prediction = results[0][0]
                        token_prediction = results[0][1]
                    else:
                        prediction = self.prev_prediction
                        token_prediction = self.prev_token_prediction

                if prediction != self.prev_prediction or self.states.source_finished:
                    self.prev_prediction = prediction
                    self.prev_token_prediction = token_prediction
                    prediction = MosesDetokenizer(self.lang)(prediction.split(" "))

                    unwritten_length = len(prediction) - len(
                        "".join(self.states.target)
                    )
                else:
                    unwritten_length = 0

                if self.states.source_finished:
                    self.clean()

                if unwritten_length > 0:
                    ret = prediction[-unwritten_length:]
                    print(self.processed_index, ret)
                    return WriteAction(ret, finished=self.states.source_finished)
                elif self.states.source_finished:
                    return WriteAction("", finished=self.states.source_finished)

            return ReadAction()
