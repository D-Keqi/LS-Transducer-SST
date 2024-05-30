# LS-Transducer-SST
This is the implementation of "Label-Synchronous Neural Transducer for E2E Simultaneous Speech Translation"

## Implemented
- [Wait-k](https://aclanthology.org/2020.aacl-main.58.pdf)
- [CIF-IL](https://www.isca-archive.org/interspeech_2022/chang22f_interspeech.pdf)
- [CAAT](https://aclanthology.org/2021.emnlp-main.4.pdf)
- LS-Transducer-SST

## Setup
1. Install ESPnet (v.202211) and dependencies following [Installation](https://espnet.github.io/espnet/installation.html)
2. Install Fairseq package following [Installation](https://github.com/espnet/espnet/blob/v.202211/tools/installers/install_fairseq.sh)
3. Install SimulEval (v.1.1.4) toolkit following [Installation](https://github.com/facebookresearch/SimulEval)
4. (Optional) Install CAAT loss following [Installation](https://github.com/danliu2/caat) if you want to train a CAAT model.

## Pipeline
1. Pre-process the data following ESPnet examples: [Fisher-CallHome Spanish](https://github.com/espnet/espnet/tree/v.202211/egs2/fisher_callhome_spanish/st1) and [MuST-C](https://github.com/espnet/espnet/tree/v.202211/egs2/must_c/st1)
2. (Optional) ASR models can be trained to initialise the encoder of ST models following ESPnet ASR examples.
3. (Optional) Sequence-level knowledge distillation ([KD](https://aclanthology.org/D16-1139.pdf)) can be used following [example](https://github.com/danliu2/caat) to augment the training data.
4. Train the ST models using the code given in this project. A pre-trained LS-Transducer-SST model is provided [here](https://huggingface.co/Kiko98/LS-Transducer-S)
5. Evaluate the translation quality and latency: bash ./egs2/local/simuleval_*.sh

## Citation
If the paper or the code helps you, please cite the paper in the following format :
```
@inproceedings{deng2024lst,
  title={Label-Synchronous Neural Transducer for E2E Simultaneous Speech Translation},
  author={Deng, Keqi and Woodland, Philip C},
  booktitle={Proceedings of the 62st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2024},
  publisher = "Association for Computational Linguistics",
  address = "Bangkok, Thailand",
}
```
