# [Official, ACL2024-Findings] DADA: Distribution Aware Domain Adaptation

## 🗂 Project Structure
The directory structure of new project looks like this:
```
├── configs                   <- Hydra configuration files
│   ├── callbacks                <- Callbacks configs
│   ├── datamodule               <- Datamodule configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── tokenizer                <- Tokenizer configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data (will be generated automatically)
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── scripts                <- Shell scripts
│
├── distilColBERT                <- Source code
│   ├── data2tensor              <- Tokenizers
│   ├── datamodule               <- Lightning datamodules
│   ├── losses                   <- Loss functions
│   ├── metrics                  <- torchmetrics based metrics
│   ├── models                   <- Lightning models
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── .gitignore                <- List of files ignored by git
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
└── README.md
```
**Be aware that this repository loads BEIR dataset from `/workspace/beir` and GPL dataset from `/workspace/gpl`**


## 🚀  Quickstart

```bash
# clone project
git clone
cd dada

# [OPTIONAL] create conda environment
python3 -m venv dada_env
source dada_env/bin/activate

# install requirements
pip install -r requirements.txt

# run training
python train.py

# run evaluation
python eval.py
```

## 💡 How to

### 🏭 Reproduce
~~~bash
# Parameters
CUDA_GPUS = 0,1,2,3
DATASET = scifact
MODEL_TYPE = cond # "coco", "tasb"
VALIDATION_INTERVAL = 10_000
SANITY_CHECK = 32
MODEL = "cond"

# Train / Eval
## GPL Baseline 
python3 -m src.train \
    experiment=gpl-baseline-${MODEL} \
    trainer.devices=[${CUDA_GPUS}] \
    datamodule.train_dataset_type=gpl datamodule.train_dataset=${DATASET} \
    datamodule.test_dataset_type=beir datamodule.test_datasets=[${DATASET}] \
    trainer.num_sanity_val_steps=${SANITY_CHECK} trainer.val_check_interval=${VALIDATION_INTERVAL}
python3 -m src.eval \
    experiment=eval-gpl-baseline-${MODEL} \
    trainer.devices=[${CUDA_GPUS}] \
    ckpt_path="'$ckpt_path'" \
    method_name="gpl" \
    model_type=${MODEL} \
    dataset=${DATASET}

## DADA
### Model: CoCondener
python3 -m src.train \
    experiment=hybrid-curr-semantic-idf-norm-${MODEL} \
    trainer.devices=[${CUDA_GPUS}] \
    datamodule.train_dataset_type=gpl datamodule.train_dataset=${DATASET} \
    datamodule.test_dataset_type=beir datamodule.test_datasets=[$DATASET] dataset=$DATASET \
    method_name=dada model_type=${MODEL_TYPE} trainer.val_check_interval=${VALIDATION_INTERVAL}
python3 -m src.eval \
    experiment=eval-hybrid-curr-semantic-idf-norm-${MODEL} \
    trainer.devices=[${CUDA_GPUS}] \
    ckpt_path="'$ckpt_path'" \
    method_name="dada" \
    model_type=${MODEL} \
    dataset=${DATASET}
~~~

### 👓 Read Code
This repository is consist of mainly two parts, `configs` and `src (the main source code)`
- Configs
    - Configs are mostly written beforehand. Therefore, some yaml files may be difficult to understand.
    - Only config folder that you need to understand is `experiment`, `model` and `tokenizer`
        - `experiment` folder includes experiment settings
        - `model` folder contains Pytorch-Lightning module configuration (eg. retrieval_module)
        - `tokenizer` folder shows the tokenizer setting. By default, it is set as `BaseTokenizer`.
- distilColBERT (source code)
    - `data2tensor` defines how data (text) is being tokenized with huggingface tokenizer.
    - `datamodule` defines pytorch-lightning datamodule which loading GPL data (140k steps, 32 batch) for training and BEIR data for evaluation.
    - `losses` define loss functions (objectives).
        - `modules` are custom sub-functions for loss function (eg. MarginMSELoss, KL-Divergence)
    - `metrics` define metrics for either train or evaluation. (evaluation metrics written in advance)
    - `model` defines retrieval model.
        - `modules` are layers used by retrieval model.
    - `utils` is for logging, wrappers and hydra instantiation written from template.

### 🖋 Write code (workflow)

If you are trying to experiment new model, follow these steps.

1. define your model in `src.model` (if new sub-layers are needed, write in `modules`).
2. write the objectives in `src.losses` (if new sub-functions are needed, write in `modules`).
3. describe your setting in `configs.model` with appropriate name.
    - This name is used on experiment configs
    - if help needed
        - try to read `default.yaml` for which key-values are set.
        - try to read `colbert.yaml` as an example.
4. illustrate your experiment configuration on `configs.experiment` with relevant name.
    - This name is used on running train and eval.
    - if help needed try to read `example.yaml`.
5. run train: `python train.py experiment={experiment name} datamodule.data_name={beir dataset name you want to use}`
6. run eval:  `python eval.py experiment={experiment name} datamodule.data_name={beir dataset name} ckpt_path={checkpoint path}`

## ?? To Cite this work
```
@inproceedings{dada,
  title={DADA: Distribution Aware Domain Adaptation},
  author
    year={2024},
    booktitle={Findings of ACL},
}
```

## 📚  References
- Repo Template
    - Thanks to "ashleve" providing [Repository Template](https://github.com/ashleve/lightning-hydra-template)
- Weight and Bias (W&B)
  ```
  @misc{wandb,
  title =        {Experiment Tracking with Weights and Biases},
  year =         {2020},
  note =         {Software available from wandb.com},
  url=           {https://www.wandb.com/},
  author =       {Biewald, Lukas},
  }
  ```
- Hydra
  ```
  @Misc{Yadan2019Hydra,
  author =       {Omry Yadan},
  title =        {Hydra - A framework for elegantly configuring complex applications},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/facebookresearch/hydra}
  }
  ```
- GPL
  ```
  @article{wang2021gpl,
    title = "GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval",
    author = "Kexin Wang and Nandan Thakur and Nils Reimers and Iryna Gurevych",
    journal= "arXiv preprint arXiv:2112.07577",
    month = "4",
    year = "2021",
    url = "https://arxiv.org/abs/2112.07577",
  }
  ```
- BEIR
  ```
  @inproceedings{
    thakur2021beir,
    title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
    author={Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={https://openreview.net/forum?id=wCu6T5xFjeJ}
  }
  ```
- huggingface
  ```
  @inproceedings{wolf-etal-2020-transformers,
      title = "Transformers: State-of-the-Art Natural Language Processing",
      author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
      booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
      month = oct,
      year = "2020",
      address = "Online",
      publisher = "Association for Computational Linguistics",
      url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
      pages = "38--45"
  }
  ```
- SBERT
  ```
  @inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
  }
  ```
