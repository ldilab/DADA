import json
from pathlib import Path

import hydra
import torch
import typer
from omegaconf import DictConfig
from torch import tensor
from tqdm.rich import tqdm
from transformers import AutoModel, AutoModelForMaskedLM

from src.datamodule.modules.dataloader import GenericDataLoader
from src.models.trainer import RetrievalModel, HybridRetrievalModel
from src.predict import get_prediction_model


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="infer.yaml")
def amnesia(cfg: DictConfig):
    ckpt_model, object_dict = get_prediction_model(cfg)
    device = torch.device(f'cuda:{cfg["trainer"]["devices"][0]}')

    model = ckpt_model.dense
    model.eval()
    model.to(device)

    model_name = object_dict["model"].model.bert.config.name_or_path
    mlm_head = AutoModelForMaskedLM.from_pretrained(model_name).cls
    mlm_head = mlm_head.eval()
    mlm_head.to(device)

    root_dir = Path("/workspace")
    beir_dir = root_dir / "beir"

    dataname = cfg["datamodule"]["test_datasets"][0]
    amnesia_dir = root_dir / "amnesia"
    expr_name = cfg["student"]
    target_dir = amnesia_dir / expr_name / dataname
    target_dir.mkdir(exist_ok=True, parents=True)
    mlm_pth = target_dir / "mlm.jsonl"

    data_name = cfg["datamodule"]["test_datasets"][0]
    batch_size = cfg["datamodule"]["test_batch_size"]
    tokenizer = hydra.utils.instantiate(cfg["datamodule"]["tokenizer"]["dense"])

    # loader
    (corpus, _) = GenericDataLoader(
        data_folder=str((beir_dir / data_name).absolute()),
    ).load_custom(which="corpus", split="test")

    corpus = list(corpus.items())
    for k in tqdm(range(0, len(corpus) // batch_size),
                  desc=f"MLM: {dataname}",
                  unit_scale=1000000):
        batch_corpus = []
        batch_ids = []
        batch_idxs = range(k * batch_size, (k + 1) * batch_size)
        for i in batch_idxs:
            doc = corpus[i]
            batch_ids.append(doc[0])
            batch_corpus.append(
                f'{doc[1]["title"]}. {doc[1]["text"]}'
            )

        with torch.no_grad():
            docs_tokenized = tokenizer.tokenize_passage(
                batch_corpus
            )
            docs_tokenized = docs_tokenized.to(device)

            docs_embed = model.encode_passage(
                docs_tokenized
            )
            docs_mlm = mlm_head(docs_embed["encoded_embeddings"])

        with mlm_pth.open("a") as f:
            for did, doc_mlm in zip(batch_ids, docs_mlm):
                doc_mlm = doc_mlm.tolist()
                doc_mlm = json.dumps(
                    {
                        "id": did,
                        "mlm": doc_mlm
                    }
                )
                f.write(doc_mlm)
                f.write("\n")

    import requests
    requests.post(
        'https://hooks.slack.com/services/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',
        json={'text': f'{dataname} mlm Done'}
    )

if __name__ == '__main__':
    amnesia()
