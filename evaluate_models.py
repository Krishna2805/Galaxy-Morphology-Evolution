"""
Evaluate the trained checkpoints on the seed-42 test split -> outputs/test_metrics.json
(the dashboard and README read their accuracy/F1 numbers from this file).

--redshift-control: the confound check for the drift analysis. Redshift is one of the
model's inputs, so it moves embeddings by itself. This takes the same random galaxies,
sets ONLY the redshift input to a low-z vs a mid-z value, and reports how far the mean
embedding moves -> outputs/drift_redshift_control.json.

Usage (from the repo root; needs images_gz2/ and checkpoints/):
    python evaluate_models.py                              # all four models (~35 min on CPU)
    python evaluate_models.py --models multimodal_resnet18 # one model; merges into the JSON
    python evaluate_models.py --redshift-control           # ~1 min on CPU
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from baseline_model import BaselineResNet18
from dataset import MASTER_CSV, GalaxyDataset, create_splits, get_transforms
from dataset_multimodal import MultimodalGalaxyDataset
from efficientnet_model import EfficientNetB0Classifier
from multimodal_model import MultimodalFusionNet

CKPT_DIR, OUT_DIR = "checkpoints", "outputs"
MODELS = {  # name -> (checkpoint file, constructor, uses metadata)
    "baseline_resnet18":          ("baseline_resnet18.pth",          lambda: BaselineResNet18(num_classes=4), False),
    "efficientnet_b0":            ("efficientnet_b0.pth",            lambda: EfficientNetB0Classifier(num_classes=4), False),
    "multimodal_resnet18":        ("multimodal_resnet18.pth",        lambda: MultimodalFusionNet(backbone="resnet18"), True),
    "multimodal_efficientnet_b0": ("multimodal_efficientnet_b0.pth", lambda: MultimodalFusionNet(backbone="efficientnet_b0"), True),
}


def load(ctor, fname):
    model = ctor()
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, fname), map_location="cpu", weights_only=True))
    return model.eval()


def evaluate(names):
    path = os.path.join(OUT_DIR, "test_metrics.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    _, _, test_df = create_splits()
    stats = torch.load(os.path.join(CKPT_DIR, "metadata_stats.pth"), weights_only=True)
    tfm = get_transforms("test")
    for name in names:
        fname, ctor, multimodal = MODELS[name]
        model = load(ctor, fname)
        ds = MultimodalGalaxyDataset(test_df, meta_stats=stats, transform=tfm) if multimodal else GalaxyDataset(test_df, transform=tfm)
        preds, labels = [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=128):
                preds.append(model(*batch[:-1]).argmax(1).numpy())
                labels.append(np.asarray(batch[-1]))
        p, y = np.concatenate(preds), np.concatenate(labels)
        results[name] = {"test_acc": round(float(accuracy_score(y, p)), 4),
                         "test_f1_macro": round(float(f1_score(y, p, average="macro")), 4),
                         "n_test": int(len(y))}
        print(name, results[name], flush=True)
        json.dump(results, open(path, "w"), indent=2)  # saved after each model


def redshift_control(n=256, z_low=0.035, z_mid=0.08):
    master = pd.read_csv(MASTER_CSV)
    stats = torch.load(os.path.join(CKPT_DIR, "metadata_stats.pth"), weights_only=True)
    idx = np.sort(np.random.default_rng(0).choice(len(master), n, replace=False))
    ds = MultimodalGalaxyDataset(master.iloc[idx], meta_stats=stats, transform=get_transforms("test"))
    images = torch.stack([ds[i][0] for i in range(n)])
    results = {"n_galaxies": n, "redshift_input_low": z_low, "redshift_input_mid": z_mid,
               "metric": "euclidean distance between mean embeddings (same images, only redshift input changed)"}
    for bb in ["resnet18", "efficientnet_b0"]:
        model = load(lambda: MultimodalFusionNet(backbone=bb), f"multimodal_{bb}.pth")
        centroids = []
        for z in (z_low, z_mid):
            meta = ds.metadata.clone()
            meta[:, 0] = (z - stats["mean"][0]) / (stats["std"][0] + 1e-8)
            with torch.no_grad():
                emb = torch.cat([model.get_embedding(images[i:i + 64], meta[i:i + 64]) for i in range(0, n, 64)])
            centroids.append(emb.mean(0))
        results[bb] = round(float((centroids[0] - centroids[1]).norm()), 3)
        print(bb, results[bb], flush=True)
    json.dump(results, open(os.path.join(OUT_DIR, "drift_redshift_control.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--redshift-control", action="store_true", help="run only the redshift-input confound check")
    args = ap.parse_args()
    redshift_control() if args.redshift_control else evaluate(args.models)
