import csv
import hashlib
import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

FEW_SHOT_NS = [1, 5, 10, 20, 50]
N_TRIALS = 10
MIN_TEST_PER_CLASS = 6    # minimum val examples to include a class
MIN_TRAIN_PER_CLASS = 15  # need enough to support up to 20-shot sampling

# Included despite fewer val examples — high-priority rare disease.
FORCE_INCLUDE = {"pulmonary fibrosis"}

COLORS = {
    "SSL pretrained": "#2196F3",
    "ImageNet": "#4CAF50",
    "Random init": "#FF5722",
}

EXCLUDE_LABELS = {
    # uninformative
    "normal", "unchanged", "exclude", "suboptimal study",
    # implanted hardware / devices
    "nsg tube", "endotracheal tube", "tracheostomy tube", "chest drain tube",
    "central venous catheter", "central venous catheter via jugular vein",
    "central venous catheter via subclavian vein",
    "central venous catheter via umbilical vein",
    "reservoir central venous catheter", "ventriculoperitoneal drain tube",
    "pacemaker", "dual chamber device", "single chamber device",
    "electrical device", "artificial heart valve", "artificial mitral heart valve",
    "artificial aortic heart valve", "heart valve calcified", "aortic endoprosthesis",
    "humeral prosthesis", "mammary prosthesis", "osteosynthesis material",
    "suture material", "metal",
    # surgical history
    "sternotomy", "surgery", "surgery neck", "surgery breast", "surgery lung",
    "surgery heart", "surgery humeral", "mastectomy", "post radiotherapy changes",
    # artifacts / normal variants
    "nipple shadow", "end on vessel", "dai",
    # common diseases with large public labeled datasets
    "pneumonia", "pleural effusion", "cardiomegaly", "nodule", "atelectasis",
    "laminar atelectasis", "emphysema", "scoliosis", "copd signs", "aortic elongation",
    # non-specific radiological descriptors
    "increased density", "chronic changes", "infiltrates", "calcified densities",
    "bronchovascular markings", "pseudonodule", "vascular hilar enlargement",
    "apical pleural thickening", "costophrenic angle blunting", "hilar enlargement",
    "air trapping", "hypoexpansion", "volume loss", "hyperinflated lung",
    # structural variants and age-related changes
    "vertebral degenerative changes", "vertebral anterior compression",
    "vertebral compression", "kyphosis", "axial hyperostosis",
    "diaphragmatic eventration", "azygos lobe", "pectum excavatum",
    "aortic atheromatosis", "aortic button enlargement",
    "descendent aortic elongation", "ascendent aortic elongation",
    "supra aortic elongation", "non axial articular degenerative changes",
    "sclerotic bone lesion",
}

# Computed from 2000 sampled PadChest images (grayscale, repeated to 3 channels).
_XRAY_MEAN = [0.4776, 0.4776, 0.4776]
_XRAY_STD  = [0.2674, 0.2674, 0.2674]

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_XRAY_MEAN, std=_XRAY_STD),
])


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return _TRANSFORM(Image.open(self.paths[idx]).convert("RGB"))


def _parse_labels(raw: str) -> list[str]:
    # PadChest stores labels as single-quoted Python list literals, e.g. ['finding', 'other']
    return json.loads(raw.strip().replace("'", '"'))


def load_padchest_splits(
    data_dir: Path, val_frac: float = 0.2, seed: int = 42
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Return (train_by_label, val_by_label) for single-label PA/AP images, split by PatientID."""
    import gzip

    gz_candidates = sorted(data_dir.glob("*.csv.gz"))
    csv_candidates = sorted(data_dir.glob("*.csv"))
    if not gz_candidates and not csv_candidates:
        raise FileNotFoundError(f"No CSV or CSV.gz found in {data_dir}")

    disk_index = {
        p.name: p for p in data_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg")
    }
    print(f"  Images on disk: {len(disk_index)}")

    def open_csv():
        if gz_candidates:
            return gzip.open(gz_candidates[0], "rt")
        return open(csv_candidates[0])

    by_patient: dict[str, list[tuple[Path, str]]] = {}
    with open_csv() as f:
        for row in csv.DictReader(f):
            image_id = row.get("ImageID", "").strip()
            if image_id not in disk_index:
                continue
            if row.get("Projection", "").strip() not in ("PA", "AP"):
                continue
            try:
                labels = _parse_labels(row["Labels"])
            except (ValueError, KeyError):
                continue
            if len(labels) != 1:
                continue
            label = str(labels[0]).strip().lower()
            if not label or label in EXCLUDE_LABELS:
                continue
            patient_id = row.get("PatientID", "unknown").strip()
            by_patient.setdefault(patient_id, []).append((disk_index[image_id], label))

    # Hash-stable split — reproducible regardless of zip ordering.
    val_patients = {
        pid
        for pid in by_patient
        if int(hashlib.md5(pid.encode()).hexdigest(), 16) % round(1 / val_frac) == 0
    }

    train_by_label: dict[str, list[Path]] = {}
    val_by_label:   dict[str, list[Path]] = {}
    for patient_id, samples in by_patient.items():
        target = val_by_label if patient_id in val_patients else train_by_label
        for path, label in samples:
            target.setdefault(label, []).append(path)

    all_labels = sorted(set(train_by_label) | set(val_by_label))
    print(f"  Found {len(all_labels)} unique labels across {len(by_patient)} patients")
    print("  Label counts (train / val):")
    for lbl in all_labels:
        tr = len(train_by_label.get(lbl, []))
        vl = len(val_by_label.get(lbl, []))
        print(f"    {lbl:<40s}  train={tr:>4d}  val={vl:>4d}")

    classes = sorted(
        c for c in train_by_label
        if c in val_by_label
        and len(train_by_label[c]) >= MIN_TRAIN_PER_CLASS
        and (len(val_by_label[c]) >= MIN_TEST_PER_CLASS or c in FORCE_INCLUDE)
    )
    return {c: train_by_label[c] for c in classes}, {c: val_by_label[c] for c in classes}


def load_negative_pool(
    data_dir: Path,
    rare_classes: set[str],
    val_frac: float = 0.2,
    max_train: int = 5000,
    max_val: int = 2000,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Return (train_paths, val_paths) of single-label PA/AP images not in rare_classes."""
    import gzip

    gz_candidates = sorted(data_dir.glob("*.csv.gz"))
    disk_index = {
        p.name: p for p in data_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg")
    }

    def open_csv():
        if gz_candidates:
            return gzip.open(gz_candidates[0], "rt")
        return open(sorted(data_dir.glob("*.csv"))[0])

    by_patient: dict[str, list[Path]] = {}
    with open_csv() as f:
        for row in csv.DictReader(f):
            image_id = row.get("ImageID", "").strip()
            if image_id not in disk_index:
                continue
            if row.get("Projection", "").strip() not in ("PA", "AP"):
                continue
            try:
                labels = _parse_labels(row["Labels"])
            except (ValueError, KeyError):
                continue
            if len(labels) != 1:
                continue
            label = str(labels[0]).strip().lower()
            if not label or label in rare_classes:
                continue
            patient_id = row.get("PatientID", "unknown").strip()
            by_patient.setdefault(patient_id, []).append(disk_index[image_id])

    val_patients = {
        pid
        for pid in by_patient
        if int(hashlib.md5(pid.encode()).hexdigest(), 16) % round(1 / val_frac) == 0
    }

    train_paths, val_paths = [], []
    for patient_id, paths in by_patient.items():
        if patient_id in val_patients:
            val_paths.extend(paths)
        else:
            train_paths.extend(paths)

    rng = random.Random(seed)
    if len(train_paths) > max_train:
        train_paths = rng.sample(train_paths, max_train)
    if len(val_paths) > max_val:
        val_paths = rng.sample(val_paths, max_val)

    print(f"  Negative pool: {len(train_paths)} train  {len(val_paths)} val")
    return train_paths, val_paths


@torch.no_grad()
def extract_features(
    model: nn.Module, paths: list[Path], device: torch.device
) -> np.ndarray:
    """Returns L2-normalised (N, D) feature array."""
    dataset = ImageDataset(paths)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=False)
    feats = []
    for imgs in loader:
        feats.append(model(imgs.to(device)).cpu().numpy())
    return normalize(np.concatenate(feats))
