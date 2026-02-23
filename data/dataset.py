import ast
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_CACHE_SIZE = 256  # pre-resize to this before caching


def _load_gray256(path: Path) -> np.ndarray:
    """Open an image, convert to grayscale, resize to 256×256, return uint8 array."""
    img = Image.open(path).convert("L")  # L = 8-bit grayscale
    img = img.resize((_CACHE_SIZE, _CACHE_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


class UnlabeledChestXrayDataset(Dataset):
    """Returns two augmented views of each image for contrastive learning."""

    def __init__(
        self,
        image_paths: list[Path],
        transform: transforms.Compose,
        cache_in_ram: bool = False,
    ):
        self.image_paths = image_paths
        self.transform = transform
        # Stored as (N, 256, 256) uint8 tensor in shared memory; workers access without pickling
        self._cache: torch.Tensor | None = None

        if cache_in_ram:
            print(f"  Caching {len(image_paths)} images as 256×256 grayscale arrays...", flush=True)
            arrays = [_load_gray256(p) for p in image_paths]
            stacked = np.stack(arrays, axis=0)  # (N, 256, 256) uint8
            self._cache = torch.from_numpy(stacked).share_memory_()
            mb = stacked.nbytes / 1024**2
            print(f"  Cache ready — {mb:.0f} MB in RAM.", flush=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            # fromarray('L') reconstructs PIL grayscale; convert('RGB') repeats the
            # channel 3× so ImageNet-pretrained backbones see the expected shape.
            img = Image.fromarray(self._cache[idx].numpy(), mode="L").convert("RGB")
        else:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)



def collect_image_paths(
    dataset_name: str, root_dir: str, txt_file: str | None = None
) -> list[Path]:
    """Discover image files for a given dataset.

    If ``txt_file`` is provided it must be a plain-text file with one image
    filename (basename only) per line.  The function builds a lookup index
    from all images under ``root_dir`` and resolves each listed filename to
    its full path, skipping any that are not found on disk.
    """
    root = Path(root_dir)

    if dataset_name == "nih":
        all_paths = sorted(root.glob("images*/**/*.png"))
        if not all_paths:
            all_paths = sorted(root.glob("**/*.png"))

        if txt_file:
            index = {p.name: p for p in all_paths}
            paths = []
            missing = 0
            with open(txt_file) as f:
                for line in f:
                    name = line.strip()
                    if not name:
                        continue
                    if name in index:
                        paths.append(index[name])
                    else:
                        missing += 1
            if missing:
                print(f"  WARNING — {missing} filenames from {txt_file} not found on disk")
        else:
            paths = all_paths
    elif dataset_name == "chexpert":
        csv_path = root / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CheXpert CSV not found: {csv_path}")
        paths = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = root / row["Path"]
                if img_path.exists():
                    paths.append(img_path)
    elif dataset_name == "mimic-cxr":
        paths = sorted(root.glob("files/**/*.jpg"))
    elif dataset_name == "padchest":
        # Find the CSV — name varies between full dataset and Kaggle sample
        csv_candidates = sorted(root.glob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(f"No CSV found in {root}")
        csv_path = csv_candidates[0]
        paths = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                image_id = row.get("ImageID", "").strip()
                if not image_id:
                    continue
                # Full PadChest splits images into numbered subdirs (ImageDir column).
                # Kaggle sample stores them flat under root.
                try:
                    subdir = str(int(float(row["ImageDir"])))
                    p = root / subdir / image_id
                except (KeyError, ValueError):
                    p = root / image_id
                if not p.exists():
                    p = root / image_id  # fall back to flat layout
                if p.exists():
                    paths.append(p)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'nih', 'chexpert', 'mimic-cxr', or 'padchest'.")

    return paths


