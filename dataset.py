# dataset.py
import os
import sys
_IS_WIN = sys.platform.startswith("win")
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD = [0.2241, 0.2215, 0.2239]


def _default_num_workers() -> int:
    if _IS_WIN:
        return 0
    cpu = os.cpu_count() or 4
    return min(8, max(2, cpu // 2))


def _resolve_stl10_root(data_root: str) -> str:
    p = Path(data_root)


    if p.name.lower() == "stl10_binary" and p.is_dir():
        p = p.parent

    candidates = [
        p,
        p / "STL-10",
        p / "STL10",
        p / "stl10",
        p / "stl-10",
    ]

    for cand in candidates:
        if (cand / "stl10_binary").is_dir():
            return str(cand)

    raise FileNotFoundError(
        "Không tìm thấy thư mục 'stl10_binary'. Hãy đảm bảo tồn tại 1 trong các đường dẫn sau:\n"
        + "\n".join([f"- {str(c/'stl10_binary')}" for c in candidates])
    )


def _loader_kwargs(num_workers: int, use_cuda: bool):
    kw = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "shuffle": True,
    }
    # if num_workers > 0:
    #     kw.update({
    #         "persistent_workers": True,
    #         "prefetch_factor": 4,
    #     })
    if num_workers > 0 and (not _IS_WIN):
        kw.update({
            "persistent_workers": True,
            "prefetch_factor": 1,
            "multiprocessing_context": "spawn",
        })
    return kw


def get_dataloaders(
    batch_size: int = 128,
    data_root: str = "./data",
    num_workers: Optional[int] = None,
    download: bool = True,
    use_cuda: Optional[bool] = None,
):
    """
    Supervised dataloaders (train/test) — giữ lại để khi bạn muốn train có nhãn.
    IMPORTANT: tự resolve root để đọc được ./data/STL-10/stl10_binary/...
    """
    if num_workers is None:
        num_workers = _default_num_workers()

    if use_cuda is None:
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False

    stl_root = _resolve_stl10_root(data_root)

    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
    ])

    train_set = datasets.STL10(
        root=stl_root,
        split="train",
        download=download,
        transform=transform_train,
    )

    test_set = datasets.STL10(
        root=stl_root,
        split="test",
        download=download,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        drop_last=False,
        **_loader_kwargs(num_workers=num_workers, use_cuda=use_cuda),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, test_loader



class GaussianBlur(object):
    """Gaussian blur augmentation (PIL-based) used by SimCLR."""
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image):
        import random
        if random.random() > self.p:
            return img
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius))


class TwoCropsTransform:
    """Create two differently augmented views of the same image."""
    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


class ViewsOnly(Dataset):
    """
    Wrapper: nếu dataset gốc trả (views, label) thì chỉ trả views.
    Dùng cho STL10 để bỏ label khi self-supervised.
    """
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        # STL10: ((q,k), label) -> trả (q,k)
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return item[0]
        return item


class UnlabeledImageFolder(Dataset):
    """Loads images recursively from a directory and returns only images (no labels)."""
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        self.samples = [
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.IMG_EXTS
        ]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.root}. Supported extensions: {sorted(self.IMG_EXTS)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_simclr_transform(image_size: int = 96) -> transforms.Compose:
    """SimCLR-style augmentation pipeline."""
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
    ])


def get_simclr_dataloader(
    batch_size: int = 256,
    data_root: str = "./data",
    dataset: str = "stl10",
    image_size: int = 96,
    num_workers: Optional[int] = None,
    download: bool = False,
    use_cuda: Optional[bool] = None,
) -> DataLoader:
    """
    Unlabeled dataloader cho self-supervised (SimCLR):
    - dataset="stl10": dùng STL10 split='unlabeled' (100k)
    - dataset="folder": đọc ảnh từ folder data_root (không label)
    """
    if num_workers is None:
        num_workers = _default_num_workers()

    if use_cuda is None:
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False

    base_transform = build_simclr_transform(image_size=image_size)
    transform = TwoCropsTransform(base_transform)

    dataset = dataset.lower().strip()
    if dataset == "stl10":
        stl_root = _resolve_stl10_root(data_root)
        base_ds = datasets.STL10(
            root=stl_root,
            split="unlabeled",   # <-- 100k unlabeled_X.bin
            download=download,
            transform=transform,
        )
        ds = ViewsOnly(base_ds)  # <-- bỏ label, chỉ trả (q,k)
    elif dataset == "folder":
        ds = UnlabeledImageFolder(root=data_root, transform=transform)
    else:
        raise ValueError('dataset must be "stl10" or "folder"')

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=True,
        **_loader_kwargs(num_workers=num_workers, use_cuda=use_cuda),
    )
    return loader