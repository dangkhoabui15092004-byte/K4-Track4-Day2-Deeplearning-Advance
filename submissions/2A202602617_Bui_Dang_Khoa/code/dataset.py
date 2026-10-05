from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee Apple","Lantana","Parkinsonia","Parthenium","Prickly Acacia",
               "Rubber Vine","Siam Weed","Snake Weed","Negatives"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir, fold=0):
    d = Path(labels_dir)
    return tuple(pd.read_csv(d / f"{s}_subset{fold}.csv") for s in ("train", "val", "test"))


def check_split(train_df, val_df, test_df, images_dir):
    sets = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: len(v) for k, v in sets.items()}
    total = sum(n.values())
    per_class = {k: v.Label.value_counts().sort_index().to_dict() for k, v in sets.items()}
    S = {k: set(v.Filename) for k, v in sets.items()}
    overlap = {"train&val": len(S["train"] & S["val"]),
               "train&test": len(S["train"] & S["test"]),
               "val&test": len(S["val"] & S["test"])}
    union = len(S["train"] | S["val"] | S["test"])
    missing = sum(not (Path(images_dir) / f).exists()
                  for f in pd.concat(sets.values()).Filename)
    ratio = {k: round(v / total, 4) for k, v in n.items()}
    print("n:", n, "| tổng:", total, "| tỉ lệ:", ratio)
    for k, v in per_class.items(): print(k, v)
    print("giao:", overlap, "| hợp:", union, "| thiếu file:", missing)
    assert all(v == 0 for v in overlap.values()), "Giao khác rỗng!"
    assert union == 17509 and total == 17509, "Hợp != 17509"
    assert missing == 0, "Có file không tồn tại"
    assert all(abs(ratio[k] - r) < 0.01 for k, r in zip(("train","val","test"), (.6,.2,.2))), "Lệch 60/20/20"
    return {"n": n, "ratio": ratio, "per_class": per_class, "overlap": overlap, "union": union}


def build_transforms(train, img_size=224, aug="basic", mean=IMAGENET_MEAN, std=IMAGENET_STD):
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        # ảnh gốc 256: <256 -> CenterCrop, ==256 giữ nguyên, >256 -> Resize
        if img_size < 256: pre = [T.CenterCrop(img_size)]
        elif img_size == 256: pre = []
        else: pre = [T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC)]
        return T.Compose(pre + norm)
    base = [T.RandomResizedCrop(img_size, scale=(0.35, 1.0)), T.RandomHorizontalFlip()]
    if aug == "basic": extra = []
    elif aug == "color": extra = [T.ColorJitter(0.3, 0.3, 0.3, 0.05)]
    elif aug == "trivial": extra = [T.TrivialAugmentWide()]
    elif aug == "randaug": extra = [T.RandAugment(num_ops=2, magnitude=9)]
    else: raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose(base + extra + norm)


class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None, cache=False):
        self.df = df.reset_index(drop=True)
        self.dir = Path(images_dir)
        self.transform = transform
        self.files = self.df.Filename.tolist()
        self.labels = self.df.Label.astype(int).tolist()
        self.cache = None
        if cache:  # nạp trước uint8 vào RAM (~65GB/1000 ảnh? không: ~196KB/ảnh)
            self.cache = [np.asarray(Image.open(self.dir / f).convert("RGB")) for f in self.files]

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        if self.cache is not None: img = Image.fromarray(self.cache[i])
        else: img = Image.open(self.dir / self.files[i]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[i], self.files[i]


def _worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); import random; random.seed(s)


def make_loader(df, images_dir, transform, batch_size, train, sampler=None,
                num_workers=2, seed=0, cache=False):
    ds = DeepWeedsDataset(df, images_dir, transform, cache=cache)
    g = torch.Generator(); g.manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
              worker_init_fn=_worker_init, generator=g,
              persistent_workers=num_workers > 0)
    if train:
        if sampler == "balanced":
            cnt = np.bincount(ds.labels, minlength=NUM_CLASSES)
            w = torch.tensor([1.0 / cnt[l] for l in ds.labels], dtype=torch.double)
            smp = WeightedRandomSampler(w, num_samples=len(ds), replacement=True, generator=g)
            return DataLoader(ds, sampler=smp, drop_last=True, **kw)
        return DataLoader(ds, shuffle=True, drop_last=True, **kw)
    return DataLoader(ds, shuffle=False, drop_last=False, **kw)
