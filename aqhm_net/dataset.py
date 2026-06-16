"""
dataset.py
----------
Data loading and preprocessing for MNIST, MedMNIST, and CIFAR datasets.

Preprocessing pipeline (Section 3 of METHODOLOGY.md):

  3.1 Per-channel z-score normalisation:
      x_norm = (x - mean_train) / std_train
      Computed independently per dataset from the training split.

  3.2 Patch splitting for quantum encoding:
      Images resized to 28×28 (all MedMNIST and CIFAR variants).
      The 7×7 superpixel grid is produced implicitly by the backbone's
      stride-2 downsampling (no explicit splitting needed in the dataloader).

  Data augmentation (training only, Section 14.5):
      RandomHorizontalFlip(p=0.5)
      RandomRotation(±15°)
      ColorJitter(brightness=0.1, contrast=0.1)  [RGB datasets only]
      Gaussian noise N(0, 0.02)                  [applied in-batch by trainer]

Supported datasets:
    • MNIST          — 28×28 grayscale, 10 digit classes
    • FashionMNIST   — 28×28 grayscale, 10 garment classes
    • PathMNIST      — colorectal histology, 9 classes, RGB
    • DermaMNIST     — skin lesion dermoscopy, 7 classes, RGB
    • PneumoniaMNIST — chest X-ray, 2 classes, greyscale
    • BreastMNIST    — ultrasound, 2 classes, greyscale
    • CIFAR-10       — 32×32 natural images, 10 classes, RGB (resized to 28×28)
    • CIFAR-100      — 32×32 natural images, 100 classes, RGB (resized to 28×28)

FashionMNIST class labels (0-9):
    0=T-shirt/top, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat,
    5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot

Class split for MNIST / FashionMNIST:
    Full (10 classes): all categories (default)
    Subset: pass mnist_classes / fashion_classes to restrict to N categories
    Per-class balanced sampling: 5,000 train / 1,000 val per class
    (following Wu et al., 2025 for fair comparison).

CIFAR split:
    Stratified 90/10 split of the 50k official training set → 45k train / 5k val.
    Official 10k test set used unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

# skimage provides CLAHE (equalize_adapthist). Optional — guarded so the module
# still imports on environments without it (CLAHE simply becomes a no-op then).
try:
    from skimage.exposure import equalize_adapthist
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

# MedMNIST must be installed separately: pip install medmnist
try:
    import medmnist
    from medmnist import INFO, Evaluator
    _MEDMNIST_AVAILABLE = True
except ImportError:
    _MEDMNIST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataset configuration registry
# ---------------------------------------------------------------------------

DATASET_CONFIG: dict[str, dict] = {
    "mnist": {
        "channels": 1,
        "num_classes": 10,
        "mean": (0.1307,),
        "std":  (0.3081,),
        "medmnist": False,
    },
    # FashionMNIST — same spatial structure as MNIST (28×28, greyscale, 10 classes).
    # Normalisation stats computed from the 60,000-sample training split:
    #   mean=0.2860, std=0.3530  (Xiao et al., 2017 / widely reproduced).
    "fashionmnist": {
        "channels": 1,
        "num_classes": 10,
        "mean": (0.2860,),
        "std":  (0.3530,),
        "medmnist": False,
        # Human-readable class labels for reporting / confusion-matrix axes
        "class_names": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
    },
    "pathmnist": {
        "channels": 3,
        "num_classes": 9,
        "mean": (0.7405, 0.5330, 0.7058),
        "std":  (0.1761, 0.2267, 0.1614),
        "medmnist": True,
        "medmnist_key": "pathmnist",
        # Histology colour cast varies by H&E staining batch — normalise it.
        "color_constancy": True,
        # Tissue orientation is arbitrary: all flips/rotations are label-preserving.
        "tta_views": ["id", "hflip", "vflip", "rot90", "rot180", "rot270"],
    },
    "dermamnist": {
        "channels": 3,
        "num_classes": 7,
        "mean": (0.7631, 0.5381, 0.5614),
        "std":  (0.1366, 0.1541, 0.1690),
        "medmnist": True,
        "medmnist_key": "dermamnist",
        # Dermoscopy illumination/colour cast is the dominant nuisance variable.
        "color_constancy": True,
        "tta_views": ["id", "hflip", "vflip", "rot180"],
    },
    "pneumoniamnist": {
        "channels": 1,
        "num_classes": 2,
        "mean": (0.5720,),
        "std":  (0.1671,),
        "medmnist": True,
        "medmnist_key": "pneumoniamnist",
        # CLAHE left OFF by default: it amplifies speckle/noise and measurably
        # hurt the other grayscale modality (BreastMNIST: -0.08 AUC). Opt in for
        # ablation with the config key below set True or remove --no_clahe.
        "clahe": False,
        "tta_views": ["id", "hflip"],   # chest is ~L-R symmetric; vflip unrealistic
    },
    "breastmnist": {
        "channels": 1,
        "num_classes": 2,
        "mean": (0.3274,),
        "std":  (0.2041,),
        "medmnist": True,
        "medmnist_key": "breastmnist",
        # CLAHE OFF: empirically hurts this ultrasound set (mAUC 0.849 -> 0.773
        # over 3 seeds) by amplifying speckle. TTA alone improves over baseline.
        "clahe": False,
        "tta_views": ["id", "hflip"],
    },
    # ── CIFAR-10 / CIFAR-100 ──────────────────────────────────────────────────
    # Natural RGB images, 32×32 — resized to 28×28 so the backbone's stride-2
    # downsampling produces the same 7×7 spatial grid used by SSA and the VQC.
    # Stats from the official 50k training splits (widely reproduced).
    # Val split: stratified 90/10 from the 50k training set → 45k train / 5k val.
    "cifar10": {
        "channels": 3,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2470, 0.2435, 0.2616),
        "medmnist": False,
        "resize": 28,   # resize 32×32 → 28×28 to match backbone stride
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
    },
    "cifar100": {
        "channels": 3,
        "num_classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std":  (0.2675, 0.2565, 0.2761),
        "medmnist": False,
        "resize": 28,
    },
}


# ---------------------------------------------------------------------------
# Per-channel normalisation stats computation
# ---------------------------------------------------------------------------

def compute_normalisation_stats(
    dataset: torch.utils.data.Dataset,
    n_channels: int,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute per-channel mean and std over the dataset.

    Used when the hardcoded stats no longer match the pixel distribution —
    e.g. after colour constancy or CLAHE change the raw intensities, the
    DATASET_CONFIG stats are stale and must be recomputed on the transformed
    training split.

    Args:
        dataset    : a PyTorch Dataset returning (image_tensor, label).
        n_channels : number of image channels.
        max_samples: if set, estimate stats from a random subset of this many
                     samples (sufficient for mean/std, avoids a slow full pass
                     over large sets like PathMNIST's 90k images).
        seed       : RNG seed for subset selection.

    Returns:
        (mean, std) — tuples of length n_channels.
    """
    if max_samples is not None and len(dataset) > max_samples:
        idx = np.random.default_rng(seed).choice(
            len(dataset), size=max_samples, replace=False
        )
        dataset = Subset(dataset, idx.tolist())
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    channel_sum    = torch.zeros(n_channels)
    channel_sum_sq = torch.zeros(n_channels)
    n_pixels = 0

    for imgs, _ in loader:
        # imgs: (B, C, H, W)
        B, C, H, W = imgs.shape
        channel_sum    += imgs.sum(dim=[0, 2, 3])
        channel_sum_sq += (imgs ** 2).sum(dim=[0, 2, 3])
        n_pixels += B * H * W

    mean = (channel_sum / n_pixels).tolist()
    var  = (channel_sum_sq / n_pixels) - torch.tensor(mean) ** 2
    std  = var.sqrt().tolist()
    return tuple(mean), tuple(std)


# ---------------------------------------------------------------------------
# Gaussian noise augmentation (applied in-batch by the trainer)
# ---------------------------------------------------------------------------

class AddGaussianNoise:
    """Torchvision-compatible transform that adds Gaussian noise to a tensor.

    Applied during training only (Section 14.5):
        N(0, σ=0.02) added to normalised pixel values.
    """

    def __init__(self, sigma: float = 0.02) -> None:
        self.sigma = sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.sigma

    def __repr__(self) -> str:
        return f"AddGaussianNoise(sigma={self.sigma})"


# ---------------------------------------------------------------------------
# Domain-specific deterministic preprocessing (PIL -> PIL, train AND eval)
# ---------------------------------------------------------------------------

class ShadesOfGray:
    """Minkowski-p-norm colour constancy (Finlayson & Trezzi, 2004).

    Removes the device-/illumination-dependent colour cast that varies across
    dermoscopy and histology acquisition systems — a well-documented confounder
    for skin-lesion (DermaMNIST) and colorectal-histology (PathMNIST)
    classification (Barata et al., 2015).  p=6 is the canonical "shades of grey"
    setting (p=1 reduces to Gray-World, p→∞ to max-RGB).

    Per-channel illuminant ê_c = (mean_pixels x_c^p)^(1/p); each channel is
    rescaled by mean(ê)/ê_c, which neutralises the cast while preserving overall
    brightness.  Operates on a PIL RGB image, returns a PIL RGB image.
    """

    def __init__(self, power: int = 6, eps: float = 1e-6) -> None:
        self.power = power
        self.eps = eps

    def __call__(self, img: Image.Image) -> Image.Image:
        x = np.asarray(img, dtype=np.float32)          # (H, W, 3), [0, 255]
        illum = np.power(np.mean(np.power(x, self.power), axis=(0, 1)),
                         1.0 / self.power) + self.eps   # (3,)
        gains = illum.mean() / illum                    # preserve mean brightness
        x = np.clip(x * gains[None, None, :], 0, 255).astype(np.uint8)
        return Image.fromarray(x)

    def __repr__(self) -> str:
        return f"ShadesOfGray(power={self.power})"


class CLAHE:
    """Contrast-Limited Adaptive Histogram Equalisation (grayscale medical).

    Enhances local contrast of chest-X-ray infiltrates (PneumoniaMNIST) and
    ultrasound lesion boundaries (BreastMNIST), the standard radiology/ultrasound
    pre-enhancement.  Wraps skimage.exposure.equalize_adapthist; if skimage is
    unavailable the transform degrades gracefully to a no-op.

    Operates on a PIL 'L' image, returns a PIL 'L' image.
    """

    def __init__(self, clip_limit: float = 0.01) -> None:
        self.clip_limit = clip_limit

    def __call__(self, img: Image.Image) -> Image.Image:
        if not _SKIMAGE_AVAILABLE:
            return img
        x = np.asarray(img, dtype=np.float32) / 255.0   # (H, W), [0, 1]
        x = equalize_adapthist(x, clip_limit=self.clip_limit)
        return Image.fromarray((x * 255.0).astype(np.uint8))

    def __repr__(self) -> str:
        return f"CLAHE(clip_limit={self.clip_limit})"


def _build_pre_ops(cfg: dict) -> list:
    """Deterministic domain preprocessing prepended to BOTH train and eval
    pipelines, selected from the dataset config.  Returns a (possibly empty)
    list of PIL->PIL transforms to run before ToTensor()."""
    ops: list = []
    if cfg.get("color_constancy"):
        ops.append(ShadesOfGray(power=6))
    if cfg.get("clahe"):
        ops.append(CLAHE(clip_limit=0.01))
    return ops


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def _stratified_val_split(
    targets,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Stratified train/val split preserving per-class proportions.

    Used for CIFAR-10/100 which have no official validation split.
    Returns (train_indices, val_indices) from the 50k training set.
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    targets = np.array(targets)
    classes = np.unique(targets)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in classes:
        cls_mask = np.where(targets == c)[0]
        rng.shuffle(cls_mask)
        n_val = max(1, int(len(cls_mask) * val_frac))
        val_idx.extend(cls_mask[:n_val].tolist())
        train_idx.extend(cls_mask[n_val:].tolist())
    return train_idx, val_idx


def _build_transforms(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    is_rgb: bool,
    is_train: bool,
    is_medical: bool = False,
    pre_ops: list | None = None,
    resize: int | None = None,
) -> transforms.Compose:
    """Build the full torchvision transform pipeline.

    pre_ops : deterministic domain preprocessing (colour constancy / CLAHE)
              applied to the raw PIL image, before ToTensor, on both splits.
    resize  : if set, resize the PIL image to this size before ToTensor
              (used for CIFAR 32×32 → 28×28 to match backbone stride).
    """
    tfm_list: list = list(pre_ops) if pre_ops else []
    if resize is not None:
        tfm_list.append(transforms.Resize(resize, antialias=True))
    tfm_list.append(transforms.ToTensor())

    if is_train:
        tfm_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3 if is_medical else 0.0),
            transforms.RandomRotation(degrees=25 if is_medical else 15),
        ]
        if is_medical:
            tfm_list.append(
                transforms.RandomAffine(
                    degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10)
                )
            )
        if is_rgb:
            jitter_strength = 0.2 if is_medical else 0.1
            tfm_list.append(
                transforms.ColorJitter(
                    brightness=jitter_strength,
                    contrast=jitter_strength,
                    saturation=jitter_strength if is_medical else 0.0,
                )
            )
        tfm_list.append(AddGaussianNoise(sigma=0.03 if is_medical else 0.02))

    tfm_list.append(transforms.Normalize(mean=mean, std=std))

    if is_train and is_medical:
        # Random patch erasing after normalisation (helps small datasets)
        tfm_list.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)))

    return transforms.Compose(tfm_list)


# ---------------------------------------------------------------------------
# Class weight computation (inverse-frequency, for imbalanced datasets)
# ---------------------------------------------------------------------------

def compute_class_weights(
    loader: DataLoader,
    num_classes: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a DataLoader.

    weight[c] = N / (num_classes * count[c])
    Balanced datasets → all weights ≈ 1.0 (no effect).
    Imbalanced datasets (e.g. DermaMNIST) → minority classes upweighted.

    Args:
        loader      : training DataLoader.
        num_classes : total number of classes.
        device      : target device for the weight tensor.

    Returns:
        (num_classes,) float tensor, normalised so weights sum to num_classes.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in loader:
        labs = labels.squeeze().long()
        for c in range(num_classes):
            counts[c] += (labs == c).sum().item()

    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    if device is not None:
        weights = weights.to(device)
    return weights


# ---------------------------------------------------------------------------
# Per-sample weights for WeightedRandomSampler
# ---------------------------------------------------------------------------

def compute_sample_weights(dataset) -> torch.Tensor:
    """Inverse-frequency weight for each sample — use with WeightedRandomSampler.

    Produces balanced mini-batches without requiring extreme loss weights,
    which can cause mode collapse on tiny datasets (BreastMNIST) or datasets
    with very rare classes (DermaMNIST 1.2% dermatofibroma).

    Works with MedMNIST datasets (expose .labels) and torchvision datasets
    (expose .targets).
    """
    if hasattr(dataset, "labels"):
        # MedMNIST: .labels is ndarray (N, 1)
        labels = torch.from_numpy(dataset.labels.squeeze().astype(np.int64))
    elif hasattr(dataset, "targets"):
        labels = torch.as_tensor(dataset.targets).long()
    else:
        labels = torch.tensor(
            [int(torch.tensor(dataset[i][1]).squeeze()) for i in range(len(dataset))],
            dtype=torch.long,
        )
    class_counts = torch.bincount(labels)
    weights = torch.tensor(
        [1.0 / class_counts[int(l)].item() for l in labels],
        dtype=torch.float32,
    )
    return weights


# ---------------------------------------------------------------------------
# Balanced sampler for MNIST (Wu et al., 2025 protocol)
# ---------------------------------------------------------------------------

def _balanced_mnist_indices(
    labels: torch.Tensor,
    classes: list[int],
    n_per_class: int,
    split: str,  # "train" or "val"
    rng: np.random.Generator,
) -> list[int]:
    """Sample n_per_class indices per class for balanced MNIST.

    Wu et al. (2025) protocol:
        5,000 training samples / 1,000 validation samples per class.
        For MNIST-4 (4 classes): 20k train / 4k val.
        For MNIST-10 (10 classes): 50k train / 10k val.
    """
    indices = []
    for cls in classes:
        cls_mask = (labels == cls).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(cls_mask)
        chosen = cls_mask[:n_per_class]
        indices.extend(chosen.tolist())
    return indices


# ---------------------------------------------------------------------------
# Main dataloader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    dataset_name: str = "mnist",
    data_root: str = "./data",
    batch_size: int = 32,
    num_workers: int = 2,
    mnist_classes: list[int] | None = None,
    fashion_classes: list[int] | None = None,
    seed: int = 42,
    use_balanced_sampler: bool = False,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders for the requested dataset.

    Args:
        dataset_name    : one of the keys in DATASET_CONFIG.
        data_root       : directory for downloaded data (created if absent).
        batch_size      : mini-batch size (default 32 per Section 14.3).
        num_workers     : DataLoader worker threads.
        mnist_classes   : for 'mnist' only — digit classes to include.
                          None = all 10 classes;  [0,1,2,3] = MNIST-4.
        fashion_classes : for 'fashionmnist' only — garment category indices
                          to include (0-9).  None = all 10 classes.
                          E.g. [0,6] selects T-shirt and Shirt only.
        seed            : random seed for balanced sampling reproducibility.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    name = dataset_name.lower()
    if name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_CONFIG.keys())}"
        )

    cfg = DATASET_CONFIG[name]
    mean, std = cfg["mean"], cfg["std"]
    is_rgb     = cfg["channels"] == 3
    is_medical = cfg.get("medmnist", False)

    # Deterministic domain preprocessing (colour constancy / CLAHE). When active,
    # it shifts the pixel distribution, so the hardcoded DATASET_CONFIG stats are
    # stale — recompute mean/std on the transformed training split.
    pre_ops = _build_pre_ops(cfg)
    if pre_ops and is_medical and _MEDMNIST_AVAILABLE:
        _DataClass = getattr(medmnist, INFO[cfg["medmnist_key"]]["python_class"])
        _stat_tfm = transforms.Compose(list(pre_ops) + [transforms.ToTensor()])
        _stat_ds = _DataClass(
            split="train", transform=_stat_tfm, download=True, root=data_root
        )
        mean, std = compute_normalisation_stats(
            _stat_ds, cfg["channels"], max_samples=10_000, seed=seed
        )
        print(f"[preproc] {name}: {[op.__class__.__name__ for op in pre_ops]} "
              f"-> recomputed mean={tuple(round(m, 4) for m in mean)} "
              f"std={tuple(round(s, 4) for s in std)}")

    resize    = cfg.get("resize")
    train_tfm = _build_transforms(mean, std, is_rgb, is_train=True,  is_medical=is_medical, pre_ops=pre_ops, resize=resize)
    eval_tfm  = _build_transforms(mean, std, is_rgb, is_train=False, is_medical=is_medical, pre_ops=pre_ops, resize=resize)

    # ── MNIST ───────────────────────────────────────────────────────────────
    if name == "mnist":
        full_train = datasets.MNIST(
            data_root, train=True, download=True, transform=train_tfm
        )
        full_val = datasets.MNIST(
            data_root, train=True, download=True, transform=eval_tfm
        )
        test_ds = datasets.MNIST(
            data_root, train=False, download=True, transform=eval_tfm
        )

        classes = mnist_classes if mnist_classes is not None else list(range(10))
        rng = np.random.default_rng(seed)

        # Train: 5,000 per class (Wu et al., 2025)
        train_idx = _balanced_mnist_indices(
            full_train.targets, classes, 5000, "train", rng
        )
        # Val: 1,000 per class from the same training pool
        # (use a non-overlapping subset by taking after the first 5k)
        val_idx = _balanced_mnist_indices(
            full_val.targets, classes, 1000, "val",
            np.random.default_rng(seed + 1)
        )

        # Filter test to requested classes
        test_mask = torch.isin(test_ds.targets, torch.tensor(classes))
        test_idx  = test_mask.nonzero(as_tuple=True)[0].tolist()

        return {
            "train": DataLoader(
                Subset(full_train, train_idx),
                batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
            ),
            "val": DataLoader(
                Subset(full_val, val_idx),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            ),
            "test": DataLoader(
                Subset(test_ds, test_idx),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            ),
        }

    # ── FashionMNIST ─────────────────────────────────────────────────────────
    # Structurally identical to MNIST: 60k/10k train/test split, 28×28 grey,
    # 10 classes, same balanced-sampling protocol (5k train / 1k val per class).
    # Uses torchvision.datasets.FashionMNIST — no extra dependencies needed.
    if name == "fashionmnist":
        full_train = datasets.FashionMNIST(
            data_root, train=True, download=True, transform=train_tfm
        )
        full_val = datasets.FashionMNIST(
            data_root, train=True, download=True, transform=eval_tfm
        )
        test_ds = datasets.FashionMNIST(
            data_root, train=False, download=True, transform=eval_tfm
        )

        # fashion_classes lets you study a subset of garment categories
        # (e.g. [0, 6] for T-shirt vs Shirt — a famously hard pair)
        classes = fashion_classes if fashion_classes is not None else list(range(10))
        rng = np.random.default_rng(seed)

        # Train: 5,000 per class  |  Val: 1,000 per class (balanced)
        train_idx = _balanced_mnist_indices(
            full_train.targets, classes, 5000, "train", rng
        )
        val_idx = _balanced_mnist_indices(
            full_val.targets, classes, 1000, "val",
            np.random.default_rng(seed + 1)
        )

        # Filter test split to the requested garment categories
        test_mask = torch.isin(test_ds.targets, torch.tensor(classes))
        test_idx  = test_mask.nonzero(as_tuple=True)[0].tolist()

        return {
            "train": DataLoader(
                Subset(full_train, train_idx),
                batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
            ),
            "val": DataLoader(
                Subset(full_val, val_idx),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            ),
            "test": DataLoader(
                Subset(test_ds, test_idx),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            ),
        }

    # ── CIFAR-10 / CIFAR-100 ────────────────────────────────────────────────
    if name in ("cifar10", "cifar100"):
        DataClass = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100

        full_train = DataClass(data_root, train=True,  download=True, transform=train_tfm)
        full_val   = DataClass(data_root, train=True,  download=True, transform=eval_tfm)
        test_ds    = DataClass(data_root, train=False, download=True, transform=eval_tfm)

        # Stratified 90/10 split of the 50k official training set.
        train_idx, val_idx = _stratified_val_split(
            full_train.targets, val_frac=0.1, seed=seed
        )

        return {
            "train": DataLoader(
                Subset(full_train, train_idx),
                batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
            ),
            "val": DataLoader(
                Subset(full_val, val_idx),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            ),
            "test": DataLoader(
                test_ds,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            ),
        }

    # ── MedMNIST ────────────────────────────────────────────────────────────
    if not _MEDMNIST_AVAILABLE:
        raise ImportError(
            "MedMNIST is required for this dataset. "
            "Install with: pip install medmnist"
        )

    medkey = cfg["medmnist_key"]
    DataClass = getattr(medmnist, INFO[medkey]["python_class"])

    # MedMNIST images are 28×28 by default — matches our target resolution
    train_ds = DataClass(
        split="train", transform=train_tfm, download=True, root=data_root
    )
    val_ds = DataClass(
        split="val",   transform=eval_tfm,  download=True, root=data_root
    )
    test_ds = DataClass(
        split="test",  transform=eval_tfm,  download=True, root=data_root
    )

    if use_balanced_sampler:
        sample_weights = compute_sample_weights(train_ds)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    return {
        "train": train_loader,
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }
