"""
ELEC 475: Lab 2 - Unified Dataset & DataLoader Factory
Supports CIFAR-10 (primary) and Tiny ImageNet-200 (secondary benchmark).
"""
import os
import urllib.request
import zipfile
import torch
# CHANGED: Dataset supports the custom Tiny ImageNet validation loader.
from torch.utils.data import DataLoader, Dataset
# ADDED: Loads validation images as RGB, matching ImageFolder's default loader.
from torchvision.datasets.folder import default_loader
from torchvision.datasets import CIFAR10, ImageFolder
from .transforms import get_train_transforms, get_test_transforms

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def download_and_extract_tiny_imagenet(data_dir: str = "./data"):
    target_dir = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.exists(target_dir):
        return target_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet-200 from {TINY_IMAGENET_URL}...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("Download complete.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")
    return target_dir


# ADDED: The standard Tiny ImageNet validation split stores images together
# in val/images and labels in val/val_annotations.txt, rather than in
# class-specific folders. This loader reads that separate validation split.
class TinyImageNetValidation(Dataset):
    def __init__(self, tiny_dir, class_to_idx, transform=None):
        self.transform = transform

        # Reuse training class indices so labels have identical meanings
        # in training and validation.
        self.class_to_idx = dict(class_to_idx)
        self.classes = [
            name for name, index in sorted(
                self.class_to_idx.items(),
                key=lambda item: item[1]
            )
        ]

        val_dir = os.path.join(tiny_dir, "val")
        annotation_path = os.path.join(val_dir, "val_annotations.txt")
        image_dir = os.path.join(val_dir, "images")

        self.samples = []

        with open(annotation_path, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                filename, class_name, *_ = line.split()
                image_path = os.path.join(image_dir, filename)

                if not os.path.isfile(image_path):
                    raise FileNotFoundError(image_path)

                target = self.class_to_idx[class_name]
                self.samples.append((image_path, target))

        self.targets = [target for _, target in self.samples]

        # Catch incomplete or unexpected validation data before training.
        if len(self.samples) != 10000:
            raise ValueError(
                f"Expected 10,000 validation images, "
                f"found {len(self.samples)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = default_loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def get_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_augmentation: bool = True,
    target_size: int = 64
):
    dataset_name = dataset.lower()

    if dataset_name == "cifar10":
        train_tf = get_train_transforms("cifar10", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("cifar10", target_size=target_size)

        train_set = CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        val_set   = CIFAR10(root=data_dir, train=False, download=True, transform=val_tf)
        num_classes = 10
        classes = CIFAR10_CLASSES

    elif dataset_name in ["tiny_imagenet", "tinyimagenet"]:
        tiny_dir = download_and_extract_tiny_imagenet(data_dir)
        train_dir = os.path.join(tiny_dir, "train")

        train_tf = get_train_transforms("tiny_imagenet", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("tiny_imagenet", target_size=target_size)

        train_set = ImageFolder(root=train_dir, transform=train_tf)

        # CHANGED: Previously loaded train_dir again, evaluating on training
        # images. Now load the separate 10,000-image validation split,
        # using the same class indices and deterministic validation transforms.
        val_set = TinyImageNetValidation(
            tiny_dir=tiny_dir,
            class_to_idx=train_set.class_to_idx,
            transform=val_tf
        )

        num_classes = 200
        classes = train_set.classes

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from 'cifar10', 'tiny_imagenet'.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader, num_classes, classes