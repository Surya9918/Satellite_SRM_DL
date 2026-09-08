"""Multi-spectral data augmentation preserving spectral band relationships."""
import numpy as np
from typing import Tuple

class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            lr, hr = t(lr, hr)
        return lr, hr

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            lr = np.flip(lr, axis=-1).copy()
            hr = np.flip(hr, axis=-1).copy()
        return lr, hr

class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            lr = np.flip(lr, axis=-2).copy()
            hr = np.flip(hr, axis=-2).copy()
        return lr, hr

class RandomRot90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            k = np.random.choice([1, 2, 3])
            lr = np.rot90(lr, k, axes=(-2, -1)).copy()
            hr = np.rot90(hr, k, axes=(-2, -1)).copy()
        return lr, hr

def get_transforms(is_train: bool = True) -> ComposeTransforms:
    if is_train:
        return ComposeTransforms([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRot90(p=0.5)
        ])
    return ComposeTransforms([])
