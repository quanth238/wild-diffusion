import numpy as np

from training.dataset import ImageFolderDataset


class ToyImageFolderSubsetDataset(ImageFolderDataset):
    """ImageFolderDataset with an explicit raw-index subset.

    This keeps the main EDM dataloader stack intact while letting the toy lane
    reuse its existing deterministic train-subset policy.
    """

    def __init__(self, *args, subset_raw_idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        if subset_raw_idx is None:
            return

        subset = np.asarray(subset_raw_idx, dtype=np.int64)
        if subset.ndim != 1:
            raise ValueError(f"subset_raw_idx must be rank-1, got shape={subset.shape}")
        if subset.size == 0:
            raise ValueError("subset_raw_idx must be non-empty.")
        if np.min(subset) < 0 or np.max(subset) >= self._raw_shape[0]:
            raise ValueError(
                f"subset_raw_idx contains out-of-range entries for dataset size {self._raw_shape[0]}"
            )

        self._raw_idx = np.sort(subset.copy())
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
