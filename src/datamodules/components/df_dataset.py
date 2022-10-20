from typing import Optional, Callable

import pandas as pd
from torch.utils.data import Dataset


class DFDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            message_col: str = "message",
            category_col: str = "category",
            id_col: str = "id",
            preprocess: Optional[Callable] = None,
    ):
        super().__init__()
        self.categories = df[category_col]
        self.messages = df[message_col]
        self.ids = df[id_col]
        self.preprocess = preprocess

    def __getitem__(self, item):
        if not 0 <= item < len(self):
            raise ValueError(f"item should be from {0} to {len(self)}, got {item}")
        message, category, _id = self.messages[item], self.categories[item], self.ids[item]
        if self.preprocess is not None:
            message = self.preprocess(message)
        return {
            "id": _id,
            "message": message,
            "category": category,
        }

    def __len__(self):
        return len(self.messages)
