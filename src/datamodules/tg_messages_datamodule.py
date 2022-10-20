import itertools
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import torch
import os
from nltk import TweetTokenizer
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path

from src.datamodules.components.df_dataset import DFDataset
from src.utils.collators import collate_batch
from src.utils.utils import prepare_vocab


class TgMessagesDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
            self,
            data_dir: str = "data/",
            train_val_split: Tuple[float, float] = (1., 0.),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.tokenizer = TweetTokenizer()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.vocab_size: int = 0

    @property
    def num_classes(self):
        return 3

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        data_dir = Path(self.hparams.data_dir)
        download = False
        try:
            if not (data_dir / 'default' / 'train_data.csv').is_file() or \
                    not (data_dir / 'default' / 'train_solution.csv').is_file() or \
                    not (data_dir / 'default' / 'test_data.csv').is_file():
                download = True
        except FileNotFoundError as e:
            download = True

        if download:
            os.makedirs(str(data_dir / 'default'), exist_ok=True)
            os.system(
                """wget https://codalab.lisn.upsaclay.fr/my/datasets/"""
                """download/dc591346-f10f-4b61-8932-d7a170e80dbf -O """
                f"""{str(data_dir / 'default' / 'public_data.zip')}"""
            )
            os.system(
                f"""unzip {str(data_dir / 'default' / 'public_data.zip')}"""
                f""" -d {str(data_dir / 'default')}"""
            )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            df_train, df_test = read_dfs(self.hparams.data_dir)

            def tokenize_preprocess(text: str) -> str:
                return " ".join(self.tokenizer.tokenize(text.lower()))

            vocab = prepare_vocab(
                itertools.chain(*(tokenize_preprocess(message).split() for message in df_train.message))
            )
            self.vocab_size = len(vocab)
            print(f"{self.vocab_size=}")

            def str2id_preprocess(text: str) -> List[int]:
                return vocab(tokenize_preprocess(text).split())

            train_dataset = DFDataset(df_train, preprocess=str2id_preprocess)
            train_size = int(len(train_dataset) * self.hparams.train_val_split[0])
            val_size = len(train_dataset) - train_size
            self.data_train, self.data_val = random_split(
                dataset=train_dataset,
                lengths=(train_size, val_size),
                generator=torch.Generator().manual_seed(42),
            )

            self.data_test = DFDataset(df_test, preprocess=str2id_preprocess)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_batch,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_batch,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {
            "vocab_size": self.vocab_size
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


def read_dfs(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = os.path.join(data_dir, 'default')
    data = pd.read_csv(os.path.join(path, 'train_data.csv'), index_col="id")
    solution = pd.read_csv(os.path.join(path, 'train_solution.csv'), index_col="id")
    df_train = data.join(solution, on="id").reset_index()
    df_test = pd.read_csv(os.path.join(path, 'test_data.csv'), index_col="id").reset_index()
    df_test['category'] = -1
    return df_train, df_test


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
