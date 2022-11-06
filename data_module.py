import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


class AudioDataset(Dataset):
    def __init__(self, features, lengths, labels):
        self.tracks = features
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return {"input": self.tracks[idx], "length": self.lengths[idx], "label": self.labels[idx]}


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        s = pd.read_csv("train_meta.tsv", sep="\t")
        try:
            track_features = np.load("train.npy")
            track_lengths = np.load("train_lengths.npy")
        except:
            track_features = np.zeros((len(s), 81, 512), np.float16)
            track_lengths = []
            for i, f in enumerate(tqdm(s.archive_features_path.values)):
                tmp = np.load("train_features/" + f).astype(np.float16).T
                track_features[i, : tmp.shape[0], :] = tmp
                track_lengths.append(tmp.shape[0])
        if self.cfg["debug"]:
            s = s.sample(1000)
        kf = KFold(n_splits=4, random_state=17, shuffle=True)
        for f, (_, a) in enumerate(kf.split(s.artistid.unique())):
            s.loc[s.artistid.isin(a), "fold"] = f
        tr_ind = s[s.fold != self.cfg["data_module"]["fold"]].index.values
        self.train_dataset = AudioDataset(track_features[tr_ind], track_lengths[tr_ind], s.artistid[tr_ind].values - 1)
        tr_ind = s[s.fold == self.cfg["data_module"]["fold"]].index.values
        self.val_dataset = AudioDataset(track_features[tr_ind], track_lengths[tr_ind], s.artistid[tr_ind].values - 1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collator,
            num_workers=4,
            batch_size=self.cfg["data_module"]["train_batch_size"],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.collator,
            num_workers=4,
            batch_size=self.cfg["data_module"]["train_batch_size"],
        )

    def collator(self, batch):
        batch_size = len(batch)
        input_lengths = [i["length"] for i in batch]
        max_len = max(input_lengths)
        input = np.zeros((batch_size, 81, 512))
        attention_mask = np.zeros((batch_size, max_len))
        labels = np.ones(batch_size) * -100
        for i, b in enumerate(batch):
            input[i, : len(b["input"])] = b["input"]
            attention_mask[i, : b["length"]] = 1
            labels[i] = b["label"]
        return {
            "input_lengths": input_lengths,
            "input": torch.tensor(input, dtype=torch.float),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
