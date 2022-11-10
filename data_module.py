import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


class AudioDataset(Dataset):
    def __init__(self, artist_tracks, features, lengths, labels, examples_per_artist=2, augment=False, seed=17):
        self.artist_tracks = artist_tracks
        self.tracks = features
        self.lengths = lengths
        self.labels = labels
        self.examples_per_artist = examples_per_artist
        self.augment = augment
        self.chunks = []
        self.rng = np.random.default_rng(seed)
        self._get_chunks()

    def _split_list(self, l):
        return [l[i : i + self.examples_per_artist] for i in range(0, len(l), self.examples_per_artist)]

    def _get_chunks(self):
        self.chunks = []
        for track_list in self.artist_tracks:
            self.rng.shuffle(track_list)
            self.chunks += self._split_list(track_list)

    def get_track(self, i):
        if self.augment:
            l = self.lengths[i]
            newl = int(self.rng.uniform(40, l))
            start = int(self.rng.uniform(0, l - newl))
            return {"input": self.tracks[i][start : start + newl, :], "length": newl, "label": self.labels[i]}
        else:
            return {"input": self.tracks[i], "length": self.lengths[i], "label": self.labels[i]}

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return [self.get_track(i) for i in self.chunks[idx]]


class SimpleAudioDataset(Dataset):
    def __init__(self, features, lengths):
        self.tracks = features
        self.lengths = lengths

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return [{"input": self.tracks[idx], "length": self.lengths[idx], "label": -100}]


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
            np.save("train.npy", track_features)
            np.save("train_lengths.npy", track_lengths)
        if self.cfg["debug"]:
            s = s.sample(1000)
        kf = KFold(n_splits=4, random_state=17, shuffle=True)
        for f, (_, a) in enumerate(kf.split(s.artistid.unique())):
            s.loc[s.artistid.isin(a), "fold"] = f
        s.artistid -= 1
        s.trackid -= 1
        train_meta = s[s.fold != self.cfg["data_module"]["fold"]]
        train_artist_tracks = train_meta.groupby("artistid").trackid.agg(list)
        self.train_dataset = AudioDataset(
            train_artist_tracks,
            track_features,
            track_lengths,
            s.artistid.values,
            examples_per_artist=self.cfg["data_module"]["examples_per_artist"],
            augment=True,
        )
        val_meta = s[s.fold == self.cfg["data_module"]["fold"]]
        val_artist_tracks = val_meta.groupby("artistid").trackid.agg(list)
        self.val_dataset = AudioDataset(
            val_artist_tracks,
            track_features,
            track_lengths,
            s.artistid.values,
            examples_per_artist=self.cfg["data_module"]["examples_per_artist"],
        )
        # submit
        if self.cfg["submit"]:
            s = pd.read_csv("test_meta.tsv", sep="\t")
            self.test_ids = s.trackid.astype(str).values
            try:
                test_track_features = np.load("test.npy")
                test_track_lengths = np.load("test_lengths.npy")
            except:
                test_track_features = np.zeros((len(s), 81, 512), np.float16)
                test_track_lengths = []
                for i, f in enumerate(tqdm(s.archive_features_path.values)):
                    tmp = np.load("test_features/" + f).astype(np.float16).T
                    test_track_features[i, : tmp.shape[0], :] = tmp
                    test_track_lengths.append(tmp.shape[0])
                np.save("test.npy", test_track_features)
                np.save("test_lengths.npy", test_track_lengths)
            self.test_dataset = SimpleAudioDataset(
                test_track_features,
                test_track_lengths,
            )

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
            batch_size=self.cfg["data_module"]["eval_batch_size"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collator,
            num_workers=4,
            batch_size=self.cfg["data_module"]["eval_batch_size"],
        )

    def collator(self, batch):
        batch = sum(batch, [])
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
