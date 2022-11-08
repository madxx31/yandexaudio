import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import faiss
import numpy as np
from datetime import datetime
import math


def position_discounter(position):
    return 1.0 / np.log2(position + 1)


def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg


def compute_dcg(query_artistid, ranked_list, labels):
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list):
        position = result_indx + 1
        discounted_position = position_discounter(position)
        if labels[result_trackid] == query_artistid:
            dcg += discounted_position
    return dcg


def compute_ndcg(labels, predictions, top_size=100):
    ndcg = []
    for query_artistid, ranked_list in zip(labels, predictions):
        query_artist_tracks_count = (labels == query_artistid).sum()
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count - 1, top_size=top_size)
        dcg = compute_dcg(query_artistid, ranked_list, labels)
        ndcg.append(dcg / ideal_dcg)
    return np.mean(ndcg)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0, k=1):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.k = k
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine_all = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = (cosine * self.cos_m - sine * self.sin_m).type(cosine.dtype)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class CNNEncoder(nn.Module):
    def __init__(self, output_features_size):
        super().__init__()
        self.output_features_size = output_features_size
        self.conv_1 = nn.Conv1d(512, output_features_size, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
        self.mp_1 = nn.MaxPool1d(2, 2)
        self.conv_3 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(self.output_features_size, self.output_features_size)
        self.linear2 = nn.Linear(self.output_features_size, 128, bias=False)

    def forward(self, batch):
        x = batch["input"]
        x = F.relu(self.conv_1(x.transpose(1, 2)))
        x = F.relu(self.conv_2(x))
        x = self.mp_1(x)
        x = F.relu(self.conv_3(x))
        x = self.conv_4(x).mean(axis=2)
        x = self.linear2(F.relu(self.linear1(x)))
        return x


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(1, input_dim, input_dim))
        self.b = nn.Parameter(torch.Tensor(1, 1, input_dim))
        self.u = nn.Parameter(torch.Tensor(input_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)
        nn.init.xavier_uniform_(self.u)

    def forward(self, x, mask=None):
        W = self.W.repeat(x.size()[0], 1, 1)
        b = self.b.repeat(x.size()[0], x.size()[1], 1)
        u = self.u.unsqueeze(0).repeat(x.size()[0], 1, 1)
        uit = torch.tanh(torch.bmm(x, W) + b)
        ait = torch.bmm(uit, u)
        if mask is not None:
            mask = mask.long()
            ait = ait.squeeze(-1).masked_fill(mask == 0, torch.finfo(ait.dtype).min)
        a = F.softmax(ait, dim=-1)
        a = a.unsqueeze(-1).repeat(1, 1, x.size()[2])
        weighted_input = x * a
        return torch.sum(weighted_input, dim=1)


class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=False):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x, mask):
        x = x.clamp(min=self.eps).pow(self.p)
        x = (x * mask).sum(axis=1) / mask.sum(axis=1)
        return x.pow(1.0 / self.p)


class LSTMEncoder(nn.Module):
    def __init__(self, output_features_size):
        super().__init__()
        self.output_features_size = output_features_size
        self.spatial_dropout = torch.nn.Dropout2d(p=0.15)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=output_features_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.attention = Attention(input_dim=self.output_features_size)
        self.gem = GeMPooling()
        self.linear1 = nn.Linear(self.output_features_size * 2, self.output_features_size)
        self.linear2 = nn.Linear(self.output_features_size, 128, bias=False)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, batch):
        inputs = batch["input"]
        inputs = inputs.permute(0, 2, 1)  # convert to [batch, channels, time]
        inputs = self.spatial_dropout(inputs)
        inputs = inputs.permute(0, 2, 1)  # back to [batch, time, channels]
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, batch["input_lengths"], batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out[:, :, : self.output_features_size] + out[:, :, self.output_features_size :]
        mask = batch["attention_mask"].unsqueeze(2).repeat(1, 1, out.shape[2])
        # max_pooling = out.masked_fill(mask == 0, torch.finfo(out.dtype).min).max(axis=1)[0]
        # avg_pooling = (out * mask).sum(axis=1) / mask.sum(axis=1)
        gem_pooling = self.gem(out, mask)
        x = torch.cat([self.attention(out, batch["attention_mask"]), gem_pooling], axis=1)
        x = self.linear2(F.relu(self.linear1(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, output_features_size):
        super().__init__()
        self.output_features_size = output_features_size
        self.spatial_dropout = torch.nn.Dropout2d(p=0.15)
        self.linear0 = nn.Linear(512, self.output_features_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_features_size, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.linear1 = nn.Linear(self.output_features_size * 2, self.output_features_size)
        self.linear2 = nn.Linear(self.output_features_size, 128, bias=False)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, batch):
        inputs = batch["input"]
        inputs = inputs.permute(0, 2, 1)  # convert to [batch, channels, time]
        inputs = self.spatial_dropout(inputs)
        inputs = inputs.permute(0, 2, 1)  # back to [batch, time, channels]
        inputs = F.relu(self.linear0(inputs))
        out = self.transformer_encoder(inputs, src_key_padding_mask=batch["attention_mask"])
        mask = batch["attention_mask"].unsqueeze(2).repeat(1, 1, out.shape[2])
        max_pooling = out.masked_fill(mask == 0, torch.finfo(out.dtype).min).max(axis=1)[0]
        avg_pooling = (out * mask).sum(axis=1) / mask.sum(axis=1)
        x = torch.cat([max_pooling, avg_pooling], axis=1)
        x = self.linear2(F.relu(self.linear1(x)))
        return x


class Model(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()
        self.train_loss = 0
        self.train_loss_cnt = 0

        # self.encoder = CNNEncoder(256)
        self.encoder = LSTMEncoder(160)
        # self.encoder = TransformerEncoder(256)
        self.arcface = ArcMarginProduct(
            128, self.cfg["model"]["num_labels"], s=self.cfg.arcface.s, m=self.cfg.arcface.m, k=self.cfg.arcface.k
        )

    def forward(self, batch):
        emb = self.encoder(batch)
        return self.arcface(emb, batch["labels"])

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, batch["labels"])
        self.train_loss += loss.item()
        self.train_loss_cnt += 1
        if self.train_loss_cnt == self.trainer.log_every_n_steps * self.trainer.accumulate_grad_batches:
            self.log("train/loss", self.train_loss / self.train_loss_cnt, on_step=False, on_epoch=True)
            self.train_loss = 0
            self.train_loss_cnt = 0
        return loss

    def validation_step(self, batch, batch_idx):
        emb = self.encoder(batch)
        return {"emb": emb, "labels": batch["labels"]}

    def validation_epoch_end(self, outputs):
        if len(outputs) < 10:
            return
        emb = torch.cat([x["emb"] for x in outputs], dim=0).detach().cpu().numpy().astype(np.float32)
        labels = torch.cat([x["labels"] for x in outputs], dim=0).detach().cpu().numpy()
        if self.cfg.faiss.distance == "dot":
            emb = emb / (((emb**2).sum(1)) ** 0.5).reshape(-1, 1).clip(min=1e-12)
            index = faiss.IndexFlatIP(emb.shape[1])
        else:
            index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
        _, I = index.search(emb, 101)
        ndcg = compute_ndcg(labels, I[:, 1:])
        self.log("eval/ndcg", ndcg, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # scheduler = CyclicLR(
        #     optimizer,
        #     base_lr=1e-5,
        #     max_lr=5e-4,
        #     cycle_momentum=False,
        #     step_size_up=4000,
        #     mode="triangular2",
        # )
        # lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
        # scheduler = {"scheduler": lr_scheduler, "monitor": "eval/auc"}
        # return [optimizer], [scheduler]
        return optimizer  # [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
