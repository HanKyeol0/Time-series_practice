import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.SelfAttention_Family import AnomalyAttention, AttentionLayer
from layers.Embed import DataEmbedding, TokenEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, configs):
        super(AnomalyTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.output_attention = configs.output_attention
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.activation = configs.activation
        self.e_layers = configs.e_layers
        self.c_out = configs.c_out
        self.k = configs.k

        self.criterion = nn.MSELoss()

        # Encoding
        self.embedding = DataEmbedding(self.enc_in, self.d_model, self.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(self.seq_len, False, attention_dropout=self.dropout, output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def my_kl_loss(p, q):
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

    def forward(self, x, x_mark, label=None, mode='TRAIN'):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        
        rec_loss = torch.mean(self.criterion(enc_out, x))

        if mode == 'TRAIN' or mode == 'VAL':
            # calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(self.my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_len)).detach())) + torch.mean(
                        self.my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_len)).detach(), series[u])))
                prior_loss += (torch.mean(self.my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_len)),
                    series[u].detach())) + torch.mean(
                    self.my_kl_loss(series[u].detach(), (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.seq_len)))))
            
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss1 = rec_loss - self.k * series_loss
            loss2 = rec_loss + self.k * prior_loss
            loss = loss1 + loss2

            return enc_out, loss

        elif mode == 'TEST':
            temperature = 50
            anomaly_ratio = 0.1

            # Evaluation on the test set
            for u in range(len(prior)):
                if u == 0:
                    series_loss = self.my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = self.my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += self.my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += self.my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            score = metric * loss
            score = score.detach().cpu().numpy()

            return enc_out, loss, score