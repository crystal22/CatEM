import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MobTransformer(nn.Module):
    def __init__(self, cfg_params, encoder, hid_dim, dropout, device, momentum_alpha):
        super(MobTransformer, self).__init__()
        cfg_params.copyAttrib(self)
        self.encoder = encoder
        self.m = momentum_alpha
        self.mob_input_embedding = nn.Linear(1, hid_dim)
        self.pos_encoder = PositionalEncoding(hid_dim, dropout, max_len=self.obs_len)
        self.fc = nn.Linear(hid_dim, 1)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, x, momentum=False, nlp_encoder=None):
        src = self.mob_input_embedding(x.float().unsqueeze(-1)) * self.scale
        h = src.permute(1, 0, 2)
        # sequence to the encoder
        transformer_input = self.pos_encoder(h)
        src = transformer_input.permute(1, 0, 2)
        src_mask = None
        if momentum:
            with torch.no_grad():  # no gradient
                self._momentum_update_key_encoder(nlp_encoder)
                enc_src = self.encoder(src, src_mask)
        else:
            enc_src = self.encoder(src, src_mask)
        score = self.fc(enc_src[:, -1, :])
        score = F.relu(score)

        return score

    @torch.no_grad()
    def _momentum_update_key_encoder(self, encoder_nlp):
        for param_q, param_k in zip(encoder_nlp.parameters(), self.encoder.parameters()):
            param_k.data = param_k.data * (1. - self.m) + param_q.data * self.m


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

