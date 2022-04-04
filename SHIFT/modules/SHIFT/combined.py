"""
two branches: one with mob, one with sentence (NLP branch)
modes of merging two branches:
'separate' mode: basic mode, two branches are separate with their own parameter
'siamese' mode: two branches have the same encoder
'momentum' mode: update mob branch encoder with NLP branch encoder parameter
"""
import torch
import torch.nn as nn
from modules.SHIFT.nlp_branch import Encoder, Decoder, Seq2Seq
from modules.SHIFT.mob_branch import MobTransformer


class SHIFT(nn.Module):
    def __init__(self, cfg_params, num_tokens, hid_dim, pad_idx, dropout, mode, momentum_alpha):
        super(SHIFT, self).__init__()
        self.mode = mode
        cfg_params.copyAttrib(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_nlp = Encoder(hid_dim,
                                   n_layers=3,
                                   n_heads=8,
                                   pf_dim=512,
                                   dropout=dropout,
                                   device=self.device)
        if self.mode == "siamese":
            self.branch_mob = MobTransformer(cfg_params, self.encoder_nlp, hid_dim, dropout,
                                             self.device, momentum_alpha)
        else:
            self.encoder_mob = Encoder(hid_dim,
                                       n_layers=3,
                                       n_heads=8,
                                       pf_dim=512,
                                       dropout=dropout,
                                       device=self.device)
            self.branch_mob = MobTransformer(cfg_params, self.encoder_mob, hid_dim, dropout,
                                             self.device, momentum_alpha)
        self.decoder_nlp = Decoder(num_tokens,
                                   hid_dim,
                                   n_layers=3,
                                   n_heads=8,
                                   pf_dim=512,
                                   dropout=dropout,
                                   device=self.device,
                                   max_length=self.output_file_max_len)
        self.branch_nlp = Seq2Seq(num_tokens, hid_dim, self.input_file_max_len,
                                  self.encoder_nlp, self.decoder_nlp, pad_idx, dropout, self.device)

        if self.mode == "momentum":
            self.momentum_init()

    def forward(self, src, trg, x):
        nlp_out, nlp_attention = self.branch_nlp(src, trg)
        if self.mode == "momentum":
            mob_out = self.branch_mob(x, momentum=True, nlp_encoder=self.encoder_nlp)
        else:
            mob_out = self.branch_mob(x)

        return mob_out, nlp_out, nlp_attention

    def momentum_init(self):
        for param_q, param_k in zip(self.encoder_nlp.parameters(), self.encoder_mob.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


