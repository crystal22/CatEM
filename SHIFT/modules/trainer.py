from data.create_hybrid_dataset import CreateDataset
from modules.SHIFT.combined import SHIFT
from gv_tools.util.logger import Logger
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, cfg_params, args, s_path, logger: Logger, result_logger: Logger):
        self.cfg = cfg_params
        self.args = args
        cfg_params.copyAttrib(self)
        self.__dict__.update(args.__dict__)
        self.logger = logger
        self.res_logger = result_logger
        self.s_path = s_path
        if not os.path.exists(self.s_path):
            os.mkdir(self.s_path)
        self.writer = SummaryWriter(os.path.join(self.s_path, "summary_log"))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_dataset()
        self.build_model()
        self.best_val_loss = float('inf')

    def get_dataset(self):
        self.dataset = CreateDataset(cfg_params=self.cfg, batch_size=self.args.batch_size)
        self.train_loader, self.val_loader, self.test_loader = self.dataset.get_dataloader()
        self.tokenizer = self.dataset.tokenizer
        self.vocab = self.dataset.tokenizer.get_vocab()
        self.pad_id = self.vocab["<pad>"]
        self.num_token = len(self.vocab)

    def get_loss(self, nlp_out, nlp_trg, mob_out, mob_trg):
        nlp_branch = F.cross_entropy(nlp_out, nlp_trg, ignore_index=self.pad_id)
        mob_branch = F.mse_loss(mob_out, mob_trg)
        combined_loss = self.args.loss_alpha * mob_branch + (1 - self.args.loss_alpha) * nlp_branch

        return combined_loss

    def build_model(self):

        self.model = SHIFT(self.cfg, num_tokens=self.num_token, hid_dim=self.args.hidden_size,
                           pad_idx=self.pad_id, dropout=self.args.drop_out, mode=self.args.mode,
                           momentum_alpha=self.args.m_alpha).to(self.device)
        self.model.apply(self.initialize_weights)
        self.logger.log(f'The model has {self.count_parameters(self.model):,} trainable parameters')

    def train(self):
        epoch = 0
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        if self.args.lr_decay:
            scheduler = ReduceLROnPlateau(model_optim, 'min', factor=0.5, patience=self.args.patience)

        while epoch < self.args.train_epochs:
            epoch += 1
            train_loss = []
            self.model.train()
            for i, (src, trg, x, y) in tqdm(enumerate(self.train_loader)):
                src = src.to(self.device)
                trg = trg.to(self.device)
                x = x.to(self.device)
                y = y.to(self.device)
                model_optim.zero_grad()
                mob_out, nlp_out, _ = self.model(src, trg[:, :-1], x)
                output_dim = nlp_out.shape[-1]
                # remove eos token
                nlp_out = nlp_out.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                loss = self.get_loss(nlp_out, trg, mob_out, y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                model_optim.step()
                train_loss.append(loss.item())

            if self.args.lr_decay:
                scheduler.step(np.mean(train_loss))

            lr = model_optim.param_groups[0]['lr']
            print(f"LR: {lr}")
            self.logger.field("epoch", epoch)
            self.logger.field("train loss", np.mean(train_loss))
            if epoch % self.args.interval_val_epochs == 0 and epoch != 0:
                self.logger.log("interval validation")
                val_loss = self.validation()
                self.res_logger.field('Val Epoch', epoch)
                self.res_logger.field('Val loss', val_loss)
                if val_loss < self.best_val_loss:
                    print("[!] saving best val model...")
                    self.save_model(name="best_val")
                    self.best_val_loss = val_loss

        self.save_model(name="end")

    def inference(self, trained_model_path):
        self.model.load_state_dict(torch.load(trained_model_path))
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for i, (src, trg, _, _) in tqdm(enumerate(self.test_loader)):
                src_tensor = src.to(self.device)
                trg = trg.to(self.device)

                trues.extend(self.tokenizer.batch_decode(trg.squeeze().to(torch.int64), skip_special_tokens=True))
                src_mask = self.model.branch_nlp.make_src_mask(src_tensor)
                enc_src = self.model.branch_nlp.forward_enc(src_tensor, src_mask)
                trg_indexes = np.array([self.vocab["<s>"]] * trg.size(0)).reshape([-1, 1])
                end_token = False
                for j in range(self.output_file_max_len-1):

                    trg_tensor = torch.from_numpy(trg_indexes).to(self.device)
                    trg_mask = self.model.branch_nlp.make_trg_mask(trg_tensor)
                    output, attention = self.model.branch_nlp.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                    output = F.log_softmax(output[:, -1, :].unsqueeze(1), dim=-1)
                    out_token = torch.argmax(output, dim=-1)
                    pred_token = out_token.cpu().detach().numpy()
                    if end_token:
                        pred_token[end_token_location] = self.vocab["</s>"]
                    if self.vocab["</s>"] in pred_token:
                        end_token = True
                        end_token_location = (pred_token[:, 0] == self.vocab["</s>"])
                    trg_indexes = np.concatenate((trg_indexes, pred_token), axis=-1)

                detokenized = self.tokenizer.batch_decode(trg_indexes[:, 1:], skip_special_tokens=True)
                preds.extend(detokenized)
            with open(os.path.join(self.s_path, "gt_decoded.txt"), "w") as f:
                for s in trues:
                    f.write(s)
                    f.write("\n")
                f.close()
            with open(os.path.join(self.s_path, "pred_decoded.txt"), "w") as f:
                for s in preds:
                    f.write(s)
                    f.write("\n")
                f.close()

    def validation(self):
        self.model.eval()
        val_loss = []

        with torch.no_grad():
            for i, (src, trg, x, y) in tqdm(enumerate(self.val_loader)):
                src = src.to(self.device)
                trg = trg.to(self.device)
                x = x.to(self.device)
                y = y.to(self.device)
                mob_out, nlp_out, _ = self.model(src, trg[:, :-1], x)
                output_dim = nlp_out.shape[-1]
                nlp_out = nlp_out.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                loss = self.get_loss(nlp_out, trg, mob_out, y.float())
                val_loss.append(loss.item())

        return np.mean(val_loss)

    def save_model(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.s_path, name + '_params.pkl'))

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

