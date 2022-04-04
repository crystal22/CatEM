from cfg.option import Options
from gv_tools.util.logger import Logger
import os
from modules.trainer import Trainer
import argparse
import torch
import numpy as np
import random


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('-cfg', '--config_file',
                      default='cfg/example.cfg', type=str,
                      help="Configure file containing parameters for the algorithm")
    args.add_argument('-s', '--save_path', default='results/example',
                      type=str)
    args.add_argument('--mode', default='momentum',
                      type=str)
    args.add_argument('--loss_alpha', default=0.01,
                      type=float)
    args.add_argument('--m_alpha', default=0.001,
                      type=float)
    args.add_argument('-val', '--val', default=True,
                      type=str2bool)
    args.add_argument('-bs', '--batch_size', default=128,
                      type=int)
    args.add_argument('--patience', default=5,
                      type=int)
    args.add_argument('-p', '--pretrain_model', default=None,
                      type=str)
    args.add_argument('-grad_clip', type=float, default=10)
    args.add_argument('-lr', '--learning_rate', default=0.0001,
                      type=float)
    args.add_argument('-hs', '--hidden_size', default=256,
                      type=int)
    args.add_argument('--interval_val_epochs', default=10,
                      type=int)
    args.add_argument('-dp', '--drop_out', default=0.2,
                      type=float)
    args.add_argument('-mr', '--multi_run', default=5,
                      type=int)
    args.add_argument('-se', '--seed', default=0,
                      type=int)
    args.add_argument('-ld', '--lr_decay', default=True,
                      type=str2bool)
    args.add_argument('--output_attention', default=False,
                      type=str2bool)
    args.add_argument('-e', '--train_epochs', default=36,
                      type=int)

    return args.parse_args()


# ------------------------------------------------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    args = get_args()
    if args.seed > 0:
        setup_seed(args.seed)

    params = Options(args.config_file)
    for run in range(args.multi_run):
        s_path = os.path.join(args.save_path, str(run+1))

        logger = Logger()
        logger.attach_file_handler(s_path, "train")
        result_logger = Logger()
        result_logger.attach_file_handler(s_path, "val")
        t = Trainer(params, args, s_path, logger, result_logger)
        t.train()

