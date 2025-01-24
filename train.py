import matplotlib
from earthformer.datasets.sevir.sevir_dataloader import PREPROCESS_SCALE_01, PREPROCESS_OFFSET_01
from matplotlib import pyplot as plt, animation

matplotlib.use('agg')

import sys

sys.path.append('../../..')
r"""
import wandb

wandb.init(
    entity="loongs",
    project="SEVIR",
    name="dev/vil/full-2",
)
"""

import matplotlib.animation as animation
import cartopy.crs as ccrs
from thop import profile
import time
import warnings
from typing import Union, Dict
from shutil import copyfile
from copy import deepcopy
import inspect
import pickle
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torchvision import models
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, MultiplicativeLR, ConstantLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange
from pytorch_lightning import Trainer, seed_everything
from src.earthformer.config import cfg
from src.earthformer.utils.optim import SequentialLR, warmup_lambda, cos_lambda
from src.earthformer.utils.utils import get_parameter_names
from src.earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict, s3_download_pretrained_ckpt
from src.earthformer.utils.layout import layout_to_in_out_slice
from src.earthformer.visualization.sevir.sevir_vis_seq import save_example_vis_results
from src.earthformer.metrics.sevir import SEVIRSkillScore
from src.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from src.earthformer.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
from src.earthformer.utils.apex_ddp import ApexDDPStrategy
from src.earthformer.datasets.weatherbench.weatherdata import WeatherLightningDataModule
from src.earthformer.datasets.weatherbench.weatherdata import BaseDataModule
from src.earthformer.datasets.weatherbench.config import get_config
_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")
pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir
pytorch_state_dict_name = "earthformer_sevir.pt"
# _input_types = ['vil', 'vis', 'ir069', 'ir107', 'lght']  # ['vil', 'vis', 'ir069', 'ir107', 'lght']
_input_types = ['r', 't', 'u', 'v']

# torch.autograd.detect_anomaly(check_nan=True)

def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = y.reshape(b, ch, w * h)
    gram = torch.matmul(features, features.permute(0, 2, 1)) / (ch * w * h)
    return gram


class PerceptualLoss(_Loss):

    def __init__(self, end_layer=22):
        super(PerceptualLoss, self).__init__()
        self.model = models.vgg16(pretrained=True).features[1:end_layer]
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()
        self.model.eval()
        self.criterion = nn.MSELoss()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs, targets):

        # preprocess on data shape
        b, t, ih, iw, c = inputs.shape
        b, t, th, tw, c = targets.shape
        shape = (48, 48)
        inputs = inputs.permute(0, 1, 4, 2, 3).reshape(b * t, c, ih, iw)
        targets = targets.permute(0, 1, 4, 2, 3).reshape(b * t, c, th, tw)

        scale_w = shape[0] / ih
        scale_h = shape[1] / iw
        # 插值到48*48
        inputs = F.interpolate(input=inputs, scale_factor=(scale_h, scale_w))
        targets = F.interpolate(input=targets, scale_factor=(scale_h, scale_w))

        inputs = torch.cat([inputs] * 64, dim=1)
        targets = torch.cat([targets] * 64, dim=1)

        feature_reconstruction_loss = torch.tensor(0, dtype=inputs.dtype, device=inputs.device)
        style_reconstruction_loss = torch.tensor(0, dtype=inputs.dtype, device=inputs.device)
        x_input = inputs
        x_target = targets
        for idx, module in enumerate(self.model):
            x_input = module(x_input)
            x_target = module(x_target)
            if idx in [3, 8, 15, 22]:
                tb, tc, th, tw = x_input.shape
                # print(x_input.shape)  [26,64,24,24] [26,128,12,12] [26,256,6,6]
                feature_reconstruction_loss += self.criterion(x_input, x_target) / (tc * th * tw)
                style_reconstruction_loss += self.criterion(gram_matrix(x_input), gram_matrix(x_target))
        loss = feature_reconstruction_loss + style_reconstruction_loss

        return loss


def cal_vil_dice_160(pred, target, tau=1):
    scale = PREPROCESS_SCALE_01['vil']  # 1 / 255
    offset = PREPROCESS_OFFSET_01['vil']  # 0
    thresholds = [16, 74, 133, 160, 181, 219]
    eps = 1e-6

    pred = pred / scale - offset
    target = target / scale - offset
    p = torch.sigmoid((pred - 160) / tau)
    # p = 1 / (1 + torch.exp(-(pred - 160) / tau))
    t = (target > 160).float()
    inter = (p * t).sum()
    uni = p.sum() + t.sum()
    dice = 2 * inter / (uni + eps)
    return 1 - dice

    # is_nan = torch.logical_or(torch.isnan(p), torch.isnan(t))
    # p[is_nan] = 0
    # t[is_nan] = 0


def cal_vil_metrics(pred, target, tau=0.01):
    scale = PREPROCESS_SCALE_01['vil']  # 1 / 255
    offset = PREPROCESS_OFFSET_01['vil']  # 0
    thresholds = [16, 74, 133, 160, 181, 219]
    eps = 1e-6

    pred = pred / scale - offset
    target = target / scale - offset
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)

    ret = {'csi': [], 'pod': [], 'sucr': [], 'log_bias': [], 'thresholds': thresholds}
    for T in thresholds:
        p = torch.sigmoid((pred - T) / tau)
        t = torch.sigmoid((target - T) / tau)
        hits = torch.sum(t * p, dim=-1)  # TP
        misses = torch.sum(t * (1 - p), dim=-1)  # FN
        fas = torch.sum((1 - t) * p, dim=-1)  # FP
        csi = hits / (hits + misses + fas + eps)
        pod = hits / (hits + misses + eps)
        sucr = hits / (hits + fas + eps)
        bias = (hits + fas) / (hits + misses + eps)
        log_bias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        ret['csi'].append(csi)
        ret['pod'].append(pod)
        ret['sucr'].append(sucr)
        ret['log_bias'].append(log_bias)
    return ret


class TimerCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_cumulative_time = 0
        self.val_cumulative_time = 0
        self.test_cumulative_time = 0

    def on_train_start(self, trainer, pl_module):
        self.train_cumulative_time = 0
        self.val_cumulative_time = 0
        self.test_cumulative_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.train_epoch_start_time
        self.train_cumulative_time += epoch_time
        print(f"Training - Epoch {trainer.current_epoch} time: {epoch_time:.2f} seconds")
        print(f"  - Total cumulative training time up to now: {self.train_cumulative_time:.2f} seconds")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_epoch_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.val_epoch_start_time
        self.val_cumulative_time += epoch_time
        print(f"Validation - Epoch {trainer.current_epoch} time: {epoch_time:.2f} seconds")
        print(f"  - Total cumulative validation time up to now: {self.val_cumulative_time:.2f} seconds")

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_epoch_start_time = time.time()

    def on_test_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.test_epoch_start_time
        self.test_cumulative_time += epoch_time
        print(f"Testing - Epoch {trainer.current_epoch} time: {epoch_time:.2f} seconds")
        print(f"  - Total cumulative testing time up to now: {self.test_cumulative_time:.2f} seconds")

    def on_train_end(self, trainer, pl_module):
        print(f"Total cumulative training time: {self.train_cumulative_time:.2f} seconds")
        print(f"Total cumulative validation time: {self.val_cumulative_time:.2f} seconds")

    def on_test_end(self, trainer, pl_module):
        print(f"Total cumulative testing time: {self.test_cumulative_time:.2f} seconds")


class CuboidSEVIRPLModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None,
                 dm: BaseDataModule = None):
        super(CuboidSEVIRPLModule, self).__init__()
        self._max_train_iter = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        self.dm = dm
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.torch_nn_module = CuboidTransformerModel(
            input_types=_input_types,
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.configure_save(cfg_file_path=oc_file)
        # evaluation
        self.metrics_list = oc.dataset.metrics_list
        self.threshold_list = oc.dataset.threshold_list
        self.metrics_mode = oc.dataset.metrics_mode
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_outputs = []
        self.valid_score = SEVIRSkillScore(
            mode=self.metrics_mode,
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4, )
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_score = SEVIRSkillScore(
            mode=self.metrics_mode,
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4, )
        # self.perceptual_loss = PerceptualLoss()

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.dataset = self.get_dataset_config()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_dataset_config():
        oc = OmegaConf.create()
        oc.dataset_name = "Weatherbench"
        oc.img_height = 32
        oc.img_width = 64
        oc.in_len = 4
        oc.out_len = 4
        oc.seq_len = 8
        oc.plot_stride = 2
        oc.interval_real_time = 5
        oc.sample_mode = "sequent"
        oc.stride = oc.out_len
        oc.layout = "NTHWC"
        oc.start_date = None
        oc.train_val_split_date = (2019, 1, 1)
        oc.train_test_split_date = (2019, 6, 1)
        oc.end_date = None
        oc.metrics_mode = "0"
        oc.metrics_list = ('csi', 'pod', 'sucr', 'bias')
        oc.threshold_list = (16, 74, 133, 160, 181, 219)
        return oc

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        height = dataset_oc.img_height
        width = dataset_oc.img_width
        in_len = dataset_oc.in_len
        out_len = dataset_oc.out_len
        data_channels = 1
        cfg.input_shape = (in_len, height, width, data_channels)
        cfg.target_shape = (out_len, height, width, data_channels)

        cfg.base_units = 64
        cfg.block_units = None  # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'
        cfg.checkpoint_level = 2
        # initial downsample and final upsample
        cfg.initial_downsample_type = "stack_conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_stack_conv_num_layers = 3
        cfg.initial_downsample_stack_conv_dim_list = [4, 16, cfg.base_units]
        cfg.initial_downsample_stack_conv_downscale_list = [1, 1, 1]
        cfg.initial_downsample_stack_conv_num_conv_list = [2, 2, 2]
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @classmethod
    def get_layout_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.in_len = dataset_oc.in_len
        oc.out_len = dataset_oc.out_len
        oc.layout = dataset_oc.layout
        return oc

    @staticmethod
    def get_optim_config():
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 32
        oc.micro_batch_size = 8

        oc.method = "adamw"
        oc.lr = 1E-3
        oc.wd = 1E-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 100
        # scheduler
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        oc.min_lr_ratio = 0.1
        oc.warmup_min_lr_ratio = 0.1
        # early stopping
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 20
        oc.save_top_k = 1
        return oc

    @staticmethod
    def get_logging_config():
        oc = OmegaConf.create()
        oc.logging_prefix = "WeatherBench"
        oc.monitor_lr = True
        oc.monitor_device = False
        oc.track_grad_norm = -1
        oc.use_wandb = False
        return oc

    @staticmethod
    def get_trainer_config():
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 1
        oc.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        oc.precision = 32
        return oc

    @classmethod
    def get_vis_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.train_example_data_idx_list = [0, ]
        oc.val_example_data_idx_list = [80, ]
        oc.test_example_data_idx_list = [0, 80, 160, 240, 320, 400]
        oc.eval_example_only = False
        oc.plot_stride = dataset_oc.plot_stride
        return oc

    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters()
                       if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters()
                       if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))
        steps_per_epoch = int(np.round(self.total_num_steps / self.oc.optim.max_epochs))
        cosine_end_epochs = self.oc.optim.cosine_end_epochs
        # cosine_end_steps = self.total_num_steps - warmup_iter
        cosine_end_steps = cosine_end_epochs * steps_per_epoch
        min_lr = self.oc.optim.min_lr_ratio * self.oc.optim.lr

        if self.oc.optim.lr_scheduler_mode == 'cosine':
            # warmup_scheduler = LambdaLR(optimizer,
            #                             lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
            #                                                     min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            #
            # cos_scheduler1 = LambdaLR(optimizer, lr_lambda=cos_lambda(max_steps=cosine_end_steps - warmup_iter,
            #                                                           min_ratio=self.oc.optim.min_lr_ratio))
            # cosine_scheduler = CosineAnnealingLR(optimizer,
            #                                      T_max=(cosine_end_steps - warmup_iter),
            #                                      eta_min=min_lr)
            # const_scheduler = ConstantLR(optimizer, factor=self.oc.optim.min_lr_ratio,
            #                             total_iters=self.total_num_steps - cosine_end_steps + 10)
            # lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cos_scheduler1, const_scheduler],
            #                             milestones=[warmup_iter, cosine_end_steps])

            cos_scheduler = LambdaLR(optimizer, lr_lambda=cos_lambda(max_steps=cosine_end_steps,
                                                                      min_ratio=self.oc.optim.min_lr_ratio))
            const_scheduler = ConstantLR(optimizer, factor=self.oc.optim.min_lr_ratio,
                                         total_iters=self.total_num_steps - cosine_end_steps + 10)
            lr_scheduler = SequentialLR(optimizer, schedulers=[cos_scheduler, const_scheduler],
                                        milestones=[cosine_end_steps])

            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise NotImplementedError
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]

        callbacks += [TimerCallback(), ]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir, name="tensorboardlog")
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir, name="csvlog")
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir, name="wandblog", log_model="all")
            logger += [wandb_logger, ]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            track_grad_norm=self.oc.logging.track_grad_norm,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            # strategy="ddp",
            strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_weather_datamodule(dataset_oc,
                             micro_batch_size: int = 1,
                             num_workers: int = 8):
        config = get_config()
        dm = WeatherLightningDataModule('weather_mv_4_4_s6_5_625', config).get_data()
        return dm

    @property
    def in_slice(self):
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

    def forward(self, x, y):
        output, loss, feature = self.torch_nn_module(x, y)
        return output, loss, feature

    def training_step(self, batch, batch_idx):
        # TCHW->BTHWC
        a,b = batch
        x_data = rearrange(a, 'B T C H W ->B T H W C')
        y_label = rearrange(b, 'B T C H W ->B T H W C')
        y = y_label[:,:,:,:,3:6]
        x = {
            'r': x_data[:,:,:,:,:3],
            't': x_data[:, :, :, :, 3:6],
            'u': x_data[:, :, :, :, 6:9],
            'v': x_data[:, :, :, :, 9:12],
            'rf': y_label[:,:,:,:,:3],
            'uf': y_label[:, :, :, :, 6:9],
            'vf': y_label[:, :, :, :, 9:12]
        }
        y_hat, loss,_ = self(x, y)
        mse_loss,phy_loss,phycell_loss,loss_future = loss
        loss = mse_loss + 0.1*phy_loss + 0.1*phycell_loss + 0.1*loss_future
        micro_batch_size = y_label.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        self.save_vis_step_end(
            data_idx=data_idx,
            in_seq=x['t'],
            target_seq=y,
            pred_seq=y_hat,
            mode="train"
        )
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        self.log('train_phy_loss', phy_loss, on_step=True, on_epoch=False)
        self.log('train_phycell_loss', phycell_loss, on_step=True, on_epoch=False)
        self.log('train_loss_future', loss_future, on_step=True, on_epoch=False)
        self.log('train_mse_loss', mse_loss, on_step=True, on_epoch=False)
        return loss

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        a, b = batch
        x_data = rearrange(a, 'B T C H W ->B T H W C')
        y_label = rearrange(b, 'B T C H W ->B T H W C')
        y = y_label[:, :, :, :, 3:6]
        x = {
            'r': x_data[:, :, :, :, :3],
            't': x_data[:, :, :, :, 3:6],
            'u': x_data[:, :, :, :, 6:9],
            'v': x_data[:, :, :, :, 9:12],
            'rf': y_label[:, :, :, :, :3],
            'uf': y_label[:, :, :, :, 6:9],
            'vf': y_label[:, :, :, :, 9:12]
        }
        micro_batch_size = y_label.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            y_hat, _ , _= self(x, y)
            self.save_vis_step_end(
                data_idx=data_idx,
                in_seq=x['t'],
                target_seq=y,
                pred_seq=y_hat,
                mode="val"
            )
            if self.precision == 16:
                y_hat = y_hat.float()
            step_mse = self.valid_mse(y_hat, y)
            step_mae = self.valid_mae(y_hat, y)
            self.valid_score.update(y_hat, y)
            self.log('valid_frame_mse_step', step_mse,
                     prog_bar=True, on_step=True, on_epoch=False)
            self.log('valid_frame_mae_step', step_mae,
                     prog_bar=True, on_step=True, on_epoch=False)
        return None

    def validation_epoch_end(self, outputs):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        self.log('valid_frame_mse_epoch', valid_mse,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_frame_mae_epoch', valid_mae,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
        valid_score = self.valid_score.compute()
        self.log("valid_loss_epoch", valid_mse,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log_score_epoch_end(score_dict=valid_score, mode="val")
        self.valid_score.reset()
        self.save_score_epoch_end(score_dict=valid_score,
                                  mse=valid_mse,
                                  mae=valid_mae,
                                  mode="val")

    def animate_time_series(self, features, batch_idx, save_dir='visualizations'):
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})  # 使用 Cartopy 投影

        # 绘制世界地图作为背景
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # 设置地图显示范围
        ax.coastlines()  # 添加海岸线
        ax.gridlines(draw_labels=True)  # 添加网格线

        # 对特征进行归一化处理
        features_min = features.min()
        features_max = features.max()
        features = (features - features_min) / (features_max - features_min + 1e-6)  # 防止除零

        # 设置特征图的地理范围 (extent)，确保填满整个地图
        # 假设特征图 shape 为 (batch, T, H, W, C)，对应全球
        extent = [-180, 180, -90, 90]

        # Loop over each time step
        for t in range(features.shape[1]):  # T dimension
            # 清空当前轴以避免叠加
            ax.clear()

            # 绘制地图和特征图
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(draw_labels=True)

            # 使用 GnBu 颜色映射，并指定地理范围
            im = ax.imshow(features[0, t, 0].detach().cpu().numpy(),
                           cmap='GnBu', interpolation='bilinear',
                           extent=extent, transform=ccrs.PlateCarree())

            # 保存每一帧为图片
            time_step_save_path = os.path.join(save_dir, f'batch{batch_idx}_timestep{t}.png')
            plt.savefig(time_step_save_path, dpi=300)  # 保存当前时间步的图片
            print(f"Saved timestep {t} as {time_step_save_path}")

        plt.close(fig)

    def animate_difference(self, y_hat, y, batch_idx, save_dir='visualizations'):

        # 创建目录
        os.makedirs(save_dir, exist_ok=True)

        # 计算差异图：y_hat - y
        difference = y_hat - y

        # 设置固定的颜色范围以保持每一帧的对比一致性
        diff_min, diff_max = -0.2, 0.2  # 设定颜色条范围，增大颜色差距

        # 设置地理范围
        extent = [-180, 180, -90, 90]

        # 创建固定布局和颜色条
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})  # 使用 Cartopy 投影
        ax.set_extent(extent, crs=ccrs.PlateCarree())  # 设置地图显示范围
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        # 初始化绘图
        im = ax.imshow(np.zeros((y.shape[-2], y.shape[-1])),  # 初始化空数据
                       cmap='RdBu_r', interpolation='bilinear',  # 红白渐变
                       extent=extent, transform=ccrs.PlateCarree(), vmin=diff_min, vmax=diff_max)

        # 创建颜色条，右侧固定并右移
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.1, shrink=0.8, aspect=20)
        cbar.set_label('Error')  # 设置颜色条标签

        # 循环生成每个时间步的图片
        for t in range(difference.shape[1]):  # T dimension
            # 更新绘图数据
            im.set_array(difference[0, t, 0].detach().cpu().numpy())  # 更新差异数据

            # 保存每一帧为图片
            time_step_save_path = os.path.join(save_dir, f'batch{batch_idx}_timestep{t}_difference.png')
            plt.savefig(time_step_save_path, dpi=300, bbox_inches='tight')  # bbox_inches 保证布局一致
            print(f"Saved difference timestep {t} as {time_step_save_path}")

        # 关闭图像
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        a, b = batch
        x_data = rearrange(a, 'B T C H W ->B T H W C')
        y_label = rearrange(b, 'B T C H W ->B T H W C')
        y = y_label[:, :, :, :, 3:6]
        x = {
            'r': x_data[:, :, :, :, :3],
            't': x_data[:, :, :, :, 3:6],
            'u': x_data[:, :, :, :, 6:9],
            'v': x_data[:, :, :, :, 9:12],
            'rf': y_label[:, :, :, :, :3],
            'uf': y_label[:, :, :, :, 6:9],
            'vf': y_label[:, :, :, :, 9:12]
        }
        # x['valid_flags'] = batch['valid_flags']
        micro_batch_size = y_label.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            y_hat, _, feature  = self(x, y)
            self.save_vis_step_end(
                data_idx=data_idx,
                in_seq=x['t'],
                target_seq=y,
                pred_seq=y_hat,
                mode="test"
            )
            x_mid = x['t'].permute(0, 1, 4, 2, 3).mean(dim=2).unsqueeze(2)
            y_mid = y.permute(0, 1, 4, 2, 3).mean(dim=2).unsqueeze(2)
            y_hat_mid = y_hat.permute(0, 1, 4, 2, 3).mean(dim=2).unsqueeze(2)
            self.animate_time_series(x_mid, batch_idx, save_dir='visualizations/features_x_ground')
            self.animate_time_series(y_mid, batch_idx, save_dir='visualizations/features_y_ground')
            self.animate_time_series(y_hat_mid, batch_idx, save_dir='visualizations/features_y_prediction')
            self.animate_difference(y_hat_mid, y_mid, batch_idx, save_dir='visualizations/difference')

            if self.precision == 16:
                y_hat = y_hat.float()
            # step_mse = self.test_mse(y_hat, y)
            # step_mae = self.test_mae(y_hat, y)
            outputs = {'inputs': x['t'].cpu().numpy(), 'preds': y_hat.cpu().numpy(), 'trues': y.cpu().numpy()}
            self.test_outputs.append(outputs)
            self.test_score.update(y_hat, y)
            # self.log('test_frame_mse_step', step_mse,
            #        prog_bar=True, on_step=True, on_epoch=False)
            # self.log('test_frame_mae_step', step_mae,
            #         prog_bar=True, on_step=True, on_epoch=False)
        return None

    def cal_mse(self,pred, true):
        # B T H W C
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum()

    def cal_mae(self,pred, true):
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred - true) / norm, axis=(0, 1)).sum()

    def cal_rmse(self, pred, true):
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum())

    def test_epoch_end(self, outputs):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        std = rearrange(self.dm.test_std[:,3:6,:,:],'T C H W -> T H W C')
        mean = rearrange(self.dm.test_mean[:, 3:6, :, :], 'T C H W -> T H W C')
        pred = results_all['preds'] * std + mean
        true = results_all['trues'] * std + mean
        test_mse = self.cal_mse(pred,true)
        test_mae = self.cal_mae(pred, true)
        test_rmse = self.cal_rmse(pred,true)
        self.log('test_frame_mse_epoch', test_mse,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_frame_mae_epoch', test_mae,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_frame_rmse_epoch', test_rmse,
                 prog_bar=True, on_step=False, on_epoch=True)
        test_score = self.test_score.compute()
        self.log_score_epoch_end(score_dict=test_score, mode="test")
        self.test_score.reset()
        self.save_score_epoch_end(score_dict=test_score,
                                  mse=test_mse,
                                  mae=test_mae,
                                  mode="test")
    r"""
    def test_epoch_end(self, outputs):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        self.log('test_frame_mse_epoch', test_mse,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_frame_mae_epoch', test_mae,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
        test_score = self.test_score.compute()
        self.log_score_epoch_end(score_dict=test_score, mode="test")
        self.test_score.reset()
        self.save_score_epoch_end(score_dict=test_score,
                                  mse=test_mse,
                                  mae=test_mae,
                                  mode="test")
    """
    def log_score_epoch_end(self, score_dict: Dict, mode: str = "val"):
        if mode == "val":
            log_mode_prefix = "valid"
        elif mode == "test":
            log_mode_prefix = "test"
        else:
            raise ValueError(f"Wrong mode {mode}. Must be 'val' or 'test'.")
        for metrics in self.metrics_list:
            for thresh in self.threshold_list:
                score_mean = np.mean(score_dict[thresh][metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_{thresh}_epoch", score_mean)
            score_avg_mean = score_dict.get("avg", None)
            if score_avg_mean is not None:
                score_avg_mean = np.mean(score_avg_mean[metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_avg_epoch", score_avg_mean)

    def save_score_epoch_end(self,
                             score_dict: Dict,
                             mse: Union[np.ndarray, float],
                             mae: Union[np.ndarray, float],
                             mode: str = "val"):
        assert mode in ["val", "test"], f"Wrong mode {mode}. Must be 'val' or 'test'."
        if self.local_rank == 0:
            save_dict = deepcopy(score_dict)
            save_dict.update(dict(mse=mse, mae=mae))
            if self.scores_dir is not None:
                save_path = os.path.join(self.scores_dir, f"{mode}_results_epoch_{self.current_epoch}.pkl")
                f = open(save_path, 'wb')
                pickle.dump(save_dict, f)
                f.close()

    def save_vis_step_end(
            self,
            data_idx: int,
            in_seq: torch.Tensor,
            target_seq: torch.Tensor,
            pred_seq: torch.Tensor,
            mode: str = "train"):
        r"""
        Parameters
        ----------
        data_idx:   int
            data_idx == batch_idx * micro_batch_size
        """
        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if data_idx in example_data_idx_list:
                save_example_vis_results(
                    save_dir=self.example_save_dir,
                    save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                    in_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    layout=self.layout,
                    plot_stride=self.oc.vis.plot_stride,
                    label=self.oc.logging.logging_prefix,
                    interval_real_time=self.oc.dataset.interval_real_time)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_sevir', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained checkpoints for test.')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on SEVIR.')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.pretrained:
        args.cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "earthformer_sevir_v1.yaml"))
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_oc = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        dataset_oc = OmegaConf.to_object(CuboidSEVIRPLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0
    seed_everything(seed, workers=True)
    dm = CuboidSEVIRPLModule.get_weather_datamodule(
        dataset_oc=dataset_oc,
        micro_batch_size=micro_batch_size,
        num_workers=8, )
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = CuboidSEVIRPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=int(micro_batch_size * args.gpus),
    )
    print('total_num_steps: ', total_num_steps)
    pl_module = CuboidSEVIRPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg,
        dm=dm)
    pre_path = '/mnt/disk1/mtr/pre.pt'
    r"""
    if pre_path.endswith('.pt'):
        m = torch.load(pre_path)
        pl_params = pl_module.torch_nn_module.state_dict()
        pl_params.clear()
        for k, v in dict(m['state_dict']).items():
            if not str(k).startswith('torch_nn_module.'):
                continue
            pl_params[str(k)[len('torch_nn_module.'):]] = v
        pl_module.torch_nn_module.load_state_dict(pl_params, strict=False)
"""

    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # 计算flops
    x_flops = {'r': torch.randn(16, 4, 32, 64, 3),
               't': torch.randn(16, 4, 32, 64, 3),
               'u': torch.randn(16, 4, 32, 64, 3),
               'v': torch.randn(16, 4, 32, 64, 3),
               'rf': torch.randn(16, 4, 32, 64, 3),
               'uf': torch.randn(16, 4, 32, 64, 3),
               'vf': torch.randn(16, 4, 32, 64, 3),
               }
    y_flops = torch.randn(16, 4, 32, 64, 3)
    x = (x_flops, y_flops)
    flops, params = profile(pl_module, inputs=x)
    print('flops: %.2f G' % (flops / 1000000000.0))

    trainer = Trainer(**trainer_kwargs)
    if args.pretrained:
        pretrained_ckpt_name = pytorch_state_dict_name
        if not os.path.exists(os.path.join(pretrained_checkpoints_dir, pretrained_ckpt_name)):
            s3_download_pretrained_ckpt(ckpt_name=pretrained_ckpt_name,
                                        save_dir=pretrained_checkpoints_dir,
                                        exist_ok=False)
        state_dict = torch.load(os.path.join(pretrained_checkpoints_dir, pretrained_ckpt_name),
                                map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict)
        trainer.test(model=pl_module,
                     datamodule=dm)
    elif args.test:
        assert args.ckpt_name is not None, f"args.ckpt_name is required for test!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module,
                     datamodule=dm,
                     ckpt_path=ckpt_path)
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
        with torch.autograd.detect_anomaly():
            trainer.fit(model=pl_module,
                        datamodule=dm,
                        ckpt_path=ckpt_path)

        state_dict = pl_ckpt_to_pytorch_state_dict(checkpoint_path=trainer.checkpoint_callback.best_model_path,
                                                   map_location=torch.device("cpu"),
                                                   delete_prefix_len=len("torch_nn_module."))
        torch.save(state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        trainer.test(ckpt_path="best",
                     datamodule=dm)


if __name__ == "__main__":
    main()
