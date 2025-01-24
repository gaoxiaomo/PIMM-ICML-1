# parser.py

# Copyright (c) CAIRI AI Lab. All rights reserved

def get_config():
    config = {
        # Set-up parameters
        'device': 'cuda',
        'dist': False,
        'res_dir': 'work_dirs',
        'ex_name': 'Debug',
        'fp16': False,
        'torchscript': False,
        'seed': 42,
        'fps': False,
        'test': False,
        'deterministic': False,
        # dataset parameters
        'batch_size': 16,
        'val_batch_size': 16,
        'num_workers': 4,
        'data_root': '/mnt/disk1/datasets',
        'dataname': 'weather_t2m_5_625',
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'use_augment': False,
        'use_prefetcher': False,
        'drop_last': False,
        # method parameters
        'method': 'SimVP',
        'config_file': 'configs/weather/t2m_5_625/SimVP_gSTA.py',
        'model_type': 'gSTA',
        'drop': 0.0,
        'drop_path': 0.0,
        'overwrite': False,
        # Training parameters (optimizer)
        'epoch': 200,
        'log_step': 1,
        'opt': 'adam',
        'opt_eps': None,
        'opt_betas': None,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'clip_grad': None,
        'clip_mode': 'norm',
        'no_display_method_info': False,
        # Training parameters (scheduler)
        'sched': 'onecycle',
        'lr': 1e-3,
        'lr_k_decay': 1.0,
        'warmup_lr': 1e-5,
        'min_lr': 1e-6,
        'final_div_factor': 1e4,
        'warmup_epoch': 0,
        'decay_epoch': 100,
        'decay_rate': 0.1,
        'filter_bias_and_bn': False,
        # Lightning parameters
        'gpus': [0],
        'metric_for_bestckpt': 'val_loss',
        'ckpt_path': None,
    }
    return config

# Example of how to use the config
if __name__ == "__main__":
    config = get_config()
    print(config)
    # You can now use the `config` dictionary directly in your training code