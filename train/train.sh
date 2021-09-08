#!/bin/bash

config_path="../conf"
config_name="nmt_en_zh"

python ./train/enc_dec_nmt.py \
    --config-path=${config_path} \
    --config-name=${config_name} \
    trainer.gpus=[0] \
    +trainer.max_steps=300000 \
    +exp_manager.create_wandb_logger=True \
    +exp_manager.wandb_logger_kwargs.name=TEST-nmt-base \
    +exp_manager.wandb_logger_kwargs.project=nmt-de-en \
    +exp_manager.create_checkpoint_callback=True \
    +exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
    +exp_manager.exp_dir=nmt_base \
    +exp_manager.checkpoint_callback_params.mode=max
