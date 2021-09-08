# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.nlp.data.machine_translation.preproc_mt_data import MTDataPreproc
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from nemo.utils.config_utils import update_model_config
from nemo.utils.exp_manager import ExpManagerConfig, exp_manager


@dataclass
class MTEncDecConfig(NemoConfig):
    name: Optional[str] = 'MTEncDec'
    do_training: bool = True
    do_testing: bool = False
    model: MTEncDecModelConfig = MTEncDecModelConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='MTEncDec', files_to_copy=[])


@hydra_runner(config_path="conf", config_name="aayn_base")
def main(cfg: MTEncDecConfig) -> None:
    # merge default config with user specified config
    default_cfg = MTEncDecConfig()
    cfg = update_model_config(default_cfg, cfg)
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')

    # training is managed by PyTorch Lightning
    trainer_cfg = OmegaConf.to_container(cfg.trainer)
    trainer_cfg.pop('plugins', None)
    trainer = Trainer(plugins=[NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes)], **trainer_cfg)

    # tokenizers will be trained and and tarred training data will be created if needed
    # model config is then updated
    if cfg.model.preproc_out_dir is not None:
        MTDataPreproc(cfg=cfg.model, trainer=trainer)

    # experiment logs, checkpoints, and auto-resume are managed by exp_manager and PyTorch Lightning
    exp_manager(trainer, cfg.exp_manager)

    # everything needed to train translation models is encapsulated in the NeMo MTEncdDecModel
    print('building model...')
    mt_model = MTEncDecModel(cfg.model, trainer=trainer)
    print('finish build...')
    sys.exit()

    logging.info("\n\n************** Model parameters and their sizes ***********")
    for name, param in mt_model.named_parameters():
        print(name, param.size())
    logging.info("***********************************************************\n\n")

    if cfg.do_training:
        trainer.fit(mt_model)

    if cfg.do_testing:
        trainer.test(mt_model)


if __name__ == '__main__':
    main()
