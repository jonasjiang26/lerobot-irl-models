import logging
import os
import sys
from pathlib import Path
from wsgiref.handlers import CGIHandler

import hydra
import torch
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging

from policies.beastf.beastf_config import BeastVLAConfig
from policies.beastf.modeling_beastf import BeastVLAPolicy

os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
log = logging.getLogger(__name__)

def _patch_lerobot_dataset_task_field() -> None:
    """
    TokenizerProcessorStep reads complementary_data['task'] and expects text.
    Some datasets store the real instruction in 'text.task' while 'task' is an integer id.

    Patch LeRobotDataset.__getitem__ so returned items use 'text.task' as 'task' when available.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:
        log.warning("Could not import LeRobotDataset to patch task field: %s", e)
        return

    if getattr(LeRobotDataset, "_beast_task_patch_applied", False):
        return

    orig_getitem = LeRobotDataset.__getitem__

    def _getitem_patched(self, idx):  # type: ignore[override]
        item = orig_getitem(self, idx)
        if "text.task" in item:
            item["task"] = item["text.task"]
        return item

    LeRobotDataset.__getitem__ = _getitem_patched  # type: ignore[assignment]
    LeRobotDataset._beast_task_patch_applied = True  # type: ignore[attr-defined]



@hydra.main(
    config_path="../configs", config_name="config", version_base="1.3"
)
def train(cfg):
    _patch_lerobot_dataset_task_field()

    dataset_cfg = DatasetConfig(
        repo_id=cfg.repo_id,
        root=cfg.dataset_path,
        video_backend="pyav",
    )
    pretrained_config = BeastVLAConfig(
        device=cfg.train.device,
        push_to_hub=cfg.train.push_to_hub,
        )
    
    train_cfg = TrainPipelineConfig(
        policy=pretrained_config,
        dataset=dataset_cfg,
        batch_size=cfg.train.batch_size,
        steps=cfg.train.steps,
        output_dir=Path(cfg.train.output_dir),
        job_name=cfg.train.job_name,
        save_freq=cfg.train.save_freq,
        seed=cfg.train.seed,
        log_freq=cfg.train.log_freq,
        num_workers=cfg.train.num_workers,
        wandb=WandBConfig(
        enable=cfg.wandb.enable,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        ),
	    )

    policy = BeastVLAPolicy(pretrained_config, task=cfg.task)

    train_cfg.pretrained_policy = policy

    init_logging()
    lerobot_train(train_cfg)

def get_beast(typename: str, **kwargs):
    return BeastVLAPolicy

if __name__ == "__main__":
    factory.get_policy_class = get_beast
    train()
