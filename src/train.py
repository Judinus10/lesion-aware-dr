import os
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from utils.logger import get_logger
from utils.seed import set_seed

def load_config(cfg_path: str = "configs/base.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return OmegaConf.create(cfg)

def main():
    # 1) Load config
    cfg = load_config()
    
    # 2) Create output directories
    Path(cfg.paths.outputs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # 3) Set seed
    set_seed(cfg.training.seed)

    # 4) Setup logger
    logger = get_logger("train")
    logger.info("Starting training script...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # 5) Optional: Initialize WandB
    if cfg.logging.use_wandb:
        import wandb
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info("WandB initialized.")

    # TODO:
    # 6) Here you will:
    #   - initialize dataloaders
    #   - create model
    #   - define loss, optimizer, scheduler
    #   - run training loop
    logger.info("Placeholder: training loop will go here.")

    if cfg.logging.use_wandb:
        wandb.finish()
        logger.info("WandB run finished.")

if __name__ == "__main__":
    main()
