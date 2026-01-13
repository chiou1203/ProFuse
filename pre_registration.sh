#!/usr/bin/env python
import os
import sys
import torch
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE_ROOT = os.path.join(REPO_ROOT, "pre_registration")


sys.path.insert(0, PRE_ROOT)

from source.utils_aux import set_seed
from source.trainer import EDGSTrainer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", default=os.environ.get("SCENE_DIR"))
    
    p.add_argument("--out_dir", default=None, help="Output dir (default: <scene_dir>/out_pre_registration)")
    return p.parse_args()

def main():
    args = parse_args()
    scene_dir = os.path.abspath(args.scene_dir)
    if not args.scene_dir:
        p.error("Provide --scene_dir or set SCENE_DIR environment variable")
    model_out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(scene_dir, "out_pre_registration")

   


    with initialize_config_dir(version_base="1.1", config_dir=os.path.join(PRE_ROOT, "configs")):
        cfg = compose(config_name="train")

    
    num_ref_views = 16  # currently not used downstream

    cfg.gs.opt.batch_size = 64
    cfg.gs.dataset.model_path = model_out_dir
    cfg.gs.dataset.source_path = scene_dir
    cfg.gs.dataset.images = "images"

    cfg.gs.opt.TEST_CAM_IDX_TO_LOG = 12
    cfg.train.gs_epochs = 30000
    cfg.gs.opt.opacity_reset_interval = 1_000_000
    cfg.train.no_densify = True

    cfg.init_wC.matches_per_ref = 15_000
    cfg.init_wC.nns_per_ref = 3
    cfg.init_wC.num_refs = 180
    cfg.init_wC.roma_model = "indoors"

    cfg.init_wC.langfeat_dir = os.path.join(scene_dir, "language_features")
    cfg.init_wC.langfeat_level = 1

    cfg.init_wC.cluster_tau_iou = 0.2
    cfg.init_wC.cluster_tau_bbox = 0.08

    OmegaConf.resolve(cfg)
    set_seed(cfg.seed)

    print("Output folder:", cfg.gs.dataset.model_path)
    os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)

    gs = hydra.utils.instantiate(cfg.gs)
    trainer = EDGSTrainer(
        GS=gs,
        training_config=cfg.gs.opt,
        device=cfg.device,
        log_wandb=False,
    )

    # --- Correspondence-based init ---
    trainer.timer.start()
    trainer.init_with_corr(cfg.init_wC)
    trainer.timer.pause()

    trainer.saving_iterations = []
    with torch.no_grad():
        trainer.save_model()

    # --- First training chunk ---
    cfg.train.gs_epochs = 7000
    trainer.train(cfg.train)
    with torch.no_grad():
        trainer.save_model()

    # --- Second training chunk ---
    cfg.train.gs_epochs = 23_000
    trainer.train(cfg.train)
    with torch.no_grad():
        trainer.save_model()


if __name__ == "__main__":
    main()
