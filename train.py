#!/usr/bin/env python3

import argparse
import torch
import sys
import os.path as osp

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# Import trainers
import trainers.pcpo

# Import datasets - we'll do this dynamically to avoid import issues
def import_datasets():
    """Dynamically import all dataset modules"""
    try:
        import datasets.office31
        import datasets.office_home
        import datasets.domainnet
        import datasets.mini_domainnet
        import datasets.visda17
    except ImportError as e:
        print(f"Warning: Failed to import some datasets: {e}")
        print("This might be okay if you're not using those specific datasets.")


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.source_model_path:
        cfg.TRAINER.PCPO.SOURCE_MODEL_PATH = args.source_model_path

    if args.target_data_dir:
        cfg.DATASET.TARGET_DATA_DIR = args.target_data_dir


def extend_cfg(cfg):
    """
    Add new config variables for PCPO.
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.PCPO = CN()
    cfg.TRAINER.PCPO.ARCH = "ViT-B/16"  # CLIP architecture
    cfg.TRAINER.PCPO.SOURCE_ARCH = "resnet50"  # Source model architecture
    cfg.TRAINER.PCPO.SOURCE_MODEL_PATH = ""  # Path to pre-trained source model
    cfg.TRAINER.PCPO.NUM_CLASSES = 31  # Number of classes
    cfg.TRAINER.PCPO.LEARNING_RATE = 0.001  # Learning rate for prototype optimization
    cfg.TRAINER.PCPO.WEIGHT_DECAY = 5e-4  # Weight decay
    cfg.TRAINER.PCPO.MOMENTUM = 0.9
    cfg.TRAINER.PCPO.PROTOTYPE_EPOCHS = 200  # Number of epochs for prototype optimization
    
    # Dataset-specific configurations
    cfg.TRAINER.PCPO.DATASET_CONFIGS = CN()
    cfg.TRAINER.PCPO.DATASET_CONFIGS.Office31 = CN()
    cfg.TRAINER.PCPO.DATASET_CONFIGS.Office31.NUM_CLASSES = 31
    cfg.TRAINER.PCPO.DATASET_CONFIGS.Office31.SOURCE_ARCH = "resnet50"
    cfg.TRAINER.PCPO.DATASET_CONFIGS.Office31.DOMAINS = ["amazon", "dslr", "webcam"]
    
    cfg.TRAINER.PCPO.DATASET_CONFIGS.OfficeHome = CN()
    cfg.TRAINER.PCPO.DATASET_CONFIGS.OfficeHome.NUM_CLASSES = 65
    cfg.TRAINER.PCPO.DATASET_CONFIGS.OfficeHome.SOURCE_ARCH = "resnet50"
    cfg.TRAINER.PCPO.DATASET_CONFIGS.OfficeHome.DOMAINS = ["art", "clipart", "product", "real_world"]
    
    cfg.TRAINER.PCPO.DATASET_CONFIGS.VisDA17 = CN()
    cfg.TRAINER.PCPO.DATASET_CONFIGS.VisDA17.NUM_CLASSES = 12
    cfg.TRAINER.PCPO.DATASET_CONFIGS.VisDA17.SOURCE_ARCH = "resnet101"
    cfg.TRAINER.PCPO.DATASET_CONFIGS.VisDA17.DOMAINS = ["train", "test"]
    
    cfg.TRAINER.PCPO.DATASET_CONFIGS.miniDomainNet = CN()
    cfg.TRAINER.PCPO.DATASET_CONFIGS.miniDomainNet.NUM_CLASSES = 126
    cfg.TRAINER.PCPO.DATASET_CONFIGS.miniDomainNet.SOURCE_ARCH = "resnet50"
    cfg.TRAINER.PCPO.DATASET_CONFIGS.miniDomainNet.DOMAINS = ["clipart", "painting", "real", "sketch"]

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.TARGET_DATA_DIR = ""  # Target domain data directory
    cfg.VERBOSE = False  # Reduce verbose output


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # 5. Normalize dataset name (fix case sensitivity issues)
    dataset_name_mapping = {
        "office31": "Office31",
        "Office31": "Office31",
        "office_home": "OfficeHome",
        "OfficeHome": "OfficeHome",
        "visda17": "VisDA17", 
        "VisDA17": "VisDA17",
        "minidomainnet": "miniDomainNet",
        "miniDomainNet": "miniDomainNet"
    }
    
    if cfg.DATASET.NAME in dataset_name_mapping:
        original_name = cfg.DATASET.NAME
        normalized_name = dataset_name_mapping[cfg.DATASET.NAME]
        if original_name != normalized_name:
            print(f"Warning: Dataset name normalized from '{original_name}' to '{normalized_name}'")
            cfg.defrost()
            cfg.DATASET.NAME = normalized_name

    cfg.freeze()

    return cfg


def main(args):
    # Import datasets dynamically
    import_datasets()
    
    cfg = setup_cfg(args)
    print(cfg.DATASET.NAME)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    if cfg.VERBOSE:
        print_args(args, cfg)

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="*", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="*", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--source-model-path", type=str, default="", 
                       help="path to pre-trained source model")
    parser.add_argument("--target-data-dir", type=str, default="", 
                       help="path to target domain data directory")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs="*",
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
