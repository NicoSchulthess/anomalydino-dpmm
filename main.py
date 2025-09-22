#########################################################################
# File adapted from AnomalyDINO (https://github.com/dammsi/AnomalyDINO) #
# Code from AnomalyDINO is used under the Apache License 2.0            #
#########################################################################


from argparse import ArgumentParser
import os
import torch
from torch.utils.data import DataLoader
import tracemalloc
import wandb

from src.backbones import get_model
from src.data import get_datasets
from src.train import train_dpmm_model
from src.test import test_dpmm_model
from src.utils import config_get, dump_config, load_config, makedirs, setup_device, setup_wandb


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="AnomalyDINO/default.yaml")
    parser.add_argument("--override_config", type=bool, default="False")
    args = parser.parse_args()
    return args


def main():
    tracemalloc.start()

    # command_line_args = OmegaConf.from_cli()
    
    cli_args = parse_args()
    cfg = load_config(cli_args.config)

    logdir = cfg.logdir
    if cfg.tag != None:
        logdir += "_" + cfg.tag
    
    cfg_export_path = os.path.join(logdir, "cfg.yaml")
    is_resumed = os.path.exists(cfg_export_path) and cfg.resume
    if is_resumed and not cli_args.override_config:
        cfg = load_config(cfg_export_path)
    else:
        os.makedirs(logdir, exist_ok=True)
        dump_config(cfg_export_path, cfg)

    print("Results will be saved to", logdir)

    benchmark = config_get(cfg, "benchmark", default=False)
    print(f"Running benchmark: {benchmark}")

    if cfg.wandb_log:
        wandb_dir = setup_wandb(logdir, is_resumed, cfg)

    device = setup_device(cfg.device)

    embedding_model = get_model(
        cfg.model_name,
        device,
        smaller_edge_size=cfg.resolution,
        num_pca_components=cfg.embedding_pca_components,
        normalize_embeddings=config_get(cfg, "normalize_embeddings", default=False),
    )

    for object_name in cfg.dataset.objects:
        experiment_dir = makedirs(logdir, object_name)
        checkpoint_best = os.path.join(experiment_dir, "checkpoint_best.pth")
        checkpoint_last = os.path.join(experiment_dir, "checkpoint_last.pth")

        dataset_train, dataset_val_normal, dataset_test_normal = get_datasets(
            cfg=cfg,
            object_name=object_name,
            anomaly_types=["good"],
            splits=["train", "val", "test"],
        )
        dataset_val_anomalous, dataset_test_anomalous = get_datasets(
            cfg=cfg,
            object_name=object_name,
            anomaly_types=cfg.dataset.object_anomalies[object_name],
            splits=["val", "test"],
        )

        dataloader_train = DataLoader(dataset_train, batch_size=cfg.embedding_batch_size, shuffle=True, drop_last=False, num_workers=0)
        dataloader_val_normal = DataLoader(dataset_val_normal, batch_size=cfg.embedding_batch_size, shuffle=False, drop_last=False, num_workers=0)
        dataloader_val_anomalous = DataLoader(dataset_val_anomalous, batch_size=cfg.embedding_batch_size, shuffle=False, drop_last=False, num_workers=0)
        dataloader_test_normal = DataLoader(dataset_test_normal, batch_size=cfg.embedding_batch_size, shuffle=False, drop_last=False, num_workers=0)
        dataloader_test_anomalous = DataLoader(dataset_test_anomalous, batch_size=cfg.embedding_batch_size, shuffle=False, drop_last=False, num_workers=0)

        if cfg.embedding_pca_components != -1:
            dataloader_pca = DataLoader(dataset_train, batch_size=cfg.embedding_pca_batchsize, shuffle=True, drop_last=False)
            images, _, _ = next(iter(dataloader_pca))
            embedding_model.fit_pca(images, cfg)


        model_name = cfg.density_model.model_name
        if model_name.startswith("dpmm"):
            if not benchmark:
                train_dpmm_model(
                    embedding_model,
                    is_resumed,
                    checkpoint_last,
                    checkpoint_best,
                    experiment_dir,
                    dataloader_train,
                    dataloader_val_normal,
                    dataloader_val_anomalous,
                    device,
                    cfg,
                )
            print("Testing DPMM")
            test_dpmm_model(
                embedding_model,
                checkpoint_best,
                experiment_dir,
                dataloader_val_normal,
                dataloader_test_normal,
                dataloader_test_anomalous,
                device,
                cfg,
                benchmark=benchmark,
            )
        else:
            raise NotImplementedError(f"unknown model name {model_name}")
        
    if cfg.wandb_log:
        wandb.finish()

    peak_cpu_memory = tracemalloc.get_traced_memory()[1]
    peak_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"Peak CPU Memory usage: {peak_cpu_memory / 1024 / 1024 / 1024} GB")
    print(f"Peak GPU Memory usage: {peak_gpu_memory / 1024 / 1024 / 1024} GB")

    print("Finished and evaluated all runs!")


if __name__=="__main__":
    main()
