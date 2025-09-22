#########################################################################
# File adapted from AnomalyDINO (https://github.com/dammsi/AnomalyDINO) #
# Code from AnomalyDINO is used under the Apache License 2.0            #
#########################################################################


import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import torch
from torchtnt.utils.loggers import CSVLogger
from tqdm import trange
import wandb

from src.DirichletProcessMixture.dpmm import DPMM
from src.evaluate import evaluate_dpmm
from src.positional_encoding import add_position_encoding
from src.utils import config_get, save_checkpoint, normalize, resample_features
from src.visualization import visualize_samples, visualize_likelihood_maps, visualize_likelihood_per_dimension, visualize_scores, visualize_embedding_maps


def train_dpmm_one_epoch(
    dataloader,
    embedding_model,
    dpmm: DPMM,
    device,
    cfg,
    epoch,
    plot_dir,
    initialize=False,
    pca=None,
):
    stats = {
        "loss": 0,
        "num_samples": 0,
        "num_steps": 0,
    }

    for image, label, image_paths in dataloader:
        with torch.no_grad():
            image, grid_size = embedding_model.crop_image(image)
            features = embedding_model.extract_features(image)

            target_res = config_get(cfg, "feature_resolution_train", default=-1)
            features, grid_size = resample_features(features, grid_size, target_res)

            object_mask = embedding_model.compute_background_mask(
                features, grid_size, threshold=cfg.pca_threshold, masking=cfg.object_mask)

        if config_get(cfg, "use_positional_encoding", default=False):
            features = add_position_encoding(
                features, grid_size[1], grid_size[0], features.shape[0], device)

        N = features.shape[0]
        object_features = features[object_mask]

        if initialize:
            dpmm.initialize(object_features)
            initialize = False

        if pca is None:
            pca = PCA(n_components=2)
            pca.fit(object_features.cpu().numpy())

            if config_get(cfg, "visualize_training_step", default=False):
                dpmm.visualize(object_features, pca=pca, normalize_pi_color=True, cov_color_curve=0, pi_cutoff=1e-10)
                outpath = os.path.join(plot_dir, f"train/epoch_0_step_0_init.jpg")
                plt.tight_layout()
                plt.savefig(outpath)
                plt.close()


        dpmm.step(object_features)
        if config_get(cfg, "visualize_training_step", default=False):
            dpmm.visualize(object_features, pca=pca, normalize_pi_color=True, cov_color_curve=0, pi_cutoff=1e-10)
            outpath = os.path.join(plot_dir, f"train/epoch_{epoch:03d}_step_{stats['num_steps']}.jpg")
            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()
        loss = -dpmm.score(object_features)

        with torch.no_grad():
            stats["loss"] += loss * N
            stats["num_samples"] += N
            stats["num_steps"] += 1

    if stats["num_samples"] > 0:
        stats["loss"] /= stats["num_samples"]
    return stats, pca


def train_dpmm_model(
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
):
    embedding_dim = embedding_model.get_embedding_dimension()
    
    if config_get(cfg, "use_positional_encoding", default=False):
        embedding_dim += 2

    dpmm = DPMM(
        K=cfg.density_model.dpmm.max_num_components,
        D=embedding_dim,
        update_rate=cfg.density_model.dpmm.update_rate,
        schedule=cfg.density_model.dpmm.schedule,
        cov_type=cfg.density_model.dpmm.gaussian.covariance_type,
        device=device,
        reg_covar=config_get(cfg, "density_model.dpmm.gaussian.covariance_regularization", 1e-6),
    )

    if is_resumed and os.path.exists(checkpoint_last):
        checkpoint = torch.load(checkpoint_last, map_location=device, weights_only=False)
        start_epoch = checkpoint["epoch"] + 1
        dpmm.load_state_dict(checkpoint["dpmm"])
        best_validation_loss = checkpoint["best_validation_loss"]
        if "embedding_model_pca" in checkpoint.keys():
            embedding_model.pca = checkpoint["embedding_model_pca"]
        if "visualization_pca" in checkpoint.keys():
            visualization_pca = checkpoint["visualization_pca"]
        else:
            visualization_pca = None

        print(f"Resuming training at epoch {start_epoch}.")
    else:
        start_epoch = 0
        best_validation_loss = torch.inf
        visualization_pca = None
        save_checkpoint(
            path=checkpoint_last,
            epoch=-1,
            dpmm=dpmm.state_dict(),
            best_validation_loss=best_validation_loss,
            embedding_model_pca=embedding_model.pca,
            visualization_pca=visualization_pca,
        )

        print("Starting training from scratch.")

    # if cfg.wandb_log:
    #     # wandb.save(os.path.join(wandb_dir, "checkpoint_last.pth"), base_path=wandb_dir)
    #     wandb.watch(density_model, loss_function, log='all')

    def eval(save_individual=False):
        with torch.inference_mode():
            _, (ax, ax_hist) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(8, 8))
            val_stats_normal = evaluate_dpmm(
                dataloader_val_normal,
                embedding_model,
                dpmm,
                device,
                cfg,
                ax=ax,
                ax_hist=ax_hist,
                pca=visualization_pca,
            )
            val_stats_anomalous = evaluate_dpmm(
                dataloader_val_anomalous,
                embedding_model,
                dpmm,
                device,
                cfg,
                ax=ax,
                ax_hist=ax_hist,
                pca=visualization_pca,
            )
            if visualization_pca is not None:
                dpmm.visualize(
                    data=None,
                    labels=None,
                    visualize_model=True,
                    pca=visualization_pca,
                    ax=ax,
                    ax_hist=ax_hist,
                    cov_color_curve=0,
                    pi_cutoff=1e-10,
                )
                plt.savefig(os.path.join(plot_dir, f"eval/epoch_{epoch}_step_{num_steps}.jpg"))
            plt.close()
            # TODO: No augmentation implemented
            # TODO: validation is only based on loss, not on F1 score,...

        val_stats_normal["squared_distance_map"], val_stats_anomalous["squared_distance_map"] = normalize(val_stats_normal["squared_distance_map"], val_stats_anomalous["squared_distance_map"])
        val_stats_normal["squared_weighted_distance_map"], val_stats_anomalous["squared_weighted_distance_map"] = normalize(val_stats_normal["squared_weighted_distance_map"], val_stats_anomalous["squared_weighted_distance_map"])
        val_stats_normal["distance_map"], val_stats_anomalous["distance_map"] = normalize(val_stats_normal["distance_map"], val_stats_anomalous["distance_map"])
        val_stats_normal["weighted_distance_map"], val_stats_anomalous["weighted_distance_map"] = normalize(val_stats_normal["weighted_distance_map"], val_stats_anomalous["weighted_distance_map"])

        val_stats_normal["cosine_distance_map"], val_stats_anomalous["cosine_distance_map"] = normalize(val_stats_normal["cosine_distance_map"], val_stats_anomalous["cosine_distance_map"])
        val_stats_normal["anomaly_map"], val_stats_anomalous["anomaly_map"] = normalize(val_stats_normal["anomaly_map"], val_stats_anomalous["anomaly_map"])

        val_stats_normal["max_log_prob_map"], val_stats_anomalous["max_log_prob_map"] = normalize(val_stats_normal["max_log_prob_map"], val_stats_anomalous["max_log_prob_map"])

        if cfg.density_model.dpmm.create_component_map_plot:
            visualize_likelihood_per_dimension(val_stats_normal, val_stats_anomalous, os.path.join(experiment_dir, f"epoch_{epoch}_step_{num_steps}_likelihood_per_dim.jpg"), cfg.resolution)
            visualize_likelihood_maps(val_stats_normal, os.path.join(experiment_dir, f"epoch_{epoch}_step_{num_steps}_likelihood_normal.jpg"))
            visualize_likelihood_maps(val_stats_anomalous, os.path.join(experiment_dir, f"epoch_{epoch}_step_{num_steps}_likelihood_anomalous.jpg"))

        visualize_samples(val_stats_anomalous, os.path.join(experiment_dir, f"epoch_{epoch}_step_{num_steps}_val_anomalous.jpg"), n_cols=10, cfg=cfg)
        visualize_samples(val_stats_normal, os.path.join(experiment_dir, f"epoch_{epoch}_step_{num_steps}_val_normal.jpg"), n_cols=10, cfg=cfg)

        visualize_samples(val_stats_anomalous, os.path.join(experiment_dir, f"epoch_{epoch}_step_{num_steps}_all_val_anomalous.jpg"), n_cols=-1, save_individual_split="val_anomalous" if save_individual else None, cfg=cfg)
        visualize_samples(val_stats_normal, os.path.join(experiment_dir, f"epoch_{epoch}_step_{num_steps}_all_val_normal.jpg"), n_cols=-1, save_individual_split="val_normal" if save_individual else None, cfg=cfg)
        visualize_scores(val_stats_normal, val_stats_anomalous, os.path.join(experiment_dir, f"epoch_{epoch}_step_{num_steps}_val_scores.jpg"))

        return val_stats_normal, val_stats_anomalous


    plot_dir = os.path.join(experiment_dir, "embeddings")
    os.makedirs(os.path.join(plot_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(plot_dir, "eval"), exist_ok=True)

    logger = CSVLogger(os.path.join(experiment_dir, "stats.csv"), steps_before_flushing=1)
    num_steps = 0
    for epoch in trange(start_epoch, cfg.epochs):
        train_stats, visualization_pca = train_dpmm_one_epoch(
            dataloader_train,
            embedding_model,
            dpmm,
            device,
            cfg,
            epoch,
            plot_dir,
            initialize=(epoch==0),
            pca=visualization_pca,
        )
        num_steps += train_stats["num_steps"]
        logger.log("training_loss", train_stats["loss"], num_steps)
        if cfg.wandb_log:
            wandb.log({
                'training_loss': train_stats["loss"],
            }, step=num_steps)

        val_stats_normal, val_stats_anomalous = eval()

        logger.log("validation_loss", val_stats_normal["loss"], num_steps)
        save_checkpoint(
            path=checkpoint_last,
            epoch=epoch,
            dpmm=dpmm.state_dict(),
            best_validation_loss=best_validation_loss,
            embedding_model_pca=embedding_model.pca,
            visualization_pca=visualization_pca,
        )

        if val_stats_normal["loss"] < best_validation_loss:
            best_validation_loss = val_stats_normal["loss"]
            save_checkpoint(
                path=checkpoint_best,
                epoch=epoch,
                dpmm=dpmm.state_dict(),
                best_validation_loss=best_validation_loss,
                embedding_model_pca=embedding_model.pca,
                visualization_pca=visualization_pca,
            )

        if cfg.wandb_log:
            wandb.log({
                'validation_loss_normal': val_stats_normal["loss"],
                'validation_loss_anomalous': val_stats_anomalous["loss"],
            }, step=num_steps)
            # wandb.save(os.path.join(wandb_dir, "checkpoint_last.pth"), base_path=wandb_dir)
            # wandb.save(os.path.join(wandb_dir, "checkpoint_best.pth"), base_path=wandb_dir)
    logger.flush()

    checkpoint = torch.load(checkpoint_best, map_location=device, weights_only=False)
    epoch = checkpoint["epoch"]
    dpmm.load_state_dict(checkpoint["dpmm"])
    best_validation_loss = checkpoint["best_validation_loss"]
    if "embedding_model_pca" in checkpoint.keys():
        embedding_model.pca = checkpoint["embedding_model_pca"]
    if "visualization_pca" in checkpoint.keys():
        visualization_pca = checkpoint["visualization_pca"]
    else:
        visualization_pca = None

    print(f"Validating best checkpoint from epoch {epoch}.")
    eval(save_individual=True)

    if config_get(cfg, "visualize_embedding_maps", default=False):
        visualize_embedding_maps(
            dataloader_val_anomalous,
            embedding_model,
            dpmm,
            visualization_pca,
            os.path.join(plot_dir, "embedding_per_sample"),
            cfg,
        )
