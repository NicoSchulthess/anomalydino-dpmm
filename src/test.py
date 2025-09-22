#########################################################################
# File adapted from AnomalyDINO (https://github.com/dammsi/AnomalyDINO) #
# Code from AnomalyDINO is used under the Apache License 2.0            #
#########################################################################


from anomalib.utils.metrics import AUROC, AUPRO, PRO, OptimalF1, AUPR
import matplotlib.pyplot as plt
import os
import time
import torch
from torch.nn import functional as F
from torchmetrics import Dice, Metric
from torchtnt.utils.loggers import CSVLogger

from src.DirichletProcessMixture.dpmm import DPMM
from src.evaluate import evaluate_dpmm
from src.utils import config_get, save_checkpoint, normalize
from src.visualization import visualize_samples, visualize_likelihood_maps, visualize_likelihood_per_dimension, visualize_test_score_histogram

def get_threshold(scores, keys, allowed_fpr=1e-2):
    thresholds = {}
    for key in keys:
        thresholds[key] = torch.quantile(scores[key], 1 - allowed_fpr)
    return thresholds


def calculate_metric(
    stats_normal: dict,
    stats_anomal: dict,
    map_key: str,
    metric: Metric,
    thresholds: dict = None,
    interpolation_mode: str = "bilinear",
):
    labels = torch.cat([stats_normal["labels"], stats_anomal["labels"]], dim=0)
    target_shape = labels.shape[-2:]

    pred = torch.cat([stats_normal[map_key], stats_anomal[map_key]], dim=0)
    pred = F.interpolate(pred, target_shape, mode=interpolation_mode)

    if thresholds is not None:
        pred = (pred > thresholds[map_key]).float()
        labels = labels.int()

    metric.update(pred[:,0,:,:], labels[:,0,:,:])
    value = metric.compute()
    if hasattr(metric, "generate_figure"):
        metric.generate_figure()
    return value


def test_dpmm_model(
    embedding_model,
    checkpoint_best,
    experiment_dir,
    dataloader_val_normal,
    dataloader_test_normal,
    dataloader_test_anomalous,
    device,
    cfg,
    benchmark=False,
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

    checkpoint = torch.load(checkpoint_best, map_location=device, weights_only=False)
    test_epoch = checkpoint["epoch"]
    print(f"Testing model from epoch {test_epoch}")
    dpmm.load_state_dict(checkpoint["dpmm"])
    if "embedding_model_pca" in checkpoint.keys():
        embedding_model.pca = checkpoint["embedding_model_pca"]
    if "visualization_pca" in checkpoint.keys():
        visualization_pca = checkpoint["visualization_pca"]
    else:
        visualization_pca = None

    plot_dir = os.path.join(experiment_dir, "embeddings")
    os.makedirs(os.path.join(plot_dir, "test"), exist_ok=True)
    
    with torch.inference_mode():
        if not benchmark:
            val_stats_normal_path = os.path.join(experiment_dir, "val_stats_normal.pth")
            if os.path.exists(val_stats_normal_path):
                val_stats_normal = torch.load(val_stats_normal_path, map_location="cpu", weights_only=False)
            else:
                val_stats_normal = evaluate_dpmm(
                    dataloader_val_normal,
                    embedding_model,
                    dpmm,
                    device,
                    cfg,
                    ax=None,
                    ax_hist=None,
                    pca=visualization_pca,
                )
                save_checkpoint(
                    path=val_stats_normal_path,
                    **val_stats_normal,
                )
        if benchmark:
            start_time = time.time()

        _, (ax, ax_hist) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(8, 8))
        test_stats_normal_path = os.path.join(experiment_dir, "test_stats_normal.pth")
        if os.path.exists(test_stats_normal_path):
            test_stats_normal = torch.load(test_stats_normal_path, map_location="cpu", weights_only=False)
        else:
            test_stats_normal = evaluate_dpmm(
                dataloader_test_normal,
                embedding_model,
                dpmm,
                device,
                cfg,
                ax=ax,
                ax_hist=ax_hist,
                pca=visualization_pca,
                benchmark=benchmark,
            )
            save_checkpoint(
                path=test_stats_normal_path,
                **test_stats_normal,
            )

        test_stats_anomalous_path = os.path.join(experiment_dir, "test_stats_anomalous.pth")
        if os.path.exists(test_stats_anomalous_path):
            test_stats_anomalous = torch.load(test_stats_anomalous_path, map_location="cpu", weights_only=False)
        else:
            test_stats_anomalous = evaluate_dpmm(
                dataloader_test_anomalous,
                embedding_model,
                dpmm,
                device,
                cfg,
                ax=ax,
                ax_hist=ax_hist,
                pca=visualization_pca,
                benchmark=benchmark,
            )
            save_checkpoint(
                path=os.path.join(experiment_dir, "test_stats_anomalous.pth"),
                **test_stats_anomalous,
            )
        if benchmark:
            torch.cuda.synchronize() # Synchronize CUDA kernels before measuring time
            inf_time = time.time() - start_time
            print(f"runtime: {inf_time} s")
            return
        
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
            plt.savefig(os.path.join(plot_dir, f"test/model_visualization.jpg"))
        plt.close()

    logger = CSVLogger(os.path.join(experiment_dir, "stats_test.csv"), steps_before_flushing=1)
    score_map_keys = [
        "anomaly_map",
        "distance_map",
        "squared_distance_map",
        "weighted_distance_map",
        "squared_weighted_distance_map",
        "cosine_distance_map",
    ]
    thresholds = {
        10: get_threshold(val_stats_normal, score_map_keys, allowed_fpr=0.1),
         5: get_threshold(val_stats_normal, score_map_keys, allowed_fpr=0.05),
         1: get_threshold(val_stats_normal, score_map_keys, allowed_fpr=0.01),
    }

    for map_key in score_map_keys:
        visualize_test_score_histogram(test_stats_normal, test_stats_anomalous, thresholds.values(), map_key, os.path.join(experiment_dir, f"test_histogram_{map_key}.eps"))

        auroc = calculate_metric(test_stats_normal, test_stats_anomalous, map_key, AUROC())
        logger.log(f"auroc", auroc, map_key)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, f"test_auroc_{map_key}.jpg"))
        plt.close()

        max_dice = calculate_metric(test_stats_normal, test_stats_anomalous, map_key, OptimalF1())
        logger.log(f"max_dice", max_dice, map_key)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, f"test_max_dice_{map_key}.jpg"))
        plt.close()

        aupr = calculate_metric(test_stats_normal, test_stats_anomalous, map_key, AUPR())
        logger.log(f"aupr", aupr, map_key)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, f"test_aupr_{map_key}.jpg"))
        plt.close()


        aupro = calculate_metric(test_stats_normal, test_stats_anomalous, map_key, AUPRO())
        logger.log(f"aupro", aupro, map_key)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, f"test_aupro_{map_key}.jpg"))
        plt.close()

        pro = calculate_metric(test_stats_normal, test_stats_anomalous, map_key, PRO(), thresholds)
        logger.log(f"pro", pro, map_key)

        for threshold_val, threshold_dict in thresholds.items():
            dice = calculate_metric(test_stats_normal, test_stats_anomalous, map_key, Dice(), threshold_dict)
            logger.log(f"dice_{threshold_val:02d}", dice, map_key)
        logger.flush()

    test_stats_normal["squared_distance_map"], test_stats_anomalous["squared_distance_map"] = normalize(test_stats_normal["squared_distance_map"], test_stats_anomalous["squared_distance_map"])
    test_stats_normal["squared_weighted_distance_map"], test_stats_anomalous["squared_weighted_distance_map"] = normalize(test_stats_normal["squared_weighted_distance_map"], test_stats_anomalous["squared_weighted_distance_map"])
    test_stats_normal["distance_map"], test_stats_anomalous["distance_map"] = normalize(test_stats_normal["distance_map"], test_stats_anomalous["distance_map"])
    test_stats_normal["weighted_distance_map"], test_stats_anomalous["weighted_distance_map"] = normalize(test_stats_normal["weighted_distance_map"], test_stats_anomalous["weighted_distance_map"])

    test_stats_normal["cosine_distance_map"], test_stats_anomalous["cosine_distance_map"] = normalize(test_stats_normal["cosine_distance_map"], test_stats_anomalous["cosine_distance_map"])
    test_stats_normal["anomaly_map"], test_stats_anomalous["anomaly_map"] = normalize(test_stats_normal["anomaly_map"], test_stats_anomalous["anomaly_map"])

    test_stats_normal["max_log_prob_map"], test_stats_anomalous["max_log_prob_map"] = normalize(test_stats_normal["max_log_prob_map"], test_stats_anomalous["max_log_prob_map"])

    if cfg.density_model.dpmm.create_component_map_plot:
        visualize_likelihood_per_dimension(test_stats_normal, test_stats_anomalous, os.path.join(experiment_dir, f"test_likelihood_per_dim.jpg"), cfg.resolution)
        visualize_likelihood_maps(test_stats_normal, os.path.join(experiment_dir, f"test_likelihood_normal.jpg"))
        visualize_likelihood_maps(test_stats_anomalous, os.path.join(experiment_dir, f"test_likelihood_anomalous.jpg"))

    visualize_samples(test_stats_anomalous, os.path.join(experiment_dir, f"test_samples_anomalous.jpg"), n_cols=10, cfg=cfg)
    visualize_samples(test_stats_normal, os.path.join(experiment_dir, f"test_samples_normal.jpg"), n_cols=10, cfg=cfg)

    visualize_samples(test_stats_anomalous, os.path.join(experiment_dir, f"test_samples_more_anomalous.jpg"), n_cols=100, cfg=cfg)
    visualize_samples(test_stats_normal, os.path.join(experiment_dir, f"test_samples_more_normal.jpg"), n_cols=100, cfg=cfg)
