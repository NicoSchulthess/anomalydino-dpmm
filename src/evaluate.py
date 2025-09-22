#########################################################################
# File adapted from AnomalyDINO (https://github.com/dammsi/AnomalyDINO) #
# Code from AnomalyDINO is used under the Apache License 2.0            #
#########################################################################


import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.DirichletProcessMixture.dpmm import DPMM
from src.positional_encoding import add_position_encoding
from src.utils import config_get, resample_features


def evaluate_dpmm(
    dataloader,
    embedding_model,
    dpmm: DPMM,
    device,
    cfg,
    ax,
    ax_hist,
    pca,
    benchmark=False,
):
    stats = {
        "loss": 0,
        "num_samples": 0,
        "anomaly_map": [],
        "distance_map": [],
        "cosine_distance_map": [],
        "weighted_distance_map": [],
        "max_log_prob_map": [],
        "object_mask": [],
        "image": [],
        "labels": [],
        "weighted_log_prob_map": [],
        "component_log_prob_map": [],
        "component_distance_map": [],
        "component_cosine_distance_map": [],
    }

    weight_threshold = 1e-6

    for image, labels, image_paths in tqdm(dataloader):
        with torch.no_grad():
            image, grid_size = embedding_model.crop_image(image)
            features = embedding_model.extract_features(image)

            target_res = config_get(cfg, "feature_resolution_eval", default=-1)
            features, grid_size = resample_features(features, grid_size, target_res)

            object_mask = embedding_model.compute_background_mask(
                features, grid_size, threshold=cfg.pca_threshold, masking=cfg.object_mask)

            if config_get(cfg, "use_positional_encoding", default=False):
                features = add_position_encoding(
                    features, grid_size[1], grid_size[0], features.shape[0], device)

            N, _, _ = features.shape
            object_features = features[object_mask]

            batch_size = config_get(cfg, "evaluation_max_batch_size", default=object_features.shape[0])
            log_prob = []
            for batch in torch.split(object_features, batch_size, dim=0):
                log_prob.append(dpmm.sample_score(batch, weight_threshold=weight_threshold))
            log_prob = torch.cat(log_prob)
            loss = -log_prob.mean()

            labels_reduced = F.interpolate(labels, grid_size, mode="nearest")
            dpmm.visualize(
                data=object_features,
                labels=labels_reduced.reshape(-1),
                visualize_model=False,
                pca=pca,
                ax=ax,
                ax_hist=ax_hist,
                scatter_alpha=0.05,
                cov_color_curve=0,
                pi_cutoff=1e-10,
            )

            sample_cosine_distance = []
            for batch in torch.split(object_features, batch_size, dim=0):
                sample_cosine_distance.append(dpmm.cosine_distance_to_nearest_cluster(batch, weight_threshold=weight_threshold))
            sample_cosine_distance = torch.cat(sample_cosine_distance)
            cosine_distance_map = create_anomaly_map(sample_cosine_distance, object_mask, N, grid_size)
            stats["cosine_distance_map"] += torch.unbind(cosine_distance_map.cpu())
            stats["loss"] += loss * N
            stats["num_samples"] += N
            stats["image_paths"] += list(image_paths)

            if not benchmark:
                weighted_log_prob = []
                component_log_prob = []
                component_distance = []
                component_cosine_distance = []
                sample_distance = []
                sample_weighted_distance = []
                for batch in torch.split(object_features, batch_size, dim=0):
                    weighted_log_prob.append(dpmm.get_weighted_log_prob(batch[:, None, :], weight_threshold))
                    component_log_prob.append(dpmm.get_log_prob(batch[:, None, :], weight_threshold))
                    component_distance.append(dpmm.distance_to_nearest_cluster(batch, weight_threshold=weight_threshold, covariance_weighted_norm=True, return_min=False))
                    component_cosine_distance.append(dpmm.cosine_distance_to_nearest_cluster(batch, weight_threshold=weight_threshold, return_min=False))
                    sample_distance.append(dpmm.distance_to_nearest_cluster(batch, weight_threshold=weight_threshold, covariance_weighted_norm=False))
                    sample_weighted_distance.append(dpmm.distance_to_nearest_cluster(batch, weight_threshold=weight_threshold, covariance_weighted_norm=True))
                weighted_log_prob = torch.cat(weighted_log_prob)
                component_log_prob = torch.cat(component_log_prob)
                component_distance = torch.cat(component_distance)
                component_cosine_distance = torch.cat(component_cosine_distance)
                sample_distance = torch.cat(sample_distance)
                sample_weighted_distance = torch.cat(sample_weighted_distance)

                pi = dpmm.calculate_log_pi().exp()
                pi_mask = (pi > weight_threshold)
                weighted_log_prob = weighted_log_prob[:, pi_mask]
                component_log_prob = component_log_prob[:, pi_mask]
                pi = pi[pi_mask]
                ordering = pi.argsort(descending=True)
                pi = pi[ordering]
                weighted_log_prob = weighted_log_prob[:, ordering]
                component_log_prob = component_log_prob[:, ordering]

                component_distance = component_distance[:, ordering]
                component_cosine_distance = component_cosine_distance[:, ordering]

                weighted_log_prob_map = create_anomaly_map(-weighted_log_prob, object_mask, N, grid_size)
                component_log_prob_map = create_anomaly_map(-component_log_prob, object_mask, N, grid_size)
                component_distance_map = create_anomaly_map(component_distance, object_mask, N, grid_size)
                component_cosine_distance_map = create_anomaly_map(component_cosine_distance, object_mask, N, grid_size)

                stats["weighted_log_prob_map"] += torch.unbind(weighted_log_prob_map.cpu())
                stats["component_log_prob_map"] += torch.unbind(component_log_prob_map.cpu())
                stats["component_distance_map"] += torch.unbind(component_distance_map.cpu())
                stats["component_cosine_distance_map"] += torch.unbind(component_cosine_distance_map.cpu())

                anomaly_map = create_anomaly_map(-log_prob, object_mask, N, grid_size)
                distance_map = create_anomaly_map(sample_distance, object_mask, N, grid_size)
                weighted_distance_map = create_anomaly_map(sample_weighted_distance, object_mask, N, grid_size)
                max_log_prob_map = component_log_prob_map.amin(1, keepdim=True)  # min of map is equivalent to max of logprob as map is negative logprob
                object_mask = object_mask.reshape(N, 1, grid_size[0], grid_size[1])

                stats["anomaly_map"] += torch.unbind(anomaly_map.cpu())
                stats["distance_map"] += torch.unbind(distance_map.cpu())
                stats["weighted_distance_map"] += torch.unbind(weighted_distance_map.cpu())
                stats["max_log_prob_map"] += torch.unbind(max_log_prob_map.cpu())
                stats["object_mask"] += torch.unbind(object_mask.cpu())
                stats["image"] += torch.unbind(image.cpu())
                stats["labels"] += torch.unbind(labels.cpu())
        
    if stats["num_samples"] > 0:
        stats["loss"] /= stats["num_samples"]
    
    stats["cosine_distance_map"] = torch.stack(stats["cosine_distance_map"])
    if not benchmark:
        stats["anomaly_map"] = torch.stack(stats["anomaly_map"])
        stats["distance_map"] = torch.stack(stats["distance_map"])
        stats["weighted_distance_map"] = torch.stack(stats["weighted_distance_map"])
        stats["max_log_prob_map"] = torch.stack(stats["max_log_prob_map"])
        stats["object_mask"] = torch.stack(stats["object_mask"])
        stats["image"] = torch.stack(stats["image"])
        stats["labels"] = torch.stack(stats["labels"])

        stats["weighted_log_prob_map"] = torch.stack(stats["weighted_log_prob_map"])
        stats["component_log_prob_map"] = torch.stack(stats["component_log_prob_map"])
        stats["component_distance_map"] = torch.stack(stats["component_distance_map"])
        stats["component_cosine_distance_map"] = torch.stack(stats["component_cosine_distance_map"])

        stats["squared_distance_map"] = stats["distance_map"] ** 2
        stats["squared_weighted_distance_map"] = stats["weighted_distance_map"] ** 2

    return stats


def create_anomaly_map(score, object_mask, N, grid_size):
    if score.ndim == 1:
        C = 1
    elif score.ndim == 2:
        C = score.shape[1]
    else:
        raise ValueError(f"score should have 1 or 2 dimensions, but has shape {score.shape}")
    
    anomaly_map = torch.zeros((N, grid_size[0] * grid_size[1], C), device=score.device)
    anomaly_map[object_mask[:, :, None].repeat(1, 1, C)] = score.reshape(-1)
    anomaly_map = anomaly_map.permute(0, 2, 1)   # Nx(H*W)xC -> NxCx(H*W)
    anomaly_map = anomaly_map.reshape(N, C, grid_size[0], grid_size[1])  # NxCx(H*W) -> NxCxHxW
    return anomaly_map
