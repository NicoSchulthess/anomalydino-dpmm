import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import torch
from torch.nn import functional as F

from src.utils import config_get, normalize


def visualize_samples(stat_dict: dict, save_path: str, n_cols: int = 10, save_individual_split: str = None, cfg = None):
    rotate_by = config_get(cfg, "rotate_visualized_samples", default=0)

    anomaly_map = stat_dict["anomaly_map"]
    image = stat_dict["image"]
    labels = stat_dict["labels"]
    object_mask = stat_dict["object_mask"]

    cosine_distance_map = stat_dict.get("cosine_distance_map", None)
    distance_map = stat_dict.get("distance_map", None)
    weighted_distance_map = stat_dict.get("weighted_distance_map", None)
    squared_distance_map = stat_dict.get("squared_distance_map", None)
    squared_weighted_distance_map = stat_dict.get("squared_weighted_distance_map", None)
    max_log_prob_map = stat_dict.get("max_log_prob_map", None)

    additional_plots = (cosine_distance_map is not None or distance_map is not None or weighted_distance_map is not None)

    if additional_plots:
        assert "component_distance_map" in stat_dict.keys()
        assert "component_cosine_distance_map" in stat_dict.keys()
        assert "component_log_prob_map" in stat_dict.keys()
        num_active_components = stat_dict["component_distance_map"].shape[1]
        component_distance_assignment = torch.argmin(stat_dict["component_distance_map"], axis=1)
        component_cosine_distance_assignment = torch.argmin(stat_dict["component_cosine_distance_map"], axis=1)
        component_log_prob_assignment = torch.argmin(stat_dict["component_log_prob_map"], axis=1)  # min because component_log_prob_map is negative log prob

    n_rows = 13 if additional_plots else 4

    N = image.shape[0]
    if n_cols > N or n_cols == -1:
        n_imgs = N
    else:
        n_imgs = n_cols

    _, ax = plt.subplots(
        n_rows,
        n_imgs,
        figsize=(2*n_imgs+2,2*n_rows),
        tight_layout=True,
        subplot_kw={"xticks": [], "yticks": []},
    )

    colormap = get_extended_colormap(num_shades=9)
    max_colors = len(colormap.colors)
    if component_log_prob_assignment.max() > max_colors or component_cosine_distance_assignment.max() > max_colors or component_distance_assignment.max() > max_colors:
        print(f"Warning: More than {max_colors} components used: {component_log_prob_assignment.max()}, {component_cosine_distance_assignment.max()}, {component_distance_assignment.max()}")

    for i_plt, i_img in enumerate(np.linspace(0, N, n_imgs, endpoint=False, dtype=int)):
        ax[0,i_plt].imshow(np.rot90(normalize(image[i_img].permute(1,2,0)), rotate_by))
        ax[1,i_plt].imshow(np.rot90(anomaly_map[i_img][0], rotate_by), vmin=0, vmax=1)
        if additional_plots:
            ax[2,i_plt].imshow(np.rot90(component_distance_assignment[i_img], rotate_by), vmin=0, vmax=num_active_components, cmap=colormap)
            ax[3,i_plt].imshow(np.rot90(component_cosine_distance_assignment[i_img], rotate_by), vmin=0, vmax=num_active_components, cmap=colormap)
            ax[4,i_plt].imshow(np.rot90(component_log_prob_assignment[i_img], rotate_by), vmin=0, vmax=num_active_components, cmap=colormap)
            ax[5,i_plt].imshow(np.rot90(cosine_distance_map[i_img][0], rotate_by), vmin=0, vmax=1)
            ax[6,i_plt].imshow(np.rot90(distance_map[i_img][0], rotate_by), vmin=0, vmax=1)
            ax[7,i_plt].imshow(np.rot90(squared_distance_map[i_img][0], rotate_by), vmin=0, vmax=1)
            ax[8,i_plt].imshow(np.rot90(weighted_distance_map[i_img][0], rotate_by), vmin=0, vmax=1)
            ax[9,i_plt].imshow(np.rot90(squared_weighted_distance_map[i_img][0], rotate_by), vmin=0, vmax=1)
            ax[10,i_plt].imshow(np.rot90(max_log_prob_map[i_img][0], rotate_by), vmin=0, vmax=1)
        ax[-2,i_plt].imshow(np.rot90(labels[i_img][0], rotate_by), vmin=0, vmax=1)
        ax[-1,i_plt].imshow(np.rot90(object_mask[i_img][0], rotate_by), vmin=0, vmax=1)

        if save_individual_split is not None:
            folder = os.path.join(os.path.split(save_path)[0], "individual_images", save_individual_split)
            os.makedirs(folder, exist_ok=True)
            plt.imsave(os.path.join(folder, f"image_{i_img:04}.png"),                            np.rot90(normalize(image[i_img].permute(1,2,0)).numpy(), rotate_by))
            plt.imsave(os.path.join(folder, f"full_likelihood_{i_img:04}.png"),                  np.rot90(anomaly_map[i_img][0], rotate_by), vmin=0, vmax=1)
            plt.imsave(os.path.join(folder, f"l2_assignment_{i_img:04}.png"),                    np.rot90(component_distance_assignment[i_img], rotate_by), vmin=0, vmax=num_active_components, cmap=colormap)
            plt.imsave(os.path.join(folder, f"cosine_assignment_{i_img:04}.png"),                np.rot90(component_cosine_distance_assignment[i_img], rotate_by), vmin=0, vmax=num_active_components, cmap=colormap)
            plt.imsave(os.path.join(folder, f"likelihood_assignment_{i_img:04}.png"),            np.rot90(component_log_prob_assignment[i_img], rotate_by), vmin=0, vmax=num_active_components, cmap=colormap)
            plt.imsave(os.path.join(folder, f"min_cosine_distance_{i_img:04}.png"),              np.rot90(cosine_distance_map[i_img][0], rotate_by), vmin=0, vmax=1)
            plt.imsave(os.path.join(folder, f"min_l2_distance_{i_img:04}.png"),                  np.rot90(distance_map[i_img][0], rotate_by), vmin=0, vmax=1)
            plt.imsave(os.path.join(folder, f"min_squared_l2_distance_{i_img:04}.png"),          np.rot90(squared_distance_map[i_img][0], rotate_by), vmin=0, vmax=1)
            plt.imsave(os.path.join(folder, f"min_mahalanobis_distance_{i_img:04}.png"),         np.rot90(weighted_distance_map[i_img][0], rotate_by), vmin=0, vmax=1)
            plt.imsave(os.path.join(folder, f"min_squared_mahalanobis_distance_{i_img:04}.png"), np.rot90(squared_weighted_distance_map[i_img][0], rotate_by), vmin=0, vmax=1)
            plt.imsave(os.path.join(folder, f"max_likelihood_{i_img:04}.png"),                   np.rot90(max_log_prob_map[i_img][0], rotate_by), vmin=0, vmax=1)
            plt.imsave(os.path.join(folder, f"ground_truth_{i_img:04}.png"),                     np.rot90(labels[i_img][0], rotate_by), vmin=0, vmax=1)
            plt.imsave(os.path.join(folder, f"object_mask_{i_img:04}.png"),                      np.rot90(object_mask[i_img][0], rotate_by), vmin=0, vmax=1)
            np.savez(
                os.path.join(folder, f"data_{i_img:04}.npz"),
                image                            = np.rot90(normalize(image[i_img].permute(1,2,0)).numpy(), rotate_by),
                full_likelihood                  = np.rot90(anomaly_map[i_img][0], rotate_by),
                l2_assignment                    = np.rot90(component_distance_assignment[i_img], rotate_by),
                cosine_assignment                = np.rot90(component_cosine_distance_assignment[i_img], rotate_by),
                likelihood_assignment            = np.rot90(component_log_prob_assignment[i_img], rotate_by),
                min_cosine_distance              = np.rot90(cosine_distance_map[i_img][0], rotate_by),
                min_l2_distance                  = np.rot90(distance_map[i_img][0], rotate_by),
                min_squared_l2_distance          = np.rot90(squared_distance_map[i_img][0], rotate_by),
                min_mahalanobis_distance         = np.rot90(weighted_distance_map[i_img][0], rotate_by),
                min_squared_mahalanobis_distance = np.rot90(squared_weighted_distance_map[i_img][0], rotate_by),
                max_likelihood                   = np.rot90(max_log_prob_map[i_img][0], rotate_by),
                ground_truth                     = np.rot90(labels[i_img][0], rotate_by),
                object_mask                      = np.rot90(object_mask[i_img][0], rotate_by),
            )

    ax[0,0].set_ylabel("Image", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
    ax[1,0].set_ylabel("Full Likelihood\n" + r"$\log\left(\sum_k \pi_k \mathcal{N}(y_n | \mu_k, \Sigma_k)\right)$", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
    if additional_plots:
        ax[2,0].set_ylabel("L2 Distance Assignment\n" + r"$\arg\min_k ||y_n - \mu_k||_2$", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
        ax[3,0].set_ylabel("Cosine Distance Assignment\n" + r"$\arg\min_k 1 - \frac{y_n^{\top} \cdot \mu_k}{||y_n||_2 \cdot ||\mu_k||_2}$", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
        ax[4,0].set_ylabel("Likelihood Assignment\n" + r"$\arg\max_k \log \mathcal{N}(y_n | \mu_k, \Sigma_k)$", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
        ax[5,0].set_ylabel("Min Cosine Distance\n" + r"$\min_k 1 - \frac{y_n^{\top} \cdot \mu_k}{||y_n||_2 \cdot ||\mu_k||_2}$", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
        ax[6,0].set_ylabel("Min L2 Distance\n" + r"$\min_k ||y_n - \mu_k||_2$", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
        ax[7,0].set_ylabel("Min Squared L2 Distance\n" + r"$\min_k ||y_n - \mu_k||^2_2$", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
        ax[8,0].set_ylabel("Min Mahalanobis Distance\n" + r"$\min_k \sqrt{(y_n - \mu_k)^T \Sigma_k^{-1} (y_n - \mu_k)}$", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
        ax[9,0].set_ylabel("Min Squared Mahalanobis Distance\n" + r"$\min_k (y_n - \mu_k)^T \Sigma_k^{-1} (y_n - \mu_k)$", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
        ax[10,0].set_ylabel("Max Component Likelihood\n" + r"$\max_k \log \mathcal{N}(y_n | \mu_k, \Sigma_k)$", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
    ax[-2,0].set_ylabel("Ground Truth", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
    ax[-1,0].set_ylabel("Object Mask", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
    plt.savefig(save_path)
    plt.close()


def visualize_likelihood_maps(stat_dict: dict, save_path: str, n_cols: int = 10, n_max_components: int = -1):
    anomaly_map = stat_dict["anomaly_map"]
    weighted_log_prob_map = stat_dict["weighted_log_prob_map"]
    component_distance_map = stat_dict["component_distance_map"]
    component_cosine_distance_map = stat_dict["component_cosine_distance_map"]
    image = stat_dict["image"]
    labels = stat_dict["labels"]

    weighted_log_prob_map = normalize(weighted_log_prob_map, dim=(0,2,3))
    component_distance_map = normalize(component_distance_map, dim=(0,2,3))
    component_cosine_distance_map = normalize(component_cosine_distance_map, dim=(0,2,3))

    n_components = weighted_log_prob_map.shape[1] if n_max_components == -1 else min(n_components, weighted_log_prob_map.shape[1])
    n_rows = 1 + n_components

    N = image.shape[0]
    if n_cols > N:
        n_imgs = N
    else:
        n_imgs = n_cols

    _, ax = plt.subplots(
        n_rows,
        3*n_imgs,
        figsize=(6*n_imgs,2*n_rows),
        tight_layout=True,
        subplot_kw={"xticks": [], "yticks": []},
    )

    for i_plt, i_img in enumerate(np.linspace(0, N, n_imgs, endpoint=False, dtype=int)):
        ax[0,3*i_plt].imshow(normalize(image[i_img].permute(1,2,0)))
        ax[0,3*i_plt].set_ylabel("Image")
        ax[0,3*i_plt + 1].imshow(labels[i_img][0], vmin=0, vmax=1)
        ax[0,3*i_plt + 1].set_ylabel("Ground Truth")
        ax[0,3*i_plt + 2].imshow(anomaly_map[i_img][0], vmin=0, vmax=1)
        ax[0,3*i_plt + 2].set_ylabel("Anomaly Map")
        for k in range(n_components):
            ax[1+k,3*i_plt].imshow(weighted_log_prob_map[i_img][k], vmin=0, vmax=1)
            ax[1+k,3*i_plt].set_title("Component\nLoglikelihood")
            ax[1+k,3*i_plt + 1].imshow(component_distance_map[i_img][k], vmin=0, vmax=1)
            ax[1+k,3*i_plt + 1].set_title("Mahalanobis distance\nto component center")
            ax[1+k,3*i_plt + 2].imshow(component_cosine_distance_map[i_img][k], vmin=0, vmax=1)
            ax[1+k,3*i_plt + 2].set_title("Cosine distance to\ncomponent center")
    for k in range(n_components):
        ax[1+k,0].set_ylabel(f"Component {k}")
    plt.savefig(save_path)
    plt.close()


def visualize_likelihood_per_dimension(stat_dict_norm: dict, stat_dict_anom: dict, save_path: str, resolution: int, n_bins: int = 50):
    labels = stat_dict_anom["labels"]
    labels = F.interpolate(labels, size=(resolution, resolution), mode="nearest")
    pool_size = resolution // stat_dict_anom["anomaly_map"].shape[-1]
    labels_downsized = F.avg_pool2d(labels, kernel_size=pool_size, stride=pool_size)
    labels_fg = labels_downsized >= 0.5
    labels_bg = labels_downsized < 0.5

    fig, ax = plt.subplots(nrows=4, figsize=(8,15))

    log_prob_map_anom = stat_dict_anom["component_log_prob_map"]
    log_prob_map_norm = stat_dict_norm["component_log_prob_map"]
    log_prob_map_norm, log_prob_map_anom = normalize(log_prob_map_norm, log_prob_map_anom, dim=(0,2,3))
    C = log_prob_map_anom.shape[1]
    log_prob_fg = torch.masked_select(log_prob_map_anom, labels_fg).reshape(C, -1)
    log_prob_bg = torch.masked_select(log_prob_map_anom, labels_bg).reshape(C, -1)
    log_prob_map_norm = log_prob_map_norm.permute(1, 0, 2, 3).reshape(C, -1)

    ax[0].hist(log_prob_bg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal")
    ax[0].hist(log_prob_fg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="anomalous")
    ax[0].hist(log_prob_map_norm.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal samples")
    ax[0].legend()
    ax[0].grid(visible=True, which="both")
    ax[0].set_title(r"Log-likelihood per Patch and Component: $\log\mathcal{N}(y_n | \mu_k, \Sigma_k)$")
    ax[0].set_xlabel("Normalized Negative Log-likelihood")
    ax[0].set_ylabel("Number of Components and Patches")

    max_log_prob_map_norm = stat_dict_norm["component_log_prob_map"].amin(1, keepdim=True)  # minimum of component_log_prob_map corresponds to maximum of log_prob as log_prob_map is negative log_prob
    max_log_prob_map_anom = stat_dict_anom["component_log_prob_map"].amin(1, keepdim=True)  # minimum of component_log_prob_map corresponds to maximum of log_prob as log_prob_map is negative log_prob
    max_log_prob_map_norm, max_log_prob_map_anom = normalize(max_log_prob_map_norm, max_log_prob_map_anom, dim=(0,2,3))
    max_log_prob_fg = torch.masked_select(max_log_prob_map_anom, labels_fg)
    max_log_prob_bg = torch.masked_select(max_log_prob_map_anom, labels_bg)

    ax[1].hist(max_log_prob_bg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal")
    ax[1].hist(max_log_prob_fg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="anomalous")
    ax[1].hist(max_log_prob_map_norm.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal samples")
    ax[1].legend()
    ax[1].grid(visible=True, which="both")
    ax[1].set_title(r"Maximum Log-likelihood per Patch and Component: $\max_k\left(\log\mathcal{N}(y_n | \mu_k, \Sigma_k)\right)$")
    ax[1].set_xlabel("Normalized Negative Log-likelihood")
    ax[1].set_ylabel("Number and Patches")

    anomaly_score_norm = -(-stat_dict_norm["weighted_log_prob_map"]).logsumexp(1, keepdim=True)
    anomaly_score_anom = -(-stat_dict_anom["weighted_log_prob_map"]).logsumexp(1, keepdim=True)
    anomaly_score_norm, anomaly_score_anom = normalize(anomaly_score_norm, anomaly_score_anom, dim=(0,2,3))
    anomaly_score_fg = torch.masked_select(anomaly_score_anom, labels_fg)
    anomaly_score_bg = torch.masked_select(anomaly_score_anom, labels_bg)

    ax[2].hist(anomaly_score_bg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal")
    ax[2].hist(anomaly_score_fg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="anomalous")
    ax[2].hist(anomaly_score_anom.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal samples")
    ax[2].legend()
    ax[2].grid(visible=True, which="both")
    ax[2].set_title(r"Log-likelihood per Patch: $\log\left(\sum_k \pi_k \mathcal{N}(y_n | \mu_k, \Sigma_k)\right)$")
    ax[2].set_xlabel("Normalized Negative Log-likelihood")
    ax[2].set_ylabel("Number of Patches")

    distance_map_norm = stat_dict_norm["component_distance_map"].amin(1, keepdim=True)  # minimum of component_log_prob_map corresponds to maximum of log_prob as log_prob_map is negative log_prob
    distance_map_anom = stat_dict_anom["component_distance_map"].amin(1, keepdim=True)  # minimum of component_log_prob_map corresponds to maximum of log_prob as log_prob_map is negative log_prob
    distance_map_norm, distance_map_anom = normalize(distance_map_norm, distance_map_anom, dim=(0,2,3))
    distance_map_fg = torch.masked_select(distance_map_anom, labels_fg)
    distance_map_bg = torch.masked_select(distance_map_anom, labels_bg)

    ax[3].hist(distance_map_bg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal")
    ax[3].hist(distance_map_fg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="anomalous")
    ax[3].hist(distance_map_norm.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal samples")
    ax[3].legend()
    ax[3].grid(visible=True, which="both")
    ax[3].set_title(r"Maximum Log-likelihood per Patch and Component: $\max_k\left(\log\mathcal{N}(y_n | \mu_k, \Sigma_k)\right)$")
    ax[3].set_xlabel("Normalized Negative Log-likelihood")
    ax[3].set_ylabel("Number and Patches")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    for k in range(stat_dict_anom["labels"].shape[0]):
        labels = stat_dict_anom["labels"][k:k+1]
        labels = F.interpolate(labels, size=(resolution, resolution), mode="nearest")
        pool_size = resolution // stat_dict_anom["anomaly_map"].shape[-1]
        labels_downsized = F.avg_pool2d(labels, kernel_size=pool_size, stride=pool_size)
        labels_fg = labels_downsized >= 0.5
        labels_bg = labels_downsized < 0.5

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15,15), gridspec_kw={'width_ratios': [2, 1]})

        log_prob_map_anom = stat_dict_anom["component_log_prob_map"][k:k+1]
        log_prob_map_anom = normalize(log_prob_map_anom, dim=(0,2,3))
        C = log_prob_map_anom.shape[1]
        log_prob_fg = torch.masked_select(log_prob_map_anom, labels_fg).reshape(C, -1)
        log_prob_bg = torch.masked_select(log_prob_map_anom, labels_bg).reshape(C, -1)

        ax[0,0].hist(log_prob_bg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal")
        ax[0,0].hist(log_prob_fg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="anomalous")
        ax[0,0].legend()
        ax[0,0].grid(visible=True, which="both")
        ax[0,0].set_title(r"Log-likelihood per Patch and Component: $\log\mathcal{N}(y_n | \mu_k, \Sigma_k)$")
        ax[0,0].set_xlabel("Normalized Negative Log-likelihood")
        ax[0,0].set_ylabel("Number of Components and Patches")

        max_log_prob_map_anom = stat_dict_anom["component_log_prob_map"][k:k+1].amin(1, keepdim=True)  # minimum of component_log_prob_map corresponds to maximum of log_prob as log_prob_map is negative log_prob
        max_log_prob_map_anom = normalize(max_log_prob_map_anom, dim=(0,2,3))
        max_log_prob_fg = torch.masked_select(max_log_prob_map_anom, labels_fg)
        max_log_prob_bg = torch.masked_select(max_log_prob_map_anom, labels_bg)

        ax[1,0].hist(max_log_prob_bg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal")
        ax[1,0].hist(max_log_prob_fg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="anomalous")
        ax[1,0].legend()
        ax[1,0].grid(visible=True, which="both")
        ax[1,0].set_title(r"Maximum Log-likelihood per Patch and Component: $\max_k\left(\log\mathcal{N}(y_n | \mu_k, \Sigma_k)\right)$")
        ax[1,0].set_xlabel("Normalized Negative Log-likelihood")
        ax[1,0].set_ylabel("Number and Patches")

        anomaly_score_anom = stat_dict_anom["anomaly_map"][k:k+1]
        anomaly_score_anom = normalize(anomaly_score_anom, dim=(0,2,3))
        anomaly_score_fg = torch.masked_select(anomaly_score_anom, labels_fg)
        anomaly_score_bg = torch.masked_select(anomaly_score_anom, labels_bg)

        ax[2,0].hist(anomaly_score_bg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="normal")
        ax[2,0].hist(anomaly_score_fg.reshape(-1).numpy(), bins=n_bins, range=(0,1), histtype="step", density=True, label="anomalous")
        ax[2,0].legend()
        ax[2,0].grid(visible=True, which="both")
        ax[2,0].set_title(r"Log-likelihood per Patch: $\log\left(\sum_k \pi_k \mathcal{N}(y_n | \mu_k, \Sigma_k)\right)$")
        ax[2,0].set_xlabel("Normalized Negative Log-likelihood")
        ax[2,0].set_ylabel("Number of Patches")

        ax[0,1].imshow(normalize(stat_dict_anom["image"])[k].permute(1,2,0).numpy())
        ax[1,1].imshow(normalize(stat_dict_anom["weighted_log_prob_map"], stat_dict_norm["weighted_log_prob_map"])[0][k:k+1].amin(1, keepdim=True).squeeze().numpy())
        ax[2,1].imshow(normalize(stat_dict_anom["anomaly_map"], stat_dict_norm["anomaly_map"])[0][k].squeeze().numpy())
        ax[0,1].axis("off")
        ax[1,1].axis("off")
        ax[2,1].axis("off")
        ax[0,1].set_title("Image")
        ax[1,1].set_title("Labels")
        ax[2,1].set_title("Anomaly Map")

        plt.tight_layout()
        path, filename = os.path.split(save_path)
        path = os.path.join(path, "likelihood_per_dim")
        os.makedirs(path, exist_ok=True)
        filename, ext = os.path.splitext(filename)
        plt.savefig(os.path.join(path, f"{filename}_{k:03d}{ext}"))
        plt.close()


def visualize_test_score_histogram(
    stats_normal: dict,
    stats_anomalous: dict,
    thresholds: list[dict],
    map_key: str,
    save_path: str,
):
    if (
        map_key not in stats_normal.keys() or
        map_key not in stats_anomalous.keys()
    ):
        return
    
    label_shape = stats_normal["labels"].shape[-2:]

    labels = torch.cat([stats_normal["labels"], stats_anomalous["labels"]]).bool().reshape(-1)
    anomaly_map = torch.cat([stats_normal[map_key], stats_anomalous[map_key]])
    anomaly_map = F.interpolate(anomaly_map, label_shape, mode="bilinear").reshape(-1)

    if "object_mask" in stats_normal.keys() and "object_mask" in stats_anomalous.keys():
        object_mask = torch.cat([stats_normal["object_mask"], stats_anomalous["object_mask"]])
        object_mask = F.interpolate(object_mask.float(), label_shape, mode="nearest").bool().reshape(-1)
    else:
        object_mask = torch.ones_like(labels)

    scores_normal = anomaly_map[torch.logical_and(object_mask, ~labels)]
    scores_anomalous = anomaly_map[torch.logical_and(object_mask, labels)]

    _, ax = plt.subplots(1, 1, figsize=(9,5), tight_layout=True)
    ax.hist(scores_normal.numpy(), 100, label="normal", color="tab:blue", density=True, histtype="step")
    ax.hist(scores_anomalous.numpy(), 100, label="anomalous", color="tab:orange", density=True, histtype="step")
    ax.set_title("Score Histogram")
    ax.set_xlabel("Anomaly Score")

    for threshold_dict in thresholds:
        if map_key in threshold_dict.keys():
            ax.plot([threshold_dict[map_key], threshold_dict[map_key]], ax.get_ylim(), label="threshold", color="tab:green")
    ax.legend()
    plt.savefig(save_path)
    plt.close()


def visualize_scores(stats_normal: dict, stats_anomalous: dict, save_path: str):

    # anomaly_map = stats_normal["anomaly_map"][0]
    # image = stats_normal["image"][0].shape
    # labels = stats_normal["labels"][0].shape
    # object_mask = stats_normal["object_mask"][0].shape


    if (
        "anomaly_map" not in stats_normal.keys() or
        "object_mask" not in stats_normal.keys() or
        "anomaly_map" not in stats_anomalous.keys() or
        "object_mask" not in stats_anomalous.keys()
    ):
        return
    
    label_shape = stats_normal["labels"].shape[-2:]

    labels_normal = stats_normal["labels"].bool().reshape(-1)
    labels_anomalous = stats_anomalous["labels"].bool().reshape(-1)

    if labels_normal.any():
        print("Warning: Labels for normal images contain anomalies.")

    # TODO: Resize anomaly maps to full resolution, probably move resizing to validate function
    anomaly_map_normal = F.interpolate(stats_normal["anomaly_map"], label_shape).reshape(-1)
    anomaly_map_anomalous = F.interpolate(stats_anomalous["anomaly_map"], label_shape).reshape(-1)
    object_mask_normal = F.interpolate(stats_normal["object_mask"].float(), label_shape, mode="nearest").bool().reshape(-1)
    object_mask_anomalous = F.interpolate(stats_anomalous["object_mask"].float(), label_shape, mode="nearest").bool().reshape(-1)

    normal_image_scores = anomaly_map_normal[torch.logical_and(object_mask_normal, ~labels_normal)]
    normal_pixel_scores = anomaly_map_anomalous[torch.logical_and(object_mask_anomalous, ~labels_anomalous)]
    anomalous_scores = anomaly_map_anomalous[torch.logical_and(object_mask_anomalous, labels_anomalous)]

    normal_scores = torch.cat([
        normal_image_scores,
        normal_pixel_scores,
    ])

    _, ax = plt.subplots(1, 1, figsize=(9,5), tight_layout=True)
    ax.hist(normal_scores.numpy(), 100, (0, 1), label="normal pixels", color=plt.get_cmap("tab10").colors[0], density=True, histtype="step")
    ax.hist(anomalous_scores.numpy(), 100, (0, 1), label="anomalous pixels", color=plt.get_cmap("tab10").colors[1], density=True, histtype="step")
    ax.set_title("Score Histogram")
    ax.set_xlabel("Anomaly Score")
    ax.legend()
    plt.savefig(save_path)
    plt.close()


def visualize_embedding_maps(
    dataloader,
    embedding_model,
    dpmm,
    pca,
    plot_dir,
    cfg,
):
    os.makedirs(plot_dir, exist_ok=True)
    image_count = 0
    for image, labels in dataloader:
        with torch.no_grad():
            image, grid_size = embedding_model.crop_image(image)
            features = embedding_model.extract_features(image)
            object_mask = embedding_model.compute_background_mask(
                features, grid_size, threshold=cfg.pca_threshold, masking=cfg.object_mask)

            N, _, _ = features.shape
            object_features = features[object_mask]

            labels_imshape = F.interpolate(labels, image.shape[2:], mode="nearest")
            labels_reduced = F.interpolate(labels, grid_size, mode="nearest")
            for n in range(N):
                object_features = features[n, object_mask[n]]

                fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), sharex="row", sharey="row")
                ax[0,0].imshow(normalize(image[n].permute(1,2,0)).cpu().numpy())
                ax[0,1].imshow(labels_imshape[n].permute(1,2,0).cpu().numpy())
                ax[1,0].grid(which="both", visible=True)
                ax[1,1].grid(which="both", visible=True)
                dpmm.visualize(
                    data=object_features,
                    labels=labels_reduced[n].reshape(-1),
                    visualize_model=True,
                    visualize_samples=1e5,
                    pca=pca,
                    ax=ax[1,0],
                    ax_model=ax[1,1],
                    ax_hist=None,
                    scatter_alpha=0.5,
                    cov_color_curve=0,
                    pi_cutoff=1e-6,
                )
                plt.savefig(os.path.join(plot_dir, f"sample_{image_count:04d}.jpg"))
                plt.tight_layout()
                plt.close()
                image_count += 1


def extend_colormap(base_colormap="tab20b", base_num_shades=4, extended_num_shades=6):
    base_colors = plt.get_cmap(base_colormap).colors
    new_colors = []
    for i in range(0, len(base_colors), base_num_shades):
        base_shades = base_colors[i:i+base_num_shades]
        base_shades = np.array(base_shades)

        # Interpolating additional shades
        x = np.linspace(0, 1, len(base_shades))
        x_new = np.linspace(0, 1, extended_num_shades)
        r = np.interp(x_new, x, base_shades[:, 0])
        g = np.interp(x_new, x, base_shades[:, 1])
        b = np.interp(x_new, x, base_shades[:, 2])

        new_colors.extend(zip(r, g, b))

    return new_colors

def get_extended_colormap(num_shades):
    extended_colors = extend_colormap("tab20b", 4, num_shades) + extend_colormap("tab20c", 4, num_shades)

    # Create a ListedColormap
    extended_cmap = mcolors.ListedColormap(extended_colors, name="extended_tab20")

    return extended_cmap
