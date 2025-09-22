#########################################################################
# File adapted from AnomalyDINO (https://github.com/dammsi/AnomalyDINO) #
# Code from AnomalyDINO is used under the Apache License 2.0            #
#########################################################################


import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf, DictConfig
import os
import random
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F
import wandb
import yaml


def normalize(*tensors: torch.Tensor, dim: list = None):
    maxval = torch.stack([t.amax(dim, keepdim=True) for t in tensors]).amax(dim=0)
    minval = torch.stack([t.amin(dim, keepdim=True) for t in tensors]).amin(dim=0)
    normalized_tensors = [(t - minval) / (maxval - minval) for t in tensors]

    if len(tensors) == 1:
        return normalized_tensors[0]
    
    return normalized_tensors


def makedirs(*folders):
    if len(folders) == 0:
        return "."
    
    path = os.path.join(*folders)
    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    return path


def seed_everything(seed:int=0) -> None:
    """
    Seed method for PyTorch for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(path, **kwargs):
    torch.save(kwargs, path)


def ifequal_else(variable: str, expected_value: str, true_value: str, false_value: str = "", *, _root_):
    return true_value if variable == expected_value else false_value


def load_config(path):
    OmegaConf.register_new_resolver("ifequal_else", ifequal_else, replace=True)
    cfg = OmegaConf.load(path)
    return cfg


def config_get(
    cfg,
    key: str,
    default: any = None,
    throw_on_resolution_failure: bool = True,
    throw_on_missing: bool = False,
):
    if cfg is None:
        return default

    return OmegaConf.select(
        cfg,
        key,
        default=default,
        throw_on_resolution_failure=throw_on_resolution_failure,
        throw_on_missing=throw_on_missing,
    )


def dump_config(path, cfg):
    if isinstance(cfg, DictConfig):
        with open(path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
    elif isinstance(cfg, dict):
        with open(path, "w") as f:
            yaml.dump(cfg, f)
    else:
        raise NotImplementedError(
            f"dump_config supports DictConfig and dict, but got {type(cfg).__name__}")


def setup_wandb(logdir, is_resumed, cfg):
    wandb_params_path = os.path.join(logdir, 'wandb.yaml')
    if os.path.exists(wandb_params_path) and is_resumed:
        wandb_params = load_config(wandb_params_path)
    else:
        wandb_params = {'id': wandb.util.generate_id()}

    wandb.init(
        project=cfg.wandb_project,
        config=cfg,
        resume='allow',
        id=wandb_params['id'],
    )

    wandb_params['name'] = wandb.run.name
    wandb_params['project'] = wandb.run.project
    wandb_params['link'] = f'https://wandb.ai/{wandb.run.path}'
    dump_config(wandb_params_path, wandb_params)
    
    wandb_dir = wandb.run.dir
    return wandb_dir


def setup_device(device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[-1])
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    return device


def augment_image(img_ref, augmentation = "rotate", angles = [0, 45, 90, 135, 180, 225, 270, 315]):
    """
    Simply augmentation of images, currently just rotation.
    """
    imgs = []
    if augmentation == "rotate":
        for angle in angles:
            imgs.append(rotate_image(img_ref, angle))
    return imgs


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)
    return result


def dists2map(dists, img_shape):
    # resize and smooth the distance map
    # caution: cv2.resize expects the shape in (width, height) order (not (height, width) as in numpy, so indices here are swapped!
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation = cv2.INTER_LINEAR)
    dists = gaussian_filter(dists, sigma=4)
    return dists


def resize_mask_img(mask, image_shape, grid_size1):
    mask = mask.reshape(grid_size1)
    imgd1 = image_shape[0] // grid_size1[0]
    imgd2 = image_shape[1] // grid_size1[1]
    mask = np.repeat(mask, imgd1, axis=0)
    mask = np.repeat(mask, imgd2, axis=1)
    return mask


def plot_ref_images(img_list, mask_list, vis_background_list, grid_size, save_path, title = "Reference Images", img_names = None):
    k = min(len(img_list), 32)  # reduce max number of ref samples to plot to 32

    n_aug = len(img_list)//len(img_names)

    fig, axs = plt.subplots(k, 3, figsize=(10, 3.5*k))
    if k == 1:
        axs = axs.reshape(1, -1)
    for i in range(k):
        axs[i, 0].imshow(img_list[i])
        axs[i, 1].imshow(vis_background_list[i])
        axs[i, 2].imshow(img_list[i])
        axs[i, 2].imshow(resize_mask_img(mask_list[i], img_list[i].shape, grid_size), alpha=0.5)
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        axs[i, 2].axis('off')
        if i % n_aug == 0:
            axs[i, 0].title.set_text(f"Image: {img_names[i // n_aug]}")
        else:
            axs[i, 0].title.set_text(f"Augmentation of Image {img_names[i // n_aug]}")
        axs[i, 1].title.set_text("PCA + Mask")
        axs[i, 2].title.set_text("Mask")
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + "reference_samples.png")
    plt.close()


def conv_to_vit_features(features):
    N, C, H, W = features.shape
    grid_size = (H, W)
    features = features.permute(0,2,3,1).reshape(N, H*W, C)  # NCHW -> NSD
    return features, grid_size


def vit_to_conv_features(features, grid_size):
    N, S, D = features.shape
    features = features.reshape(N, grid_size[0], grid_size[1], D).permute(0,3,1,2)  # NSD -> NCHW
    return features


def resample_features(features, grid_size, target_res):
    if grid_size is None:
        features, grid_size = conv_to_vit_features(features)  # features: NCHW -> NSD

    if target_res > 0:
        features = vit_to_conv_features(features, grid_size)  # features: NSD -> NCHW
        features = F.interpolate(features, (target_res, target_res), mode="bilinear")
        features, grid_size = conv_to_vit_features(features)  # features: NCHW -> NSD

    return features, grid_size
