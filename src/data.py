#########################################################################
# File adapted from AnomalyDINO (https://github.com/dammsi/AnomalyDINO) #
# Code from AnomalyDINO is used under the Apache License 2.0            #
#########################################################################


from glob import glob
from natsort import natsorted
import numpy as np
import os
from PIL import Image
import random
import torch
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, object_name, anomaly_types, split):
        path_images_base = cfg.dataset.get(f"path_{split}_images", default_value="")
        path_labels_base = cfg.dataset.get(f"path_{split}_labels", default_value="")

        self.images = []
        self.labels = []
        for anomaly_type in anomaly_types:
            path_images = get_data_dir(path_images_base, cfg.dataset.data_root, object_name, anomaly_type)
            path_labels = get_data_dir(path_labels_base, cfg.dataset.data_root, object_name, anomaly_type)
            
            self.images += natsorted(glob(os.path.join(path_images, "*")))
            self.labels += natsorted(glob(os.path.join(path_labels, "*"))) if path_labels != "" else []

        shots = cfg.shots if split == "train" else -1
        if len(self) < shots:
            print(f"Warning: Not enough samples for {shots}-shot! Only {len(self)} samples available.")
        elif shots > 0:
            indices = np.arange(len(self))
            random.Random(cfg.seed).shuffle(indices)
            self.images = np.array(self.images)[indices[:shots]]
            self.labels = np.array(self.labels)[indices[:shots]] if len(self.labels) > 0 else []


        if cfg.normalization.lower() == "imagenet":
            normalization = transforms.Normalize(   # imagenet defaults
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        elif cfg.normalization.lower() == "none":
            normalization = torch.nn.Identity()

        self.image_transform = transforms.Compose([
            transforms.Resize(size=cfg.resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            normalization,
        ])
        if cfg.resize_labels:
            self.label_transform = transforms.Compose([
                transforms.Resize(size=cfg.resolution, interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
                transforms.ToTensor(),
            ])
        else:
            self.label_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")

        # Label map is all 0 if there is no label path (= no anomaly)
        if len(self.labels) > 0:
            label = Image.open(self.labels[index]).convert("L")
        else:
            label = Image.new("L", image.size)

        image = self.image_transform(image)
        label = self.label_transform(label)
        
        return image, label, image_path


def get_datasets(cfg, object_name, anomaly_types, splits):
    datasets = [Dataset(cfg, object_name, anomaly_types, split) for split in splits]
    return datasets


def get_data_dir(path_expression, data_root, object_name, type_anomaly):
    path = path_expression.format(
        data_root = data_root,
        object_name = object_name,
        type_anomaly = type_anomaly,
    )
    return path


def get_dataset_info(dataset, preprocess):

    if preprocess not in ["informed", "agnostic", "masking_only", "informed_no_mask", "agnostic_no_mask", "force_no_mask_no_rotation", "force_mask_no_rotation", "force_no_mask_rotation", "force_mask_rotation"]:
        # masking only: deactivate rotation, apply masking like in informed/agnostic
        raise ValueError(f"Preprocessing '{preprocess}' not yet covered!")
    
    objects = dataset.objects
    object_anomalies = dataset.object_anomalies
    if dataset.name == "MVTec":
        if preprocess in ["agnostic", "informed", "masking_only"]:
            # Define Masking for the different objects -> determine with Masking Test (see Fig. 2 and discussion in the paper)
            # True: default masking (threshold the first PCA component > 10)
            # False: No masking will be applied
            masking_default = {"bottle": False,      
                                "cable": False,         # no masking
                                "capsule": True,        # default masking
                                "carpet": False,
                                "grid": False,
                                "hazelnut": True,
                                "leather": False,
                                "metal_nut": False,
                                "pill": True,
                                "screw": True,
                                "tile": False,
                                "toothbrush": True,
                                "transistor": False,
                                "wood": False,
                                "zipper": False
                                }
            
        if preprocess in ["informed", "informed_no_mask"]:
            rotation_default = {"bottle": False,
                                "cable": False, 
                                "capsule": False,
                                "carpet": False,
                                "grid": False,
                                "hazelnut": True,       # informed: hazelnut is rotated
                                "leather": False,
                                "metal_nut": False,
                                "pill": False,          # informed: all pills in train are oriented just the same
                                "screw": True,          # informed: screws in train are oriented differently
                                "tile": False,
                                "toothbrush": False,
                                "transistor": False,
                                "wood": False,
                                "zipper": False
                                }

        elif preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        elif preprocess == "masking_only":
            rotation_default = {o: False for o in objects}

        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}

    elif dataset.name == "VisA":
        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}
        else:
            masking_default = {o: True for o in objects}

        if preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        elif preprocess in ["informed", "masking_only", "informed_no_mask"]:
            rotation_default = {o: False for o in objects}

    elif dataset.name == "bras2021":
        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}
        else:
            masking_default = {o: True for o in objects}

        if preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        elif preprocess in ["informed", "masking_only", "informed_no_mask"]:
            rotation_default = {o: False for o in objects}
    else:
        raise ValueError(f"Dataset '{dataset.name}' not yet covered!")

    if preprocess == "force_no_mask_no_rotation":
        masking_default = {o: False for o in objects}
        rotation_default = {o: False for o in objects}
    elif preprocess == "force_mask_no_rotation":
        masking_default = {o: True for o in objects}
        rotation_default = {o: False for o in objects}
    elif preprocess == "force_no_mask_rotation":
        masking_default = {o: False for o in objects}
        rotation_default = {o: True for o in objects}
    elif preprocess == "force_mask_rotation":
        masking_default = {o: True for o in objects}
        rotation_default = {o: True for o in objects}

    return objects, object_anomalies, masking_default, rotation_default
