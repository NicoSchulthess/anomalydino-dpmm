#########################################################################
# File adapted from AnomalyDINO (https://github.com/dammsi/AnomalyDINO) #
# Code from AnomalyDINO is used under the Apache License 2.0            #
#########################################################################

import cv2
from collections import deque
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch_pca import PCA
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

from anomalib.models.components import TimmFeatureExtractor
from src.utils import conv_to_vit_features


# Base Wrapper Class
class VisionTransformerWrapper:
    pca : PCA

    def __init__(
        self,
        model_name,
        device,
        smaller_edge_size=224,
        half_precision=False,
        num_pca_components=-1,
        normalize_embeddings=False,
    ):
        self.device = device
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.model_name = model_name
        self.model = self.load_model()
        self.num_pca_components = num_pca_components
        self.pca = PCA(n_components=num_pca_components) if num_pca_components != -1 else None
        self.normalize_embeddings = normalize_embeddings

    def load_model(self):
        raise NotImplementedError("This method should be overridden in a subclass")
    
    def extract_features(self, img_tensor, use_pca=True):
        raise NotImplementedError("This method should be overridden in a subclass")
    
    def get_embedding_dimension(self):
        raise NotImplementedError("This method should be overridden in a subclass")


# ViT-B/16 Wrapper
class ViTWrapper(VisionTransformerWrapper):
    def load_model(self):
        if self.model_name == "vit_b_16":
            model = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT)
            self.transform = models.ViT_B_16_Weights.DEFAULT.transforms()
            self.grid_size = (14,14)
        elif self.model_name == "vit_b_32":
            model = models.vit_b_32(weights = models.ViT_B_32_Weights.DEFAULT)
            self.transform = models.ViT_B_32_Weights.DEFAULT.transforms()
            self.grid_size = (7,7)
        elif self.model_name == "vit_l_16":
            model = models.vit_l_16(weights = models.ViT_L_16_Weights.DEFAULT)
            self.transform = models.ViT_L_16_Weights.DEFAULT.transforms()
            self.grid_size = (14,14)
        elif self.model_name == "vit_l_32":
            model = models.vit_l_32(weights = models.ViT_L_32_Weights.DEFAULT)
            self.transform = models.ViT_L_32_Weights.DEFAULT.transforms()
            self.grid_size = (7,7)
        else:
            raise ValueError(f"Unknown ViT model name: {self.model_name}")
        
        model.eval()
        # print(self.transform)

        return model.to(self.device)
    
    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor, self.grid_size

    def extract_features(self, img_tensor, use_pca=True):
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            patches = self.model._process_input(img_tensor)
            class_token = self.model.class_token.expand(patches.size(0), -1, -1)
            patches = torch.cat((class_token, patches), dim=1)
            patch_features = self.model.encoder(patches)
            return patch_features[:, 1:, :].squeeze().cpu().numpy()  # Exclude the class token

    def get_embedding_visualization(self, tokens, grid_size = (14,14), resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*self.grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens

    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False):
        # No masking for ViT supported at the moment... (Only DINOv2)
        return np.ones(img_features.shape[0], dtype=bool)
    

# DINOv2 Wrapper
class DINOv2Wrapper(VisionTransformerWrapper):
    def load_model(self):
        model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        model.eval()

        # print(f"Loaded model: {self.model_name}")
        # print("Resizing images to", self.smaller_edge_size)

        # Set transform for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
            ])
        
        return model.to(self.device)


    def get_embedding_dimension(self):
        return self.model.norm.normalized_shape[0] if self.num_pca_components == -1 else self.num_pca_components


    def prepare_image(self, img: str | np.ndarray) -> tuple[torch.Tensor, tuple]:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor: torch.Tensor = self.transform(img)

        # Crop image to dimensions that are a multiple of the patch size
        return self.crop_image(image_tensor)


    def crop_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Cropping the image to a size that is a multiple of the patch size.

        :param torch.Tensor image: Tensor with shape NCHW.
        :return torch.Tensor: Image cropped to multiple of patch size.
        :return tuple: Image size in number of patches.
        """
        _, _, height, width = image.shape
        cropped_height = height - height % self.model.patch_size
        cropped_width = width - width % self.model.patch_size

        image = image[:, :, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image, grid_size


    def extract_features(self, image: torch.Tensor, use_pca=True) -> torch.Tensor:
        """
        Extracting the model features for a given input image.

        :param torch.Tensor image: Tensor of shape NCHW.
        :return torch.Tensor: Intermediate features of size NSD.
        """
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image.half()
            
            image_batch = image.to(self.device)

            tokens = self.model.get_intermediate_layers(image_batch)[0]

            if self.normalize_embeddings:
                tokens = F.normalize(tokens, dim=2)

            if self.pca is not None and use_pca:
                B, N, D = tokens.shape
                tokens = tokens.reshape(B*N, D)
                tokens = self.pca.transform(tokens)
                tokens = tokens.reshape(B, N, -1)
        return tokens


    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens


    def fit_pca(self, img_tensor, cfg):
        if self.pca is None:
            return

        with torch.no_grad():
            img_tensor, grid_size = self.crop_image(img_tensor)
            features = self.extract_features(img_tensor, use_pca=False)
            object_mask = self.compute_background_mask(
                features, grid_size, threshold=cfg.pca_threshold, masking=cfg.object_mask)
            object_features = features[object_mask]
            self.pca.fit(object_features)


    def compute_background_mask_from_image(self, image, threshold = 10, masking_type = None):
        image_tensor, grid_size = self.prepare_image(image)
        tokens = self.extract_features(image_tensor)
        return self.compute_background_mask(tokens, grid_size, threshold, masking_type)


    def compute_background_mask(
        self,
        features: torch.Tensor,
        grid_size: tuple,
        threshold: float = 10,
        masking: bool = False,
        kernel_size: int = 3,
        # border: float = 0.2,
    ) -> torch.Tensor:
        # Kernel size for morphological operations should be odd
        
        N, S, D = features.shape
        device = features.device

        if not masking:
            return torch.ones((N, S), dtype=bool)
        
        pca = PCA(n_components=1, svd_solver='auto')
        
        features = features.cpu().reshape(N * S, D).numpy()
        first_pc = pca.fit_transform(features)
        mask = np.abs(first_pc) > threshold
        mask = mask.reshape(N, *grid_size)              # NHW

        # # test whether the center crop of the images is kept (adaptive masking), adapt if your objects of interest are not centered!
        # m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
        # if m.sum() <=  m.size * 0.35:
        #     mask = - first_pc > threshold

        # postprocess mask, fill small holes in the mask, enlarge slightly
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = mask.transpose(1,2,0).astype(np.uint8)   # HWN
        mask = cv2.dilate(mask, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = mask.astype(bool).transpose(2, 0, 1).reshape(N, S) # NS
        mask = torch.from_numpy(mask).to(device)

        return mask


class ResNetWrapper(VisionTransformerWrapper):
    model: TimmFeatureExtractor

    def load_model(self):
        model_name = self.model_name.split("-")[0]
        self.layers = self.model_name.split("-")[1:]
        
        if "dino" in model_name:
            resnet_dino = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            model = create_feature_extractor(resnet_dino, self.layers)
        else:
            model = TimmFeatureExtractor(model_name, self.layers, pre_trained=True, requires_grad=False)
        model.eval()

        # print(f"Loaded model: {self.model_name}")
        # print("Resizing images to", self.smaller_edge_size)

        # Set transform for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])
        
        return model.to(self.device)


    def get_embedding_dimension(self):
        if self.num_pca_components != -1:
            return self.num_pca_components

        model = self.model if "dino" in self.model_name else self.model.feature_extractor
        embedding_dimension = 0
        for layer in self.layers:
            (name, last_block), = deque(getattr(model, layer).named_children(), 1)
            if hasattr(last_block, "conv3"):
                embedding_dimension += last_block.conv3.out_channels
            elif hasattr(last_block, "conv2"):
                embedding_dimension += last_block.conv2.out_channels
            else:
                raise AttributeError(f"{last_block} has no attribute \"conv2\" or \"conv3\"")
        return embedding_dimension 


    def prepare_image(self, img: str | np.ndarray) -> tuple[torch.Tensor, tuple]:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor: torch.Tensor = self.transform(img)

        # Crop image to dimensions that are a multiple of the patch size
        return self.crop_image(image_tensor)


    def crop_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Cropping the image to a size that is a multiple of the patch size.

        :param torch.Tensor image: Tensor with shape NCHW.
        :return torch.Tensor: Image cropped to multiple of patch size.
        :return tuple: Image size in number of patches.
        """
        return image, None


    def extract_features(self, image: torch.Tensor, use_pca=True) -> torch.Tensor:
        """
        Extracting the model features for a given input image.

        :param torch.Tensor image: Tensor of shape NCHW.
        :return torch.Tensor: Intermediate features of size NSD.
        """
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image.half()
            
            image_batch = image.to(self.device)

            features = self.model(image_batch)
            embeddings = features[self.layers[0]]
            for layer in self.layers[1:]:
                layer_embedding = features[layer]
                layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
                embeddings = torch.cat((embeddings, layer_embedding), 1)

            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, dim=1)

            if self.pca is not None and use_pca:
                N, C, H, W = embeddings.shape
                embeddings = embeddings.permute(0,2,3,1)     # NCHW -> NHWC
                embeddings = embeddings.reshape(N*H*W, C)
                embeddings = self.pca.transform(embeddings)
                embeddings = embeddings.reshape(N, H, W, -1)
                embeddings = embeddings.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return embeddings


    def fit_pca(self, img_tensor, cfg):
        if self.pca is None:
            return
        
        # CUDA warmup:
        zeros = torch.zeros_like(img_tensor[:1])
        _ = self.extract_features(zeros, use_pca=False)

        with torch.no_grad():
            img_tensor, grid_size = self.crop_image(img_tensor)
            features = self.extract_features(img_tensor[:10], use_pca=False)
            features, grid_size = conv_to_vit_features(features)
            object_mask = self.compute_background_mask(
                features, grid_size, threshold=cfg.pca_threshold, masking=cfg.object_mask)
            object_features = features[object_mask]
            self.pca.fit(object_features)


    def compute_background_mask_from_image(self, image, threshold = 10, masking_type = None):
        image_tensor, _ = self.prepare_image(image)
        tokens = self.extract_features(image_tensor)
        return self.compute_background_mask(tokens, masking_type)


    def compute_background_mask(
        self,
        features: torch.Tensor,
        grid_size = None,
        threshold: int = 10,
        masking: bool = False,
    ) -> torch.Tensor:
        
        N, S, D = features.shape
        if not masking:
            return torch.ones((N, S), dtype=bool)
        
        raise NotImplementedError("Background Mask computation not implemented for ResNet")


def get_model(
    model_name,
    device,
    smaller_edge_size=448,
    num_pca_components=-1,
    normalize_embeddings=False,
):
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Smaller edge size: {smaller_edge_size}")

    if "resnet" in model_name:
        return ResNetWrapper(
            model_name,
            device,
            smaller_edge_size,
            num_pca_components=num_pca_components,
            normalize_embeddings=normalize_embeddings,
        )
    elif model_name.startswith("vit"):
        return ViTWrapper(
            model_name,
            device,
            smaller_edge_size,
            num_pca_components=num_pca_components,
            normalize_embeddings=normalize_embeddings,
        )
    elif model_name.startswith("dinov2"):
        return DINOv2Wrapper(
            model_name,
            device,
            smaller_edge_size,
            num_pca_components=num_pca_components,
            normalize_embeddings=normalize_embeddings,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")