"""Pretrained ResNet-18 feature extractor for SB3 MultiInputPolicy.

Replaces SB3's default CombinedExtractor (which uses a NatureCNN trained
from scratch) with an ImageNet-pretrained ResNet-18 backbone. Early
layers (conv1, bn1, layer1-3) are frozen; only layer4 is fine-tuned.
This speeds up convergence and improves late-training stability compared
to learning a CNN from scratch (v21 instability: reward -17 -> -746).

The extractor follows the same pattern as SB3's CombinedExtractor: it
iterates over each key in the Dict observation space, using the ResNet
for image keys and a Flatten layer for vector keys (e.g. "state"), then
concatenates all features. This preserves proprioceptive state info.

Input image: (N, 3, 84, 84) float in [0, 1] (CHW after VecTransposeImage
and SB3 uint8/255 preprocessing). ResNet handles 84x84 directly -- the
adaptive avgpool collapses spatial dims to 1x1, yielding a 512-dim vector.
No upsampling to 224x224 is performed (too slow and unnecessary).
"""
import torch as th
import torch.nn as nn
import torchvision.models as models
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict


class ResNetFeaturesExtractor(BaseFeaturesExtractor):
    """ResNet feature extractor for Dict observations.

    Supports ResNet-18 (512-dim) and ResNet-50 (2048-dim).

    For each key in the observation space:
      - image keys (3D Box, uint8): ImageNet-pretrained ResNet with the
        fc layer removed (conv + avgpool -> 512/2048-dim). Early layers frozen,
        only layer4 trainable.
      - vector keys (1D Box): nn.Flatten() (identity passthrough).

    All sub-features are concatenated, matching CombinedExtractor semantics
    so that the downstream mlp_extractor receives both image and state.

    :param observation_space: Dict observation space (e.g. {"image", "state"}).
    :param features_dim: ResNet output dim (512 for resnet18, 2048 for resnet50).
    :param normalized_image: Whether to assume the image is already
        normalized (disables dtype/bounds checks). SB3 MultiInputPolicy
        feeds raw uint8 images that get divided by 255, so leave False.
    :param backbone: "resnet18" or "resnet50". Default: "resnet18".
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 512,
        normalized_image: bool = False,
        backbone: str = "resnet18",
    ) -> None:
        # features_dim here is the per-image ResNet output dim; the real
        # total is computed after iterating all keys (like CombinedExtractor).
        super().__init__(observation_space, features_dim=1)
        self.cnn_output_dim = features_dim

        extractors: dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                if backbone == "resnet50":
                    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                    cnn_output_dim = 2048
                else:
                    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                    cnn_output_dim = 512
                cnn = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
                for param in cnn.parameters():
                    param.requires_grad = False
                for param in cnn[7].parameters():
                    param.requires_grad = True
                extractors[key] = cnn
                total_concat_size += cnn_output_dim
            else:
                # Vector observation (e.g. "state"): flatten passthrough.
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        # Update the real features dim (image + state concatenated).
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
