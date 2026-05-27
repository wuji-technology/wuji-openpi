"""Policy transforms for Wuji robot (dual-arm dual-dexterous-hand).

Dataset structure:
- observation.state: 54 dims (7 left arm + 20 left hand + 7 right arm + 20 right hand)
- action: 54 dims
- observation.images.cam_left_wrist: (480, 640, 3)
- observation.images.cam_right_wrist: (480, 640, 3)
- observation.images.stereo_right: (480, 640, 3)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Wuji robot action dimension: dual-arm dual-dexterous-hand
WUJI_ACTION_DIM = 54


def make_wuji_example() -> dict:
    """Creates a random input example for the Wuji policy."""
    return {
        "observation/state": np.random.rand(WUJI_ACTION_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the object with both hands",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class WujiInputs(transforms.DataTransformFn):
    """Convert inputs from Wuji dataset to the format expected by Pi0 models.

    Used for both training and inference.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        stereo_right_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": stereo_right_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class WujiOutputs(transforms.DataTransformFn):
    """Convert model outputs back to Wuji action format.

    Used for inference only.
    """

    action_dim: int = WUJI_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim]}
