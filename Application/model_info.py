import torch
from pathlib import Path
from torchvision import models


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "Dataset"


MODEL_INFO = {
    "EfficientNet": {
        "path": MODELS_DIR / "efficientnetv2_finetuned.pth",
        "accuracy": "DA INSERIRE",
    },

    "ConvNeXt": {
        "path": MODELS_DIR / "best_convnext_traffic_signs.pt",
        "accuracy": "DA INSERIRE",
    }
}

def get_model_size(model_name):

    path = MODEL_INFO[model_name]["path"]
    size_mb = path.stat().st_size / (1024 * 1024)
    return round(size_mb, 2)



def get_parameters(model_name):

    if model_name == "EfficientNet":
        model = models.efficientnet_v2_m(
            weights=None
        )

    elif model_name == "ConvNeXt":

        model = models.convnext_base(
            weights=None
        )

    else:
        return None


    params = sum(
        p.numel()
        for p in model.parameters()
    )

    return round(params / 1e6, 2)



def get_accuracy(model_name):

    return MODEL_INFO[model_name]["accuracy"]