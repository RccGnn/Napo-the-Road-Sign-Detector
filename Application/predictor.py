import torch
import torch.nn as nn
import time
from torchvision import models, transforms, datasets
from pathlib import Path

# ---
# Percorsi
# ---
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "Dataset" / "Dataset_split" / "train"
MODELS_DIR = BASE_DIR / "Dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---
# Classi
# ---
train_dataset = datasets.ImageFolder(TRAIN_DIR)
class_names = train_dataset.classes
NUM_CLASSES = len(class_names)

# ---
# Configurazione modelli
# ---
MODEL_CONFIG = {
    "EfficientNet": {
        "path": MODELS_DIR / "efficientnetv2_finetuned.pth",
        "img_size": 420
    },

    "ConvNeXt": {
        "path": MODELS_DIR / "best_convnext_traffic_signs.pt",
        "img_size": 320
    }
}

# ---
# Caricamento modello
# ---
_loaded_models = {}


def load_model(model_name):

    if model_name in _loaded_models:
        return _loaded_models[model_name]

    if model_name == "EfficientNet":

        model = models.efficientnet_v2_m(weights=None)

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            NUM_CLASSES
        )

    elif model_name == "ConvNeXt":

        model = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )

        in_features = model.classifier[2].in_features

        model.classifier[2] = nn.Linear(
            in_features,
            NUM_CLASSES
        )

    else:
        raise ValueError(f"Modello '{model_name}' non supportato.")

    model.load_state_dict(
        torch.load(
            MODEL_CONFIG[model_name]["path"],
            map_location=device
        )
    )

    model.to(device)
    model.eval()
    _loaded_models[model_name] = model

    return model


# --------------------------------------------------
# Trasformazioni
# --------------------------------------------------

def get_transform(model_name):

    img_size = MODEL_CONFIG[model_name]["img_size"]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# --------------------------------------------------
# Predizione
# --------------------------------------------------

def predict(image, model_name):

    model = load_model(model_name)
    transform = get_transform(model_name)
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    # sincronizzazione GPU per tempo corretto
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()


    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)


    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - start_time


    top3_prob, top3_idx = torch.topk(
        probabilities,
        3
    )


    results = []

    for p, idx in zip(top3_prob[0], top3_idx[0]):

        results.append(
            (
                class_names[idx.item()],
                float(p.item()*100)
            )
        )

    return results, inference_time