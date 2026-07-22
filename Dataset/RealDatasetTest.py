"""
test_model_on_new_dataset_optimized.py

Valuta un modello addestrato (.pt/.pth) su un NUOVO test set
(cartella di immagini piatta + CSV).
Versione ottimizzata con DataLoader (Batch Processing) e grafici Seaborn.
"""

import csv
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    convnext_base,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
)
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix

# Se hai utility esterne, lasciale pure
# from Scripts.utilities.utility import *

# ===========================================================================
# 1. FUNZIONI E CLASSI DI SUPPORTO
# ===========================================================================

def load_labels(csv_path: Path, image_dir: Path) -> dict:
    """Legge il CSV e filtra le immagini effettivamente presenti nella cartella."""
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filename = row["filename"].strip()
            cls_name = row["class"].strip()

            # Controllo esistenza file per evitare crash nel Dataset
            if (image_dir / filename).exists():
                labels[filename] = cls_name
            else:
                print(f"  [!] WARNING: file mancante, salto: {filename}")
    return labels

# Il Custom Dataset che permette a PyTorch di unire CSV e immagini per il DataLoader
class CSVTrafficSignDataset(Dataset):
    def __init__(self, image_dir: Path, labels_dict: dict, class_to_idx: dict, transform):
        self.image_dir = image_dir
        self.labels_dict = labels_dict
        self.filenames = list(labels_dict.keys())
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        true_class = self.labels_dict[filename]
        label_idx = self.class_to_idx[true_class]

        img_path = self.image_dir / filename
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)

        return tensor, label_idx


# ===========================================================================
# 2. CONFIGURAZIONI DEL MODELLO
# ===========================================================================

MODEL_REGISTRY = {
    "convnext_base": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "build_fn": lambda: convnext_base(weights=None),
        "classifier_index": 2,
    },
    "efficientnet_v2_m": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "build_fn": lambda: efficientnet_v2_m(weights=None),
        "classifier_index": 1,
    },
}

def build_transform(model_type: str, img_size: int):
    config = MODEL_REGISTRY[model_type]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["mean"], std=config["std"]),
    ])

def load_model(model_type: str, checkpoint_path: Path, num_classes: int, device: torch.device):
    config = MODEL_REGISTRY[model_type]
    model = config["build_fn"]()

    idx = config["classifier_index"]
    in_features = model.classifier[idx].in_features
    model.classifier[idx] = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


# ===========================================================================
# 3. INFERENZA E VISUALIZZAZIONE
# ===========================================================================

@torch.no_grad()
def predict_all(model, dataloader, device: torch.device):
    """Esegue l'inferenza usando il batching per massima velocità."""
    y_true, y_pred = [], []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred


def evaluate_and_plot(y_true, y_pred, class_names: list):
    """Stampa le metriche e genera i file PNG (Confusion Matrix + F1 Scores)."""
    print("\n=== Classification Report ===")

    # FIX: Aggiunto labels=range(len(class_names))
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=range(len(class_names)),
        digits=4,
        zero_division=0
    )
    print(report_text)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=range(len(class_names)),
        average="macro",
        zero_division=0
    )
    print(f"Macro-averaged  -> Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}\n")

    # 1. Matrice di Confusione (Heatmap)
    # FIX: Aggiunto labels=range(len(class_names)) per forzare la matrice 15x15
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Classe Predetta')
    plt.ylabel('Classe Reale')
    plt.title('Matrice di Confusione')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png", dpi=150)
    print(" -> Salvato: confusion_matrix_test.png")

    # 2. Grafico F1-Score per ogni classe
    # FIX: Aggiunto labels=range(...) anche qui
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=range(len(class_names)),
        zero_division=0,
        output_dict=True
    )

    f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=f1_scores, y=class_names, palette='viridis', hue=class_names, legend=False)
    plt.xlabel('F1-Score')
    plt.title('F1-Score per Singola Classe')
    plt.xlim(0, 1.0)  # Il punteggio va da 0 a 1
    plt.tight_layout()
    plt.savefig("f1_scores_test.png", dpi=150)
    print(" -> Salvato: f1_scores_test.png")

    plt.show()  # Mostra a schermo i grafici generati

# ===========================================================================
# 4. CONFIGURAZIONE DA MODIFICARE
# ===========================================================================

BASE_DIR = Path(__file__).parent
# TEST_IMAGE_DIR = get_dataset_dir() / "NAPO Immagini" # Decommenta se usi la tua utility
TEST_IMAGE_DIR = BASE_DIR / "NAPO Immagini"
TEST_CSV_PATH = TEST_IMAGE_DIR / "dataset.csv"

# Scegli QUALE modello testare (scrivi "convnext_base" o "efficientnet_v2_m")
#MODEL_TYPE = "convnext_base"
MODEL_TYPE= "efficientnet_v2_m"

CHECKPOINT_PATHS = {
    "convnext_base": BASE_DIR / "best_convnext_traffic_signs.pt",
    "efficientnet_v2_m": BASE_DIR / "efficientnetv2_finetuned.pth",
}

IMG_SIZES = {
    "convnext_base": 320,
    "efficientnet_v2_m": 420,
}

# La configurazione si adatta automaticamente al MODEL_TYPE scelto
CHECKPOINT_PATH = CHECKPOINT_PATHS[MODEL_TYPE]
IMG_SIZE = IMG_SIZES[MODEL_TYPE]
BATCH_SIZE = 6

CLASS_NAMES = sorted([
    "Hazard", "No Entry", "Round-About", "Speed Bump Ahead", "Stop",
    "crosswalk", "do_not_turn", "green_traffic_light", "obligation",
    "parking", "red_traffic_light", "slope", "speedlimit",
    "yellow_traffic_light", "yield sign",
])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    # 1. Carica etichette valide
    labels = load_labels(TEST_CSV_PATH, TEST_IMAGE_DIR)

    # 2. Prepara il DataLoader
    transform = build_transform(MODEL_TYPE, IMG_SIZE)
    dataset = CSVTrafficSignDataset(TEST_IMAGE_DIR, labels, class_to_idx, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Carica il modello
    model = load_model(MODEL_TYPE, CHECKPOINT_PATH, len(CLASS_NAMES), DEVICE)

    print(f"\nInizio valutazione su {len(labels)} immagini con {MODEL_TYPE} ...")
    print(f"Grandezza Batch: {BATCH_SIZE} | Dispositivo: {DEVICE}")

    # 4. Inferenza super veloce
    y_true, y_pred = predict_all(model, dataloader, DEVICE)

    # 5. Stampa report e genera file PNG
    evaluate_and_plot(y_true, y_pred, CLASS_NAMES)

if __name__ == "__main__":
    main()