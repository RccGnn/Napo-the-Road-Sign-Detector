import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------------------------------- #
# 1. FOCAL LOSS
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha) if alpha is not None else setattr(self, "alpha", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1.0 - p_t).pow(self.gamma)
        loss = -focal_term * log_p_t
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    # --------------------------------------------------------------------------- #


# 2. DATA LOADING (NO AUGMENTATION)
# --------------------------------------------------------------------------- #
def build_dataloaders(data_dir: str, img_size: int, batch_size: int, num_workers: int):
    print(f"  --> Inizializzazione trasformazioni (Resize a {img_size}x{img_size} e Normalizzazione ImageNet)...")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    deterministic_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    data_dir = Path(data_dir)
    print(f"  --> Lettura delle cartelle da: {data_dir} ...")

    image_datasets = {
        split: datasets.ImageFolder(data_dir / split, transform=deterministic_transform)
        for split in ("train", "val", "test")
    }

    print(
        f"  --> Trovate immagini: TRAIN={len(image_datasets['train'])}, VAL={len(image_datasets['val'])}, TEST={len(image_datasets['test'])}")

    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(image_datasets["val"], batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(image_datasets["test"], batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True),
    }

    class_names = image_datasets["train"].classes
    return dataloaders, class_names, image_datasets


# --------------------------------------------------------------------------- #
# 3. MODEL: ConvNeXt AS A FROZEN FEATURE EXTRACTOR
# --------------------------------------------------------------------------- #
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    print(f"  --> Scaricamento/Caricamento dei pesi pre-addestrati ConvNeXt-Base (ImageNet)...")
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    model = convnext_base(weights=weights)

    print(f"  --> Congelamento (freeze) del corpo del modello (nessun aggiornamento dei pesi base)...")
    for param in model.parameters():
        param.requires_grad = False

    print(f"  --> Sostituzione dell'ultimo strato classificatore per adattarlo a {num_classes} classi...")
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    model = model.to(device)
    print(f"  --> Modello spostato con successo su: {device}")
    return model


# --------------------------------------------------------------------------- #
# 4. TRAIN / EVALUATE LOOP
# --------------------------------------------------------------------------- #
def run_epoch(model, dataloader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    running_loss, running_correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)

    torch.set_grad_enabled(True)
    return running_loss / total, running_correct / total


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, checkpoint_path):
    print("\n[INFO] Inizio ciclo di addestramento (Transfer Learning)...")
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_acc = run_epoch(model, dataloaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, dataloaders["val"], criterion, optimizer, device, train=False)

        elapsed = time.time() - start
        print(f"  [Epoca {epoch + 1:02d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Tempo: {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, checkpoint_path)
            print(f"    ⭐ Trovato un miglioramento! Nuovo modello salvato ({checkpoint_path})")

    print("\n[INFO] Addestramento completato. Caricamento dei pesi migliori in memoria.")
    model.load_state_dict(best_weights)
    return model, best_acc


@torch.no_grad()
def evaluate_on_test(model, dataloader, class_names, device):
    print("\n[FASE 5] Inizio Valutazione sul Test Set Inedito...")
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print("\n=== TEST SET - Report di Classificazione ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    print("\n=== TEST SET - Matrice di Confusione ===")
    print(confusion_matrix(all_labels, all_preds))


# --------------------------------------------------------------------------- #
# 5. HARDCODED CONFIGURATION (IDE RUN READY)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Dataset_split"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model & Hyperparameters
NUM_CLASSES = 15
IMG_SIZE = 320
EPOCHS = 10
LEARNING_RATE = 1e-3
GAMMA = 2.0
NUM_WORKERS = 4
CHECKPOINT_PATH = BASE_DIR / "best_convnext_traffic_signs.pt"


# --------------------------------------------------------------------------- #
# 6. MAIN EXECUTION
# --------------------------------------------------------------------------- #
def main():
    print("=" * 60)
    print(" AVVIO PIPELINE DI ADDESTRAMENTO: CONVNEXT + FOCAL LOSS")
    print("=" * 60)
    print(f"\n[FASE 1] Configurazione Hardware e Hyperparametri")
    print(f"  --> Dispositivo in uso: {DEVICE}")
    print(f"  --> Batch Size: {BATCH_SIZE} | Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  --> Epoche previste: {EPOCHS} | Learning Rate: {LEARNING_RATE}")

    print(f"\n[FASE 2] Caricamento Dati")
    dataloaders, class_names, image_datasets = build_dataloaders(
        DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS
    )
    print(f"  --> Classi rilevate ({len(class_names)}): {class_names}")
    assert len(class_names) == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes but found {len(class_names)} in {DATA_DIR}."
    )

    print(f"\n[FASE 3] Configurazione del Modello ConvNeXt")
    model = build_model(NUM_CLASSES, DEVICE)

    # --- Loss Setup ---
    criterion = FocalLoss(gamma=GAMMA, reduction="mean")
    print(f"  --> Funzione di Errore configurata: Focal Loss (Gamma={GAMMA})")

    # --- Optimizer Setup ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)
    print(f"  --> Ottimizzatore configurato: Adam (Solo per i nuovi layer della testa)")

    print(f"\n[FASE 4] Addestramento")
    model, best_val_acc = train_model(
        model, dataloaders, criterion, optimizer, DEVICE, EPOCHS, str(CHECKPOINT_PATH)
    )
    print(f"\n[RISULTATO FINALE FASE 4] Migliore accuratezza sul set di Validazione: {best_val_acc * 100:.2f}%")

    # --- Test Set Verification ---
    evaluate_on_test(model, dataloaders["test"], class_names, DEVICE)
    print("\nPipeline completata con successo! Ciao!")


if __name__ == "__main__":
    main()