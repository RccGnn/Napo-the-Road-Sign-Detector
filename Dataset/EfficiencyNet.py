import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_M_Weights
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------------------------------- #
# 1. ON-THE-FLY DATA AUGMENTATION (Phase [5A])
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int):
    """
    Builds the train-time (augmented) and eval-time (deterministic)
    transform pipelines.

    TRAIN transforms include randomized operations (crop, flip, rotation,
    color jitter). Because these are torchvision transforms applied inside
    the Dataset's __getitem__, they run "on-the-fly": a brand-new random
    augmentation is generated every single time an image is loaded, rather
    than being pre-computed once and saved to disk. This is what makes it
    "on-the-fly augmentation" as opposed to an offline augmentation step
    that would create extra image files.

    VAL/TEST transforms are deterministic (only resize + normalize) so
    that validation/test metrics are stable and comparable across epochs.
    """
    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
    mean = weights.meta["mean"]
    std = weights.meta["std"]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transforms, eval_transforms


# --------------------------------------------------------------------------- #
# 2. DATASETS + WEIGHTED SAMPLER (on-the-fly class balancing)
# --------------------------------------------------------------------------- #
def build_dataloaders(data_dir, img_size: int, batch_size: int, num_workers: int):
    """
    Builds train/val/test datasets and dataloaders.

    - train_loader uses a WeightedRandomSampler instead of shuffle=True.
      Each sample's draw-probability is inversely proportional to how
      common its class is, so rare classes get seen roughly as often as
      common ones across an epoch - all WITHOUT physically duplicating any
      image file on disk. Combined with the on-the-fly augmentation above,
      a rare-class image that gets drawn multiple times in the same epoch
      will look different each time it's drawn.
    - val_loader / test_loader use the deterministic eval_transforms and
      normal (non-weighted, non-shuffled) loading, since metrics on these
      splits should reflect the TRUE class distribution, not a rebalanced
      one.
    """
    data_dir = Path(data_dir)
    train_transforms, eval_transforms = build_transforms(img_size)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transforms)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transforms)

    # --- Weighted sampler: inverse-frequency weighting per class ---
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_dataset.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,   # allows rare-class images to be drawn more than once per epoch
    )

    # NOTE: shuffle=False is required when a sampler is provided - the
    # sampler itself already determines draw order/frequency.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    class_names = train_dataset.classes
    return dataloaders, class_names


# --------------------------------------------------------------------------- #
# 3. MODEL: EfficientNetV2-M
# --------------------------------------------------------------------------- #
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """
    Loads EfficientNetV2-M pretrained on ImageNet from torchvision, and
    replaces its classification head with a fresh Linear layer sized for
    our number of classes.

    The backbone starts FULLY FROZEN here (Phase A state). Phase B later
    unfreezes it explicitly - see set_phase_a() / set_phase_b() below.
    """
    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_m(weights=weights)

    # Start frozen (Phase A: transfer learning state).
    for param in model.parameters():
        param.requires_grad = False

    # torchvision's EfficientNetV2 classifier is:
    #   Sequential(Dropout, Linear(in_features, 1000))
    # Replace only the final Linear layer. New layers default to
    # requires_grad=True, so the head becomes trainable immediately even
    # though the rest of the network above was just frozen.
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    return model


def set_phase_a(model: nn.Module):
    """
    Configures the model for PHASE A (transfer learning):
      - backbone (model.features) FROZEN
      - classification head (model.classifier, the "outer layer") TRAINABLE
    """
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def set_phase_b(model: nn.Module):
    """
    Configures the model for PHASE B (fine-tuning), exactly as specified:
      - backbone (model.features) UNFROZEN (fine-tuned)
      - classification head (model.classifier, the "outer layer") FROZEN

    This mirrors Phase A's split but swapped: Phase A trains only the
    outer layer with the backbone locked; Phase B trains only the backbone
    with the outer layer locked. Each phase therefore updates a distinct,
    non-overlapping set of parameters.
    """
    for param in model.features.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False


# --------------------------------------------------------------------------- #
# 4. SHARED TRAIN / EVAL EPOCH LOGIC
# --------------------------------------------------------------------------- #
def run_epoch(model, dataloader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    running_loss, running_corrects, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total += inputs.size(0)

    torch.set_grad_enabled(True)
    return running_loss / total, running_corrects / total


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, phase_name, checkpoint_path):
    """
    Runs `num_epochs` of train/val passes for ONE phase (A or B). Keeps and
    saves the best-validation-accuracy weights seen so far, across BOTH
    phases (the checkpoint path is shared / reused across the two calls to
    this function so Phase B can improve on Phase A's best result rather
    than overwriting it with a possibly worse one).
    """
    best_acc = 0.0
    if checkpoint_path.exists():
        # Resume "best so far" tracking if Phase A already produced a checkpoint.
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        best_acc = evaluate_accuracy_only(model, dataloaders["val"], device)
        print(f"  Loaded existing checkpoint, current best val_acc={best_acc:.4f}")

    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_acc = run_epoch(model, dataloaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, dataloaders["val"], criterion, optimizer, device, train=False)

        elapsed = time.time() - start
        print(f"[{phase_name}] Epoch {epoch + 1:02d}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, checkpoint_path)
            print(f"  -> New best model saved (val_acc={best_acc:.4f}) to {checkpoint_path}")

    model.load_state_dict(best_weights)
    return model, best_acc


@torch.no_grad()
def evaluate_accuracy_only(model, dataloader, device):
    """Quick helper: validation accuracy only, used to seed best_acc when resuming a checkpoint."""
    model.eval()
    correct, total = 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += inputs.size(0)
    return correct / total


@torch.no_grad()
def evaluate_on_test(model, dataloader, class_names, device):
    """
    Final evaluation on the held-out test set: full per-class
    precision/recall/F1 report plus the confusion matrix, so it's clear
    which of the classes still perform poorly after both training phases.
    """
    model.eval()
    all_preds, all_labels = [], []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print("\n=== TEST SET - Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))

    print("=== TEST SET - Confusion Matrix ===")
    print(confusion_matrix(all_labels, all_preds))


# --------------------------------------------------------------------------- #
# 5. CONFIG (hardcoded - no CLI needed, just click Run in your IDE)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Dataset_split"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 15                              # <-- number of unified classes in YOUR dataset
IMAGE_SIZE = 480                              # EfficientNetV2-M's native pretrained resolution
NUM_WORKERS = 4                               # set to 0 on Windows if you hit multiprocessing issues

# Phase A: transfer learning (frozen backbone, train head only)
PHASE_A_EPOCHS = 5
PHASE_A_LR = 1e-3

# Phase B: fine-tuning (unfrozen backbone, frozen head)
PHASE_B_EPOCHS = 10
PHASE_B_LR = 1e-5
PHASE_B_WEIGHT_DECAY = 1e-4

CHECKPOINT_PATH = BASE_DIR / "best_efficientnetv2m_traffic_signs.pt"
FINAL_WEIGHTS_PATH = BASE_DIR / "efficientnetv2m_final.pth"


# --------------------------------------------------------------------------- #
# 6. MAIN
# --------------------------------------------------------------------------- #
def main():
    print(f"Using device: {DEVICE}")
    print(f"Reading dataset split from: {DATA_DIR}")

    # --- Data: on-the-fly augmentation + weighted sampler (Phase [5A]) ---
    dataloaders, class_names = build_dataloaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)
    print(f"Found {len(class_names)} classes: {class_names}")
    assert len(class_names) == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes but found {len(class_names)} in {DATA_DIR}. "
        "Update NUM_CLASSES in the CONFIG section to match your dataset."
    )

    # --- Model: EfficientNetV2-M, pretrained, backbone frozen initially ---
    model = build_model(NUM_CLASSES, DEVICE)
    criterion = nn.CrossEntropyLoss()   # no manual class weights: the sampler already balances classes

    # ===================== PHASE A: TRANSFER LEARNING ===================== #
    print("\n=== Starting Phase A: Training Head (Frozen Backbone) ===")
    set_phase_a(model)   # backbone frozen, classifier (outer layer) trainable
    trainable_params_a = [p for p in model.parameters() if p.requires_grad]
    optimizer_a = optim.Adam(trainable_params_a, lr=PHASE_A_LR)

    model, best_acc = train_model(
        model, dataloaders, criterion, optimizer_a, DEVICE,
        PHASE_A_EPOCHS, "Phase A", CHECKPOINT_PATH
    )
    print(f"Phase A best val accuracy: {best_acc:.4f}")

    # ===================== PHASE B: FINE-TUNING ===================== #
    print("\n=== Starting Phase B: Fine-tuning (Unfrozen Backbone, Frozen Head) ===")
    set_phase_b(model)   # backbone unfrozen, classifier (outer layer) frozen
    trainable_params_b = [p for p in model.parameters() if p.requires_grad]
    optimizer_b = optim.AdamW(trainable_params_b, lr=PHASE_B_LR, weight_decay=PHASE_B_WEIGHT_DECAY)

    model, best_acc = train_model(
        model, dataloaders, criterion, optimizer_b, DEVICE,
        PHASE_B_EPOCHS, "Phase B", CHECKPOINT_PATH
    )
    print(f"Phase B best val accuracy: {best_acc:.4f}")

    # --- Final evaluation on the held-out test set ---
    evaluate_on_test(model, dataloaders["test"], class_names, DEVICE)

    # --- Save final weights (separate from the "best" checkpoint saved during training) ---
    torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
    print(f"\nTraining complete. Final weights saved to: {FINAL_WEIGHTS_PATH}")


# Running this file directly (e.g. clicking "Run" in your IDE) executes
# main() with the hardcoded CONFIG above - no command-line arguments needed.
if __name__ == "__main__":
    main()