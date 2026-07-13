import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
import copy
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# ==========================================
# 1. Configuration & Setup
# ==========================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Dataset_split"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================
# 2. On-the-fly Augmentation
# ==========================================
IMAGE_SIZE = 420

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# 5. Training Loop Definition
# ==========================================
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, patience=8):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"ATTENZIONE: loss non valida ({loss.item()}) — epoch {epoch + 1}, batch skippato")
                            continue

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')
            if phase == 'train':
                print(f"Memoria GPU allocata: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1-macro: {epoch_f1:.4f}')

            if phase == 'val':
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

        print()

        if patience_counter >= patience:
            print(f'Early stopping: nessun miglioramento F1-macro da {patience} epoche')
            break

    print(f'Best Val F1-macro: {best_f1:4f}')
    model.load_state_dict(best_model_wts)
    return model


# ==========================================
# ESECUZIONE SICURA PER WINDOWS MULTIPROCESSING
# ==========================================
if __name__ == '__main__':

    # Caricamento Dataset (con percorsi assoluti sicuri)
    train_dataset = datasets.ImageFolder(root=str(DATA_DIR / "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=str(DATA_DIR / "val"), transform=val_transforms)

    # Calcolo automatico del numero di classi!
    NUM_CLASSES = len(train_dataset.classes)
    print(f"Classi rilevate nel dataset: {NUM_CLASSES}")

    # ==========================================
    # 3. Automatic Balancing via Sampler
    # ==========================================
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_dataset.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}

    # ==========================================
    # 4. Model Setup (EfficientNetV2)
    # ==========================================
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)

    # Congela il backbone
    for param in model.parameters():
        param.requires_grad = False

    # Sostituisce la testa classificatore
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ==========================================
    # 6. Execution: Two-Phase Transfer Learning
    # ==========================================
    print("\n=== Starting Phase A: Training Head (Frozen Backbone) ===")
    optimizer_A = optim.Adam(model.classifier.parameters(), lr=1e-3)
    model = train_model(model, dataloaders, criterion, optimizer_A, num_epochs=5, patience=8)

    print("\n=== Starting Phase B: Fine-tuning (Unfreezing top layers) ===")
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    optimizer_B = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
    model = train_model(model, dataloaders, criterion, optimizer_B, num_epochs=10,patience=10)

    # Salvataggio
    torch.save(model.state_dict(), 'efficientnetv2_finetuned.pth')
    print("\nTraining complete. Weights saved.")


    print("\n=== Evaluating on Test Set ===")

    test_dataset = datasets.ImageFolder(root=str(DATA_DIR / "test"), transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    class_names = train_dataset.classes

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    precision = [report[c]['precision'] for c in class_names]
    recall = [report[c]['recall'] for c in class_names]
    f1 = [report[c]['f1-score'] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-score')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Precision / Recall / F1 per classe')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('classification_report.png', dpi=150)
    plt.show()

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("\nConfusion matrix salvata in confusion_matrix.png")