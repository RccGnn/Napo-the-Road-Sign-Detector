import os
import pathlib
import random
import sys
import zipfile
from tqdm import tqdm
import pandas as pd
import shutil
import re
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from Scripts.utilities import utility as u


# Colori della barra di progresso
colori = ["green", "red", "blue", "yellow", "cyan", "magenta", "white", "black"]


IMG_SIZE = 400
BATCH_SIZE = 32
BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / 'Dataset_split'

# Controlla se la GPU è disponibile
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# FASE 1: SPLIT DEL DATASET (ANTI-LEAKAGE)
# ==========================================
def esegui_split_dataset():
    dataset_root = u.get_dataset_dir() / "preprocessed_images"
    csv = u.get_dataset_dir() / "preprocessed_images.csv"
    out_dir = get_dataset_dir().parent / "Dataset_split"

    df = pd.read_csv(csv)  # colonne: nome, classe

    print(df.head())
    print(df.columns)
    print(df['classe'].unique()[:10])
    print(df['nome'].head())

    df['src_path'] = df.apply(lambda r: dataset_root / r['classe'] / r['nome'], axis=1)
    df = df[df['src_path'].apply(lambda p: p.exists())].reset_index(drop=True)

    df['group'] = df['nome'].apply(lambda f: re.sub(r'_\d+$', '', Path(f).stem))

    # Stage 1: separa train (~70%) da temp (~30%)
    # n_splits=3 -> ogni fold è ~33% di test, il resto (~67%) è train. Il più vicino a 70/30 con k intero.
    sgkf1 = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    train_idx, temp_idx = next(sgkf1.split(df, df['classe'], df['group']))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    # Stage 2: divide temp a metà -> ~15% val, ~15% test
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    val_idx, test_idx = next(sgkf2.split(df_temp, df_temp['classe'], df_temp['group']))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    # Sanity check anti-leakage
    assert not (set(df_train['group']) & set(df_val['group']))
    assert not (set(df_train['group']) & set(df_test['group']))
    assert not (set(df_val['group']) & set(df_test['group']))

    print(f"Train: {len(df_train)} ({len(df_train) / len(df):.1%})")
    print(f"Val:   {len(df_val)} ({len(df_val) / len(df):.1%})")
    print(f"Test:  {len(df_test)} ({len(df_test) / len(df):.1%})")

    # Sanity check stratificazione: distribuzione classei per split
    for name, d in [('train', df_train), ('val', df_val), ('test', df_test)]:
        print(f"\n{name}:")
        print(d['classe'].value_counts(normalize=True).round(3))

    for split_name, split_df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        for _, row in split_df.iterrows():
            dest_dir = out_dir / split_name / row['classe']
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(row['src_path'], dest_dir / row['nome'])

    print("\nFatto. Struttura pronta in", out_dir)


def get_dataset_dir() -> pathlib.Path:
    """
    Restituisce il percorso assoluto di un file nella cartella Dataset.
    SI ASSUME CHE IL PROGETTO ABBIA LA SEGUENTE STRUTTURA:
        <Napo-the-Road-Sign-Detector>/
            Dataset/
            Scripts/
                image_manipulation/
                    preprecessing.py <<<

    Returns:
        pathlib.Path: il path delle cartelle Dataset.
    """
    # __file__ è il path assoluto dello script in esecuzione
    this_file = pathlib.Path(__file__).resolve()

    # Risali le cartelle per costruire il path della cartella 'Dataset'
    for parent in this_file.parents:
        # Verifica se /Dataset è una cartella nel path
        dataset_dir = parent / "Dataset"
        if os.path.isdir(dataset_dir):
            return dataset_dir

    # Lancia errore
    raise FileNotFoundError("Cartella 'Dataset' non trovata.")


def view_csv(zip_path: str) -> None:
    """
    Visualizza il csw tramite pandas
    Args:
        zip_path: Nome del dataset zippato, cioè [nome_dataset.zip]
    Returns:
        None: visualizza a schermo il dataset
    """
    # Risolve zip_path rispetto a Dataset/ e non rispetto alla cwd
    resolved = get_dataset_dir() / zip_path

    if resolved.suffix == ".csv":
        df = pd.read_csv(resolved)
    else:
        with zipfile.ZipFile(resolved, "r") as archive:
            file_list = archive.namelist()
            df = pd.DataFrame()
            from Scripts.image_manipulation.csv_manipulation import find_csv_files
            find_csv_files()

    # ← Il try ora è FUORI da if/else, viene sempre eseguito
    try:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', 50)

        print("\n--- ANTEPRIMA DEL DATASET ---")
        print(df)

        print("\n--- INFORMAZIONI SUL DATASET ---")
        print(f"Totale righe: {len(df)}")

        print(f"Righe con class NaN: {df['class'].isna().sum()}")

        conteggio = df['class'].value_counts(dropna=False)
        print(f"Classi uniche trovate: {conteggio.index.tolist()}")

        for classe, count in conteggio.items():
            print(f"{classe}: {count}")

        print(f"Totale: {len(df)} - Calcolati: {conteggio.sum()}\n")

    except FileNotFoundError:
        print(f"Errore: Il file '{zip_path}' non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

def organize_images_by_class(images_dir, csv_path, filename_column, class_column) -> None:
    """
        Crea una sotto-cartella per ogni classe distinta presente nel csv e sposta
        dentro ciascuna di questa le immagini corrispondenti per classe.

        Args:
            images_dir: Path (o stringa) alla cartella che contiene le immagini
                "piatte" da organizzare. Al termine, questa cartella conterrà
                solo le sotto-cartelle di classe (i file spostati non restano
                più al livello radice).
            csv_path: Path (o stringa) al file csv con le colonne filename/class.
            filename_column: Nome della colonna del csv che contiene il nome
                del file immagine (usato solo per calcolare la radice di ricerca,
                non per il match esatto).
            class_column: Nome della colonna del csv che contiene l'etichetta
                di classe, usata come nome della sotto-cartella di destinazione.

        Returns:
            None.

        Side effects:
            - Crea una sotto-cartella dentro `images_dir` per ogni
              classe unica trovata nel csv (non fallisce se la cartella esiste già).
            - Sposta (non copia) ogni file trovato da `images_dir` alla sotto-cartella
              della sua classe.
        """
    # Apre il path delle stringhe
    images_dir = pathlib.Path(images_dir)
    csv_path = pathlib.Path(csv_path)
    df = pd.read_csv(csv_path)

    # (Differenza tra insiemi) Verifica se le colonne passate e le colonne del csv coincidono
    missing_cols = {filename_column, class_column} - set(df.columns)
    if missing_cols: # Entra se != 0
        raise ValueError(
            f"Colonne previste: {missing_cols} in {csv_path}. "
            f"Colonne passate: {list(df.columns)}"
        )

    # Crea una cartella per ogni classe
    class_names = df[class_column].unique()
    for class_name in class_names:
        (images_dir / str(class_name)).mkdir(parents=True, exist_ok=True)

    print(f"Create {len(class_names)} cartelle (per classe) nella cartella {images_dir}")

    moved, errors = 0, 0

    total_target = len(df) # Numero di righe di csv da processare
    with tqdm(total=total_target, desc="Spostamento immagini", colour="green", unit="img", file=sys.stdout) as pbar:
        # Per ogni riga del csv...
        for ignore, row in df.iterrows(): # ignore viene usato come buffer per il primo parametro di iterrows, che non server
            filename = str(row[filename_column])
            class_name = str(row[class_column])

            src = images_dir / filename
            dst = images_dir / class_name
            try:
                shutil.move(str(src), str(dst))
                moved += 1
            except Exception as e:
                print(f"Errore: {src} - {e}")
                errors += 1

            # Aggiorna la barra di stato
            pbar.colour = random.choice(colori)
            pbar.update(1)

    print(f"\n Spostati: {moved} file | Errori: {errors}")
