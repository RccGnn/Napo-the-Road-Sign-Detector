import os
import pathlib
import shutil
import zipfile
import pandas as pd


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

    all_files = [f for f in images_dir.iterdir() if f.is_file()]

    moved, missing, errors = 0, 0, 0

    # Per ogni riga del csv...
    for ignore, row in df.iterrows(): # ignore viene usato come buffer per il primo parametro di iterrows, che non server
        filename = str(row[filename_column])
        class_name = str(row[class_column])

        # Nome senza estensione, es. "road0.png" -> "road0"
        radice = pathlib.Path(filename).stem
        # Stesse operazioni fatte quando i file vengono salvati per la prima volta nella cartella dal csv
        if "cropped" in radice:
            radice = radice.split("cropped")[0]
        radice = radice[:50]

        # Trova tutti i file che iniziano con la stessa desinenza, indipendentemente dall'estensione
        matches = [f for f in all_files if f.name.startswith(radice)]

        if not matches:
            print(f"Warning: nessun file trovato per '{filename}' (radice='{radice}')")
            missing += 1
            continue

        for src in matches:
            dst = images_dir / class_name / src.name  # usa il nome REALE del file trovato
            try:
                shutil.move(str(src), str(dst))
                moved += 1
            except Exception as e:
                print(f"Error: file spostato {src} -> {dst}: {e}")
                errors += 1

    print(f"\n Spostati: {moved} file | Mancanti: {missing} file | Errori: {errors}")

if __name__ == "__main__":
    img = get_dataset_dir() / "preprocessed_images"
    csv = get_dataset_dir() / "merged.csv"
    organize_images_by_class(img, csv, "filename", "class")