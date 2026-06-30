import os
import pathlib
import zipfile
import pandas as pd

from Scripts.image_manipulation.preprecessing import find_csv_files


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
            df = find_csv_files(df, archive, file_list)

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
