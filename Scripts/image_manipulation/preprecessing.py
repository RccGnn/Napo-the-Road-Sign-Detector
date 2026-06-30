import io
import os
import shutil
import pandas as pd
import pathlib
import zipfile
from PIL import Image
import xml.etree.ElementTree as Et

from Scripts.utilities import utility as u

"""
    Il modulo zipfile permette di esplorare un file zip come se fosse una normale cartella, con
    l'unica differenza che tutti i file e le cartelle sono listati come delle stringhe allo stesso livello 
    di profondità e dove il nome di un file è il suo path completo con il file zip alla radice .
"""

entries=[{}]

# Variabile GLOBALE, lista di pandas.DataFrame
csv_list = list[pd.DataFrame]()

def get_dataset_dir() -> pathlib.Path:
    """
    Restituisce il percorso assoluto di un file nella cartella Dataset.
    SI ASSUME CHE IL PROGETTO ABBIA LA SEGUENTE STRUTTURA:
        <Napo-the-Road-Sign-Detector>/
            Dataset/
            Scripts/
                image_manipulation/
                    Csv_manipulation.py <<<

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


def find_csv_files() -> None:
    """
        Cerca il primo file CSV all'interno di un archivio ZIP e lo legge in un DataFrame,
        caricandolo tramite pandas.

        Returns:
            None: Come effetto collaterale vengono aggiunti nella lista globale csv_list i
            DataFrame contenente i dati letti dal file CSV.
    """

    # Recupera il percorso della cartella Dataset e trasformalo in assoluto (con '.resolve')
    dataset_path = u.get_dataset_dir().resolve()
    global csv_list

    # Usa .glob("*.zip") per ciclare solo sui file con estensione .zip
    for zip_path in dataset_path.glob("*.zip"):

        # Esplora il file zip (come se fosse una cartella normale, evitando la decompressione)
        with (zipfile.ZipFile(zip_path, "r") as archive):

            if "xml" in str(zip_path):
                # Per gli archivi di dataset che hanno annotazioni sottoforma di XML pascal
                xml_to_csv(archive)
            else:
                file_list = archive.namelist()
                for file_path in file_list:

                    # Scegli solo i file in train
                    root, ext = os.path.splitext(file_path)
                    if ext == ".csv" and "train" in root:
                        df = pd.read_csv(archive.open(file_path))
                        csv_list.append(df)
                        break


def xml_to_csv(archive: zipfile.ZipFile, output_csv="annotations.csv") -> None:
    """
    Legge tutti i file XML (PASCAL VOC) dalla cartella 'annotations/'
    di un archivio ZIP e produce un CSV compatibile con image_preprocessing_csv().

    Struttura ZIP attesa:
        annotations/road0.xml
        annotations/road1.xml
        images/road0.png
        images/road1.png

    Args:
        archive: file ZIP (es. 'dataset.zip'), cercato in Dataset/
        output_csv: Nome del CSV in output, salvato in Dataset/
    Returns:
        None: Come effetto collaterale viene creato un file .csv nella cartella /Dataset
    """

    # Crea un file .csv con il nella cartella Dataset con il nome output_csv
    resolved_csv = u.get_dataset_dir() / output_csv

    rows = []
    file_list = archive.namelist()

    # Per ogni file trovato nell'archivio, aggiungi solo gli xml
    xml_files = [
        f for f in file_list #Per ogni file f nella lista
        # Solo file .xml nella cartella "_annotation"
        if "annotations" in pathlib.Path(f).parts and f.lower().endswith(".xml")
    ]

    print(f"Trovati {len(xml_files)} file XML.")
    global csv_list

    for xml_path in xml_files:
        try:
            # Ottieni la radice del file xml
            with archive.open(xml_path) as xml_file:
                tree = Et.parse(xml_file)
                root = tree.getroot()

            # Ottieni il tag <filename>
            # i.e. annotations/road0.xml → road0
            filename_tag = root.find("filename")
            if filename_tag is not None and filename_tag.text.strip():
                filename = filename_tag.text.strip()
            else:
                # Ricava il nome dell'immagine dal nome del file XML: "annotations/road0.xml" → "road0.png"
                stem = pathlib.Path(xml_path).stem   # "road0"
                filename = stem + ".png"
                print(f"Warning: Nessun file di nome: <filename> tag in {xml_path}, inferred '{filename}'")

            # Nel tag object di xml è conservato il nome della classe a cui appartiene l'oggetto e il boundbox
            # Nel tag size di xml sono conservate le misure originali dell'immagine

            # Il tag size è univoco per ogni immagine
            size = root.find("size")

            # Invece, possono essere presenti anche più tag object (più ritagli per una foto)
            for obj in root.findall("object"):
                class_tag = obj.find("name")
                class_name = class_tag.text.strip() if class_tag is not None else ""

                bndbox = obj.find("bndbox")

                # Aggiungi alla riga del csv le informazioni ricavate
                rows.append({
                    "filename": filename,
                    "class":    class_name,
                    "xmin":     int(float(bndbox.find("xmin").text)),
                    "ymin":     int(float(bndbox.find("ymin").text)),
                    "xmax":     int(float(bndbox.find("xmax").text)),
                    "ymax": int(float(bndbox.find("ymax").text)),
                    "width": int(float(size.find("width").text)),
                    "height": int(float(size.find("height").text)),
                    "depth":    int(float(size.find("depth").text)),
                })

        except Exception as e:
            print(f"Errore - {xml_path}: {e}")

    # Imposta le colonne del csv
    df = pd.DataFrame(rows, columns=["filename", "class",  "xmin", "ymin", "xmax", "ymax", "width", "height", "depth",])
    df.to_csv(resolved_csv, index=False)
    print(f"✅ - Trovati {len(rows)} bounding boxes da {len(xml_files)} file XML → '{resolved_csv}'")
    csv_list.append(df)


def image_preprocessing_csv(output_folder="preprocessed_images", delete_previous=False, max_iter=-1) -> None:
    """
        Estrae tutte le immagini contenute in file zip nella cartella Dataset, le ritaglia in base ai bounding box
        e le memorizza in una cartella output_folder nella cartella Dataset assieme a un file csv contenente le
        informazioni di tutte le immagini ritagliate.

        Args:
            output_folder: Il percorso della cartella in cui verranno salvate le immagini
                ritagliate. Di default è "preprocessed_images".
            delete_previous: Se impostato a True, elimina la cartella `output_folder`
                (se già esistente) e tutto il suo contenuto. Di default è False.
            max_iter: Il numero massimo di file da processare (principalmente per il testing)
                Se impostato a un numero negativo, processa tutti i file. Di default è -1.
        Returns:
            None: La funzione non restituisce alcun valore. Ha come side-effect la creazione di una cartella d'immagini.
    """

    # Recupera il percorso della cartella Dataset e trasformalo in assoluto (con '.resolve')
    dataset_path = u.get_dataset_dir().resolve()

    # Recupera anche output_folder rispetto a Dataset/ e non rispetto alla cwd
    output_folder = u.get_dataset_dir() / output_folder

    # Elimina la cartella di nome output_folder se esiste e se 'delete_previous' = True
    if os.path.exists(output_folder) and delete_previous:
        # shutil = os.rmdir ma elimina anche gli elementi della cartella
        shutil.rmtree(output_folder)
        print(f"🧹 Eliminato la cartella: '{output_folder}'.")

    # Crea la cartella dove mettere le immagini modificate, se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Salva e apri i file .csv di tutti i dataset, PRIMA di iniziare a iterare sui dataset
    global csv_list
    find_csv_files()

    # Crea il file csv completo e mettilo nella cartella preprocessed_images
    merge_csv_files(output_folder / "merged.csv")

    # Se non sono stati trovati csv, termina
    if len(csv_list) == 0:
        print("Warning: nessun file csv trovato.")

    # Se ogni csv letto è vuoto, termina
    flag = False
    for df in csv_list:
        if not df.empty:
            flag = True
            break
    if not flag:
        print("Warning: nessuna annotazione trovata.")
        return

    counter = 0
    # Usa .glob("*.zip") per ciclare solo sui file con estensione .zip
    for zip_path in dataset_path.glob("*.zip"):

        # Esplora il file zip (come se fosse una cartella normale, evitando la decompressione)
        with (zipfile.ZipFile(zip_path, "r") as archive):
            file_list = archive.namelist()

            # File_path comprende tutti i file, anche cartelle o csv
            for file_path in file_list:

                # Interrompi le iterazioni se max_iter è un numero non negativo
                # Evita questo controllo solo so max_iter è un numero negativo
                if max_iter >= 0:
                    if max_iter == 0:
                        break
                    max_iter -= 1

                # Trova tutti i file JPEG o JPG o PNG
                t = file_path.lower()

                if t.endswith((".jpeg", ".jpg", ".png")):

                    # Recupera tutte le righe relative a un jpeg
                    for df in csv_list:
                        # Un'immagine appartiene ad un solo csv
                        res = df.query(f"filename == '{os.path.basename(file_path)}'")
                        if not res.empty:
                            break

                    # CROP delle immagini
                    if not res.empty:
                        # Itera su tutte le righe trovate per quella specifica immagine
                        for index, row in res.iterrows():
                            # Estrai i valori direttamente del bounding box
                            xmin = int(row["xmin"])
                            ymin = int(row["ymin"])
                            xmax = int(row["xmax"])
                            ymax = int(row["ymax"])

                            # Leggi l'immagine
                            with archive.open(file_path) as img_file:
                                img_data = io.BytesIO(img_file.read())
                                base_name = os.path.splitext(os.path.basename(file_path))[0]
                                base_name= base_name[:50]

                                with Image.open(img_data) as img:
                                    # Taglia l'immagine
                                    cropped_img = img.crop((xmin, ymin, xmax, ymax))
                                    # Salva l'immagine
                                    new_filename = f"{base_name}cropped{index}.png"
                                    save_path = os.path.join(output_folder, new_filename)
                                    cropped_img.save(save_path)
                                    print(f"Salvato: {save_path}")
                                    counter += 1

    if counter < 5000 :
        print(f"Salvate {counter} immagini! 😨")
    elif 5000 <= counter < 10000:
        print(f"Salvate {counter} immagini! 😰")
    else:
        print(f"Salvate {counter} immagini! 😱")

def merge_csv_files(output_csv="merged.csv", exec_find_csv_files = False) -> pd.DataFrame:
    """
        Concatena tutti i DataFrame presenti nella lista globale 'csv_list' in un unico DataFrame
        e lo salva come file CSV all'interno della cartella Dataset.

        Se la lista globale `csv_list` è vuota, permette di scegliere se interrompere
        l'operazione o tentare di recuperare i file automaticamente.

        Args:
            output_csv (str, opzionale): Il nome del file CSV di destinazione.
                Di default è "merged.csv".
            exec_find_csv_files (bool, opzionale): Se impostato a True e la lista è vuota,
                esegue la funzione `find_csv_files()` per tentare di popolarla.
                Se impostato a False, l'esecuzione si interrompe. Di default è False.
        Returns:
            pd.DataFrame: Il nuovo DataFrame combinato.
            Restituisce `None` se la lista globale è vuota e `exec_find_csv_files` è False.
        """
    global csv_list

    # Controlla se la lista è vuota
    if not csv_list:
        print("Warning: nessun file .csv letto.")
        if exec_find_csv_files:
            print("Lettura dei file .csv in corso...")
            find_csv_files()
        else:
            return None


    merged_csv = u.get_dataset_dir() / output_csv
    # Concatena tutti i DataFrames nella lista
    merged_df = pd.concat(csv_list, ignore_index=True)
    merged_df = merge_label(merged_df)
    # Crea il file .csv nella cartella Dataset
    merged_df.to_csv(merged_csv, index=False)
    u.view_csv(merged_csv)

    return merged_df

def merge_label( df: pd.DataFrame) -> pd.DataFrame:
    dizionario_etichette = {
        "stop": "Stop",
        "do_not_turn_l" : "do_not_turn" ,
        "do_not_turn_r" : "do_not_turn" ,
        "no straight" : "do_not_turn",
        "left" : "obligation",
        "straight": "obligation",
        "up": "slope" ,
        "down": "slop",
        "Speed Limit -100-" : "speedlimit",
        "Speed Limit -60-" : "speedlimit",
        "Speed Limit -70-" : "speedlimit",
        "Speed Limit -80-" : "speedlimit",
        "Speed Limit 30"   : "speedlimit",
    }

    df_unificato = df.copy()
    df_unificato['class'] = df_unificato['class'].replace(dizionario_etichette)
    return df_unificato


def filter_and_replace_csv(csv_label_class, df: pd.DataFrame) -> pd.DataFrame:
    global entries

    if csv_label_class not in df['class'].values:
        print(f"Warning: la classe '{csv_label_class}' non è presente in questo DataFrame.")

    df_filtrato = df[df['class'] != csv_label_class]


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    merge_csv_files(output_csv="merged.csv", exec_find_csv_files=True)
