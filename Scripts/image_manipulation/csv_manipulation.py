import io
import os
import shutil
import pandas as pd
import pathlib
import zipfile
from PIL import Image
import xml.etree.ElementTree as Et
import csv
from tqdm import tqdm
from pathlib import PurePosixPath

from Scripts.utilities import utility as u

"""
    Il modulo zipfile permette di esplorare un file zip come se fosse una normale cartella, con
    l'unica differenza che tutti i file e le cartelle sono listati come delle stringhe allo stesso livello 
    di profondità e dove il nome di un file è il suo path completo con il file zip alla radice .
"""

entries=[]

# Variabile GLOBALE, lista di pandas.DataFrame
csv_list_df = list[pd.DataFrame]()


def carica_entries(file):
    global entries
    pathfile = pathlib.Path(__file__).parent / file

    if not pathfile.exists():
        print(f"Errore: Il file {file} non esiste in {pathfile}")
        return []

    with open(pathfile, mode="r", encoding="utf-8") as f:
        # Usiamo reader classico perché il file contiene solo dati, senza header
        lettore_csv = csv.reader(f)

        for riga in lettore_csv:
            if not riga:
                continue  # Salta eventuali righe vuote

            # Puliamo ogni elemento da spazi bianchi extra
            riga = [elemento.strip() for elemento in riga]

            # Creiamo il dizionario base con tutti i campi a None
            dizionario_generico = {
                "filename": None, "class": None,
                "xmin": None, "ymin": None, "xmax": None, "ymax": None,
                "width": None, "height": None, "depth": None
            }

            # Assegniamo i dati in base a quanti elementi sono scritti nella riga
            lunghezza = len(riga)

            if lunghezza >= 6:
                dizionario_generico["filename"] = riga[0]
                dizionario_generico["class"] = riga[1]
                dizionario_generico["xmin"] = int(float(riga[2]))
                dizionario_generico["ymin"] = int(float(riga[3]))
                dizionario_generico["xmax"] = int(float(riga[4]))
                dizionario_generico["ymax"] = int(float(riga[5]))

            # Se il file è più lungo e ha anche width, height e depth (quindi almeno 9 colonne)
            if lunghezza >= 9:
                dizionario_generico["width"] = int(float(riga[6]))
                dizionario_generico["height"] = int(float(riga[7]))
                dizionario_generico["depth"] = float(riga[8])

            entries.append(dizionario_generico)

    print(f"Caricate con successo {len(entries)} entries dal file {file} di testo.")


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
    global csv_list_df

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
    csv_list_df.append(df)


def image_preprocessing_csv(
        merged_df: pd.DataFrame,
        output_folder: str = "preprocessed_images",
        delete_previous: bool = False,
        max_iter: int = -1,
        verbose: bool = False,
) -> list[str]:
    """
    Estrae le immagini elencate in `merged_df` dai file zip presenti nella cartella Dataset,
    le ritaglia in base ai rispettivi bounding box e le salva in `output_folder` (dentro Dataset).

    Args:
        merged_df: DataFrame con colonne 'filename', 'xmin', 'ymin', 'xmax', 'ymax'
            (rinomina gli attributi usati sotto se le tue colonne hanno nomi diversi,
            es. 'x'/'y'/'width'/'height'). Solo le immagini presenti qui vengono processate.
            Più righe con lo stesso filename indicano più bounding box (oggetti diversi)
            nella stessa immagine sorgente.
        output_folder: Cartella di destinazione dentro Dataset. Default "preprocessed_images".
        delete_previous: Se True, elimina output_folder (se esiste) prima di ricrearla.
        max_iter: Numero massimo di crop da salvare in totale (utile per test rapidi).
            Se negativo, processa tutto. Default -1.
            ATTENZIONE: se impostato, la lista dei "filename mancanti" restituita potrebbe
            non essere affidabile, perché la scansione degli zip si ferma appena si
            raggiunge il limite, prima di aver controllato tutti i file disponibili.
        verbose: Se True, stampa un messaggio per ogni immagine salvata (più lento su
            dataset grandi). Default False: viene mostrata solo una progress bar.

    Returns:
        List[str]: lista (ordinata) dei filename presenti in `merged_df` ma MAI trovati
        in nessuno dei tre zip. Lista vuota se tutti i filename del csv sono stati trovati.
        Side-effect: crea `output_folder` piena di immagini ritagliate.
    """
    initial_len = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    duplicates_removed = initial_len - len(merged_df)
    if duplicates_removed > 0:
        print(
            f"⚠️ Rimosse {duplicates_removed} righe duplicate esatte dal csv (stesso filename + stesso bounding box).")

    dataset_path = u.get_dataset_dir().resolve()
    output_folder = u.get_dataset_dir() / output_folder

    if output_folder.exists() and delete_previous:
        shutil.rmtree(output_folder)
        print(f"🧹 Eliminata la cartella: '{output_folder}'.")
    output_folder.mkdir(parents=True, exist_ok=True)

    grouped = merged_df.groupby("filename")

    # Insieme dei filename ancora da trovare: appena si svuota (tutte le immagini
    # richieste sono state estratte) si possono saltare gli zip rimanenti senza
    # nemmeno aprirli — utile se il csv copre solo una parte dei file negli zip
    # (es. dataset diviso in più archivi train/val/test).
    remaining = set(grouped.groups.keys())

    total_target = len(merged_df) if max_iter < 0 else min(max_iter, len(merged_df))
    counter = 0

    zip_paths = sorted(dataset_path.glob("*.zip"))  # ordine deterministico

    with tqdm(total=total_target, desc="Ritaglio immagini", unit="img") as pbar:
        for zip_path in zip_paths:
            if not remaining or (0 <= max_iter <= counter):
                break

            with zipfile.ZipFile(zip_path, "r") as archive:
                namelist = archive.namelist()

                # Controlla UNA VOLTA per zip se esiste una cartella "train": se sì,
                # ci si limita a quella; se no, si cerca in tutto lo zip come prima.
                has_train_folder = any(
                    any(part.lower() == "train" for part in PurePosixPath(fp).parent.parts)
                    for fp in namelist
                )

                for file_path in namelist:
                    if not remaining or (0 <= max_iter <= counter):
                        break
                    if not file_path.lower().endswith((".jpeg", ".jpg", ".png")):
                        continue
                    if has_train_folder:
                        parent_parts = PurePosixPath(file_path).parent.parts
                        if not any(part.lower() == "train" for part in parent_parts):
                            continue
                    base_filename = os.path.basename(file_path)
                    if base_filename not in remaining:
                        continue

                    rows = grouped.get_group(base_filename)
                    base_name = os.path.splitext(base_filename)[0][:50]

                    with archive.open(file_path) as img_file:
                        img_data = io.BytesIO(img_file.read())
                        with Image.open(img_data) as img:
                            # itertuples è molto più veloce di iterrows: niente
                            # creazione di una Series per ogni riga/bounding box.
                            for row in rows.itertuples(index=True):
                                if 0 <= max_iter <= counter:
                                    break
                                xmin, ymin = int(row.xmin), int(row.ymin)
                                xmax, ymax = int(row.xmax), int(row.ymax)
                                cropped_img = img.crop((xmin, ymin, xmax, ymax))
                                new_filename = f"{base_name}cropped{row.Index}.png"
                                cropped_img.save(output_folder / new_filename)

                                counter += 1
                                pbar.update(1)
                                if verbose:
                                    print(f"Salvato: {output_folder / new_filename}")

                    remaining.discard(base_filename)

    print(f"Salvate {counter} immagini!")

    missing = sorted(remaining)
    if missing:
        print(
            f"⚠️ {len(missing)} filename presenti nel csv ma non trovati in nessuno "
            f"dei {len(zip_paths)} zip:"
        )
        for name in missing[:20]:
            print(f"  - {name}")
        if len(missing) > 20:
            print(f"  ... e altri {len(missing) - 20}")
    else:
        print("✅ Tutti i filename del csv sono stati trovati negli zip.")

    return missing


def find_csv_files() -> None:
    """
        Cerca il primo file CSV all'interno di un archivio ZIP e lo legge in un DataFrame,
        caricandolo tramite pandas.

        Returns:
            None: Come effetto collaterale vengono aggiunti nella lista globale csv_list_df i
            DataFrame contenente i dati letti dal file CSV.
    """

    # Recupera il percorso della cartella Dataset e trasformalo in assoluto (con '.resolve')
    dataset_path = u.get_dataset_dir().resolve()
    global csv_list_df

    # Usa .glob(".zip") per ciclare solo sui file con estensione .zip
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
                        csv_list_df.append(df)
                        break


def merge_csv_files(output_csv="merged.csv", exec_find_csv_files = False) -> pd.DataFrame:
    """
        Concatena tutti i DataFrame presenti nella lista globale 'csv_list_df' in un unico DataFrame
        e lo salva come file CSV all'interno della cartella Dataset.

        Se la lista globale `csv_list_df` è vuota, permette di scegliere se interrompere
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
    global csv_list_df

    # Controlla se la lista è vuota
    if not csv_list_df:
        print("Warning: nessun file .csv letto.")
        if exec_find_csv_files:
            print("Lettura dei file .csv in corso...")
            find_csv_files()
        else:
            return None


    merged_csv = u.get_dataset_dir() / output_csv
    # Concatena tutti i DataFrames nella lista
    merged_df = pd.concat(csv_list_df, ignore_index=True)
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
        "no right": "do_not_turn" ,
        "no straight" : "do_not_turn",
        "left" : "obligation",
        "straight": "obligation",
        "up": "slope" ,
        "down": "slope",
        "Speed Limit -100-" : "speedlimit",
        "Speed Limit -60-" : "speedlimit",
        "Speed Limit -70-" : "speedlimit",
        "Speed Limit -80-" : "speedlimit",
        "Speed Limit 30"   : "speedlimit",
        "green": "green_traffic_light",
        "red" : "red_traffic_light",
    }

    df_unificato = df.copy()
    df_unificato['class'] = df_unificato['class'].replace(dizionario_etichette)
    return df_unificato

def filter_and_replace_csv(csv_label_class, df: pd.DataFrame) -> pd.DataFrame:
    global entries

    if csv_label_class not in df['class'].values:
        print(f"Warning: la classe '{csv_label_class}' non è presente in questo DataFrame.")

    df_filtrato = df[df['class'] != csv_label_class]
    print(df_filtrato.head())
    print("dopo filtro")
    df_aggiornato = add_entries(df_filtrato, "merged.csv", "entries.txt")

    return df_aggiornato


def add_entries(df:pd.DataFrame,file_csv:str,file_entries:str) -> pd.DataFrame:
    """
    Aggiunge nuove righe ad un file CSV.

    Args:
        file_csv: percorso del file CSV oppure solo il suo nome.
        entries: lista di dizionari contenenti le nuove righe.

    Returns:
        pd.DataFrame: il DataFrame aggiornato.
    """
    global entries
    carica_entries(file_entries)

    # Nessuna entry da aggiungere
    if not entries:
        print("Nessuna entry da aggiungere.")
        return df

    # Converte le nuove righe in DataFrame
    new_df = pd.DataFrame(entries)

    # Controlla che le colonne coincidano
    if list(new_df.columns) != list(df.columns):
        raise ValueError(
            f"Le colonne delle nuove entry non coincidono con quelle del CSV.\n"
            f"CSV: {list(df.columns)}\n"
            f"Entries: {list(new_df.columns)}"
        )

    # Aggiunge le nuove righe
    df = pd.concat([df, new_df], ignore_index=True)

    # Salva il CSV aggiornato (senza indice)
    file_csv=u.get_dataset_dir() / file_csv
    df.to_csv(file_csv, index=False)

    print(f"Aggiunte {len(new_df)} nuove righe a '{file_csv}'.")
    print(df.tail())

    return df


def delete_entries_flexible(df: pd.DataFrame, colonna: str, valore: str, n_da_eliminare: int = None, file_csv: str = "merged.csv") -> pd.DataFrame:
    """
    Elimina le righe che corrispondono a un determinato valore in una colonna scelta.

    Se 'n_da_eliminare' NON viene specificato, cancella TUTTE le righe corrispondenti.
    Se 'n_da_eliminare' viene specificato, ne cancella solo quel numero a caso.

    Args:
        file_csv: Nome o percorso del file CSV (es. "merged.csv").
        colonna: La colonna su cui filtrare (es. "class", "width", "ymin").
        valore: Il valore da cercare (es. "obligation", 400).
        n_da_eliminare: (Opzionale) Quante righe cancellare a caso. Di default è None (elimina tutto).
    """
    # 1. Gestione del percorso di salvataggio su disco
    file_csv_path = pathlib.Path(file_csv)
    if not file_csv_path.is_absolute():
        file_csv_path = u.get_dataset_dir() / file_csv

    # Creiamo una copia di lavoro per sicurezza
    df_lavoro = df.copy()

    # 2. Trova gli indici di tutte le righe corrispondenti
    indici_corrispondenti = df_lavoro[df_lavoro[colonna].astype(str).str.strip() == str(valore).strip()].index
    totale_trovati = len(indici_corrispondenti)

    if totale_trovati == 0:
        print(f"Nessuna riga trovata con {colonna} = '{valore}'.")
        return df_lavoro

    # 3. LOGICA DI ELIMINAZIONE
    if n_da_eliminare is None:
        # CASO 1: Elimina TUTTE le righe
        df_aggiornato = df_lavoro.drop(index=indici_corrispondenti).reset_index(drop=True)
        print(f"Target '{colonna}' = '{valore}': Eliminate TUTTE le {totale_trovati} righe trovate.")
    else:
        # CASO 2: Elimina solo un TOT di righe in modo RANDOM
        quantita_effettiva = min(n_da_eliminare, totale_trovati)

        indici_da_eliminare = pd.Series(indici_corrispondenti).sample(
            n=quantita_effettiva,
            random_state=None
        ).values

        df_aggiornato = df_lavoro.drop(index=indici_da_eliminare).reset_index(drop=True)
        print(
            f"Target '{colonna}' = '{valore}': Trovate {totale_trovati} righe. Cancellate {quantita_effettiva} a caso! ")

    # 4. Salva il file CSV aggiornato su disco e restituisce il DataFrame modificato
    df_aggiornato.to_csv(file_csv_path, index=False)
    return df_aggiornato

# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    m = merge_csv_files("merged.csv", True)

    # 2. Filtra e aggiorna 'm' in memoria in ogni passaggio
    m = filter_and_replace_csv("trafficlight", m)

    m = delete_entries_flexible(m, colonna="class", valore="speedlimit", n_da_eliminare=500)
    m = delete_entries_flexible(m, colonna="class", valore="green_traffic_light", n_da_eliminare=1100)
    m = delete_entries_flexible(m, colonna="class", valore="obligation", n_da_eliminare=500)
    m = delete_entries_flexible(m, colonna="class", valore="do_not_turn", n_da_eliminare=600)
    m = delete_entries_flexible(m, colonna="class", valore="slope", n_da_eliminare=600)

    # Controlli di verifica direttamente su 'm'
    print("Filename unici:", m["filename"].nunique())
    print("Righe totali (bounding box):", len(m))

    # 3. Passa lo stesso identico 'm' aggiornato al preprocessing delle immagini
    image_preprocessing_csv(m)