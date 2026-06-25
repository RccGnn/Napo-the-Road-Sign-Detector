import io
import os
import shutil
import pandas as pd
import pathlib
"""
    Il modulo zipfile permette di esplorare un file zip come se fosse una normale cartella, con
    l'unica differenza che tutti i file e le cartelle sono listati come delle stringhe allo stesso livello 
    di profondità e dove il nome di un file è il suo path completo con il file zip alla radice .
"""
import zipfile
from PIL import Image

def get_dataset_dir() -> pathlib.Path:
    """
    Restituisce il percorso assoluto di un file nella cartella Dataset.
    SI ASSUME CHE IL PROGETTO ABBIA LA SEGUENTE STRUTTURA:
        <Napo-the-Road-Sign-Detector>/
            Dataset/
            Scripts/
                image_manipulation/
                    functions.py    <<<
    """
    # __file__ è il path assoluto dello script in esecuzione
    this_file = pathlib.Path(__file__).resolve()

    # Risali le cartelle per costruire il path della cartella 'Dataset'
    for parent in this_file.parents:
        # '/' operatore di concatenazione
        dataset_dir = parent / "Dataset"
        if dataset_dir.is_dir():
            return dataset_dir

    raise FileNotFoundError("Cartella 'Dataset' non trovata.")


def find_csv_file(df: pd.DataFrame, archive: zipfile.ZipFile, file_list: list[str]) -> pd.DataFrame:
    """
        Cerca il primo file CSV all'interno di un archivio ZIP e lo legge in un DataFrame,
        caricandolo tramite pandas.

        :param:
            df (pd.DataFrame): DataFrame di partenza (verrà sovrascritto con i dati del CSV trovato).
            archive (zipfile.ZipFile): L'oggetto archivio ZIP già aperto in modalità lettura.
            file_list (list[str]): La lista dei nomi dei file (o percorsi) presenti nell'archivio.

        :return:
            pd.DataFrame: Un nuovo DataFrame contenente i dati letti dal file CSV.
            Se non viene trovato alcun file CSV, restituisce il DataFrame `df` passato in input.
    """
    for file_path in file_list:
        # Restituisce solo il secondo valore, l'estensione
        if os.path.splitext(file_path)[1] == ".csv":
            df = pd.read_csv(archive.open(file_path))
            break
    return df


def image_preprocessing_csv(zip_path: str, output_folder="pre-processed_images", delete_previous=False, max_iter=-1) -> None:
    """
        Estrae immagini da un archivio ZIP, le ritaglia in base ai bounding box.

        :param:
            zip_path (str): Il percorso o il nome del file ZIP da esplorare (es. "Road_Sign.v3i.voc.zip").
            output_folder (str, opzionale): Il percorso della cartella in cui verranno salvate le immagini
                ritagliate. Di default è "pre-processed_images".
            delete_previous (bool, opzionale): Se impostato a True, elimina la cartella `output_folder`
                (se già esistente) e tutto il suo contenuto. Di default è False.
            max_iter (int, opzionale): Il numero massimo di file da processare (principalmente per il testing)
                Se impostato ad un numero negativo, processa tutti i file. Di default è -1.
        :return:
            None: La funzione non restituisce alcun valore. Ha come side-effect la creazione di una cartella d'immagini.
    """

    # Risolve zip_path rispetto a Dataset/ e non rispetto alla cwd
    zip_path = get_dataset_dir() / zip_path

    # Risolve anche output_folder rispetto a Dataset/ e non rispetto alla cwd
    output_folder = get_dataset_dir() / output_folder

    # Elimina la cartella di nome output_folder se esiste
    if os.path.exists(output_folder) and delete_previous:
        # shutil = os.rmdir ma elimina anche gli elementi della cartella
        shutil.rmtree(output_folder)
        print(f"🧹 Wiped old '{output_folder}' folder.")

    # Crea la cartella dove mettere le immagini modificate, se non esiste
    os.makedirs(output_folder, exist_ok=True)


    # Esplora il file zip (come se fosse una cartella normale, evitando la decompressione)
    with (zipfile.ZipFile(zip_path, "r") as archive):
        file_list = archive.namelist()

        # Recupera e apri il file .csv
        df = pd.DataFrame
        df = find_csv_file(df, archive, file_list)

        # File_path comprende tutti i file, anche cartelle o csv
        for file_path in file_list:

            # Interrompi le iterazioni se max_iter è un numero non negativo
            # Evita questo controllo solo so max_iter è un numero negativo
            if max_iter >= 0:
                if max_iter == 0:
                    return
                max_iter -= 1

            # Scarta le cartelle (terminano con '/')
            if file_path.endswith("/"):
                continue

            # Trova tutti i file JPEG o JPG o PNG
            t = file_path.lower()
            if t.endswith((".jpeg", ".jpg", ".png")):

                # Recupera tutte le righe relative ad un jpeg
                res = df.query(f"filename == '{os.path.basename(file_path)}'")

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
                            img = Image.open(img_data)

                            # Come splitext ma si prende anche il nome del file, non solo l'estensione
                            base_name = os.path.basename(file_path)

                            # Taglia l'immagine
                            cropped_img = img.crop(
                                (xmin, ymin, xmax, ymax)
                            )
                            # Salva nella cartella predefinita
                            new_filename = (
                                f"{base_name}_cropped_{index}.jpeg"
                            )
                            save_path = os.path.join(
                                output_folder, new_filename
                            )

                            cropped_img.save(save_path)
                            print(f"Saved: {save_path}")

                            img.close()


import xml.etree.ElementTree as Et

def image_preprocessing_xml(zip_path: str, output_folder="pre-processed_images", delete_previous=False, max_iter=-1) -> None:
    """
        Estrae immagini da un archivio ZIP, le ritaglia in base ai bounding box definiti nei file XML
        (standard PASCAL VOC) e le salva in una cartella.

        :param:
            zip_path (str): Il percorso o il nome del file ZIP da esplorare (es. "Road_Sign.v3i.voc.zip").
            output_folder (str, opzionale): Il percorso della cartella in cui verranno salvate le immagini
                ritagliate. Di default è "pre-processed_images".
            delete_previous (bool, opzionale): Se impostato a True, elimina la cartella `output_folder`
                (se già esistente) e tutto il suo contenuto. Di default è False.
            max_iter (int, opzionale): Il numero massimo di file da processare (principalmente per il testing)
                Se impostato ad un numero negativo, processa tutti i file. Di default è -1.

        :return:
            None: La funzione non restituisce alcun valore. Ha come side-effect la creazione di una cartella d'immagini.
    """

    # Risolve zip_path rispetto a Dataset/ e non rispetto alla cwd
    zip_path = get_dataset_dir() / zip_path

    # Risolve anche output_folder rispetto a Dataset/ e non rispetto alla cwd
    output_folder = get_dataset_dir() / output_folder

    # Elimina la cartella di nome output_folder se esiste
    if os.path.exists(output_folder) and delete_previous:
        # shutil = os.rmdir ma elimina anche gli elementi della cartella
        shutil.rmtree(output_folder)
        print(f"🧹 Wiped old '{output_folder}' folder.")

    # Crea la cartella dove mettere le immagini modificate, se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Esplora il file zip (come se fosse una cartella normale, evitando la decompressione)
    with (zipfile.ZipFile(zip_path, "r") as archive):
        file_list = archive.namelist()

        for file_path in file_list:

            # Interrompi le iterazioni se max_iter è un numero non negativo
            # Evita questo controllo solo so max_iter è un numero negativo
            if max_iter >= 0:
                if max_iter == 0:
                    return
                max_iter -= 1

            # Scarta le cartelle (terminano con '/')
            if file_path.endswith("/"):
                continue

            # Trova tutti i file JPEG o JPG o PNG
            t = file_path.lower()
            if t.endswith((".jpeg", ".jpg", ".png")):

                # Separa la radice dall'estenzione
                # i.e.: 'test/folder1/img.jpg' -> ('test/folder1/img', '.jpg')
                base_path, _ = os.path.splitext(file_path)
                xml_path = f"{base_path}.xml"
                print(xml_path)
                # Trova il relativo file XML per ogni immagine
                if xml_path in file_list:
                    try:
                        # Leggi XML (analogo os.open ma per i file che sono in cartelle zippate)
                        with archive.open(xml_path) as xml_file:
                            # Recupera l'albero degli attributi del file XML
                            tree = Et.parse(xml_file)
                            root = tree.getroot()

                        # Leggi l'immagine
                        with archive.open(file_path) as img_file:
                            img_data = io.BytesIO(img_file.read())
                            img = Image.open(img_data)

                            # Come splitext ma si prende anche il nome del file, non solo l'estensione
                            base_name = os.path.basename(base_path)

                            # Ricerca dei bounding box
                            for index, obj in enumerate(root.findall("object")): # standard PASCAL VOC
                                bndbox = obj.find("bndbox") # Cerca l'elemento "bndbox" nel root element del file XML

                                if bndbox is not None:
                                    xmin = int(float(bndbox.find("xmin").text))
                                    ymin = int(float(bndbox.find("ymin").text))
                                    xmax = int(float(bndbox.find("xmax").text))
                                    ymax = int(float(bndbox.find("ymax").text))

                                    # Taglia l'immagine
                                    cropped_img = img.crop(
                                        (xmin, ymin, xmax, ymax)
                                    )

                                    # Salva nella cartella predefinita
                                    new_filename = (
                                        f"{base_name}_cropped_{index}.jpeg"
                                    )
                                    save_path = os.path.join(
                                        output_folder, new_filename
                                    )

                                    cropped_img.save(save_path)
                                    print(f"Saved: {save_path}")

                            img.close()

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                else:
                    print(f"Warning: No matching XML found for {file_path}")

#image_preprocessing_csv("rf1.tensorflow.zip")
#image_preprocessing_xml("k1.xml.zip")

def view_csv(zip_path: str) -> None:
    """
    Visualizza il csw tramite pandas
    :param
        zip_path: Nome del dataset zippato, cioè [nome_dataset.zip]
    :return:
        None: visualizza a schermo il dataset
    """
    # Risolve zip_path rispetto a Dataset/ e non rispetto alla cwd
    zip_path = get_dataset_dir() / zip_path

    with (zipfile.ZipFile(zip_path, "r") as archive):
        file_list = archive.namelist()

        # Recupera e apri il file .csv
        df = pd.DataFrame
        df = find_csv_file(df, archive, file_list)

        try:

            # --- CONFIGURAZIONI PER IL TERMINALE ---
            # Forza Pandas a mostrare tutte le colonne senza i fastidiosi puntini di sospensione (...)
            pd.set_option('display.max_columns', None)

            # Allarga la larghezza massima del display per evitare che le righe vadano a capo
            pd.set_option('display.width', 1000)

            # (Opzionale) Mostra un numero maggiore di righe, ad esempio 50
            pd.set_option('display.max_rows', 50)

            # --- VISUALIZZAZIONE ---
            print("\n--- ANTEPRIMA DEL DATASET ---")
            # Stampa le prime 25 righe (puoi cambiare questo valore o rimuovere .head() per vedere tutto)
            print(df)
            #
            print("\n--- INFORMAZIONI SUL DATASET ---")
            print(f"Totale righe: {len(df)}")
            lista_classi = df['class'].unique()
            print(f"Classi uniche trovate: {lista_classi}")

            # Numero di elementi per classe
            i = 0
            for classi in lista_classi:
                i += len(df[(df["class"] == classi)])
                print(classi + ": " + str(len(df[(df["class"] == classi)])))

            # Totale
            print(f"Totale: {len(df)} - Calcolati: {i}\n")

        except FileNotFoundError:
            print(f"Errore: Il file '{zip_path}' non è stato trovato.")
        except Exception as e:
            print(f"Si è verificato un errore: {e}")

view_csv("rf1.tensorflow.zip")