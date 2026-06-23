import io
import os
import shutil
import xml.etree.ElementTree as Et
"""
    Il modulo zipfile permette di esplorare un file zip come se fosse una normale cartella, con
    l'unica differenza che tutti i file e le cartelle sono listati come delle stringhe allo stesso livello 
    di profondità e dove il nome di un file è il suo path completo con il file zip alla radice .
"""
import zipfile
from PIL import Image

def image_preprocessing(zip_path: str, output_folder="pre-processed_images", delete_previous=False, max_iter=-1) -> None:
    """
        Estrae immagini da un archivio ZIP, le ritaglia in base ai bounding box definiti nei file XML
        (standard PASCAL VOC) e le salva in una cartella.

        Args:
            zip_path (str): Il percorso o il nome del file ZIP da esplorare (es. "Road_Sign.v3i.voc.zip").
            output_folder (str, opzionale): Il percorso della cartella in cui verranno salvate le immagini
                ritagliate. Di default è "pre-processed_images".
            delete_previous (bool, opzionale): Se impostato a True, elimina la cartella `output_folder`
                (se già esistente) e tutto il suo contenuto. Di default è False.
            max_iter (int, opzionale): Il numero massimo di file da processare (principalmente per il testing)
                Se impostato ad un numero negativo, processa tutti i file. Di default è -1.

        Returns:
            None: La funzione non restituisce alcun valore. Ha come side-effect la creazione di una cartella d'immagini.
    """

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

            print(max_iter)
            # Scarta le cartelle (terminano con '/')
            if file_path.endswith("/"):
                continue

            # Trova tutti i file JPEG o JPG che sono in test, train o validation
            t = file_path.lower()
            if t.endswith((".jpeg", ".jpg")) and ("test" in t or "valid" in t or "train" in t):

                # Separa la radice dall'estenzione
                # i.e.: 'test/folder1/img.jpg' -> ('test/folder1/img', '.jpg')
                base_path, _ = os.path.splitext(file_path)
                xml_path = f"{base_path}.xml"

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

image_preprocessing("Road_Sign.v3i.voc.zip", max_iter=5)