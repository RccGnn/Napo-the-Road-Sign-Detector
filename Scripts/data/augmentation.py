import random
import pandas as pd
from PIL import Image, ImageEnhance
import numpy as np
import csv

from Scripts.utilities import utility as u
from Scripts.image_manipulation import csv_manipulation as im

def add_gaussian_noise(image: Image.Image, sigma: float = 25) -> Image.Image:
    """
    Il 'rumore Gaussiano' è un modo di aggiungere rumore in modo uniforme (distribuito in modo uniforme su tutta l'immagine).

    Per generare un'immagine con rumore Gaussiano, la funzione converte l'immagine in un vettore di pixel, genera un vettore
    di ugual dimensione di 'rumore' normalmente distribuito, lo aggiunge al vettore immagine, esegue il modulo in base 255
    e infine, riconverte il vettore di pixel (rumoroso) in immagine.

    Args:
        image: Immagine PIL di input.
        sigma: Deviziazione standard; esprime quanto i valori della distribuzione sono 'lontani' dalla media.

            Nello specifico indica quanto un pixel è rumoroso rispetto al corrispettivo pixel della foto originiaria
            (i pixel rumorosi sono i valori della distribuzione ed i pixel della foto rappresentano la media).

            Valori tipici: 5 (rumore lieve, appena percettibile), 25 (rumore chiaramente visibile, DEFAULT),
            50 (rumore estremo).

    Returns:
        Immagine PIL a cui è stato applicato il rumore, aperta.
    """
    # Salva il formato originale dell'immagine
    original_mode = image.mode
    # Converti l'immagine sul canale RGB per evitare di toccare il canale Alpha
    rgb_image = image.convert("RGB")

    # Dall'immagine in pixel
    img_array = np.array(rgb_image, dtype=np.float32)
    # Crea e aggiungi il rumore
    noise = np.random.normal(loc=0, scale=sigma, size=img_array.shape)
    noisy_array = img_array + noise

    # Modulo 255
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8) # UnsignINTeger8bit (each pixel is a byte)
    # Ri-aggiungi il canale RGB
    noisy_image = Image.fromarray(noisy_array, mode="RGB")

    # Ri-converti col formato originale
    if original_mode != "RGB":
        noisy_image = noisy_image.convert(original_mode)

    return noisy_image


def augment_images(
        angle: float | int = 15,
        brightness_range: tuple[float | int, float | int] = (0.5, 1.5),
        modifiers: tuple[bool, bool, bool, bool, bool] = (True, True, True, True, True),
        select_classes: dict[str, int] | None = None,
) -> None:
    """
        Applica una serie di mutazioni alle immagini pre-processate, permettendo di scegliere quali mutazioni attivare e
        scegliendo su quali classi fare augmentation e di quante immagini aumentate creare per ciascuna.
        Si può anche scegliere di fare augmentation su tutte le immagini della cartella 'preprocessed_images'.

        Le immagini vengono lette dalla cartella 'Dataset/preprocessed_images' all'interno di 'Dataset' e le nuove
         versioni vengono salvate in 'augmented_images' nella stessa cartella padre con un file csv aggiornato
         con le informazioni delle nuove immagini.

        Args:
            angle (float|int, opzionale): I gradi di rotazione per l'inclinazione
                in senso orario e antiorario. Di default è 15.
            brightness_range (tuple[float|int, float|int], opzionale): Un intervallo (min, max)
                da cui estrarre casualmente il fattore di luminosità. La luminosità originale
                di una foto è 1; quindi per x < 1, l'immagine diventa più scura mentre per x > 1,
                 l'immagine diventa più luminosa. Di default è (0.5, 1.5).
            modifiers (tuple[bool, bool, bool, bool, bool], opzionale): Tupla che abilita o
                disabilita le 5 mutazioni nell'ordine: (Rotazione oraria, Rotazione antioraria,
                Ribaltamento orizzontale, Luminosità, Rumore gaussiano). Di default sono tutte True.
            select_classes (dict[str, int] | None, opzionale): Dizionario {nome_classe: quantità} che
                controlla su quali classi fare augmentation e quante immagini aumentate creare per ognuna.
                - La chiave è il nome della classe.
                - Il valore è il NUMERO di immagini aumentate da creare per quella classe (non il numero
                  di immagini originali da usare). Quando la quota di una classe viene raggiunta, la classe
                  viene ignorata per il resto dell'esecuzione.
                Se None o {} (vuoto), la funzione fa augmentation su TUTTE le classi presenti nella cartella
                'Dataset/preprocessed_images' senza limiti. Di default è None.
                Nota: se le immagini originali di una classe non bastano a raggiungere la quota richiesta,
                vengono create tutte quelle possibili e viene stampato un warning.

        Returns:
            None: Come effetto collaterale, la cartella di destinazione in 'Dataset' viene riempita di nuove
             immagini modificate con un file csv aggiornato con le informazioni delle nuove immagini.
    """

    dataset_dir = u.get_dataset_dir()
    images_path = dataset_dir / "preprocessed_images"

    # Augmentation su tutte le classi se il dizionario passato è vuoto o ha lunghezza 0, senza limiti
    if select_classes is None or len(select_classes) == 0:
        select_classes = {}

    # Leggo il csv delle immagini 'merged.csv'
    df, csv_file = pd.DataFrame(), ""
    for file_path in dataset_dir.iterdir():
        if "merged.csv" in str(file_path):
            csv_file = file_path.name
            df = pd.read_csv(file_path)
            break

    # Non è stato letto nessun file .csv
    if df.empty:
        print("❌ Error: Nessun file CSV trovato (o vuoto) nella cartella preprocessed_images!")
        return None

    entries_file_path = images_path / "new_entries.txt"
    total_new = 0
    # Quante immagini aumentate sono state create finora per ciascuna classe
    created_per_class: dict[str, int] = {}

    # Funzione ausiliaria definita solo nello scope della fnzione
    def quota_raggiunta(class_name: str) -> bool:
        """
        Funzione che calcola se la quota di immagini modificate è stata raggiunta per una determinata classe.
        Args:
            class_name (nome): nome della classe da calcolare.
        Returns:
            True se per 'class_name' è stata richiesta una quota ed è già stata raggiunta.
            Se select_classes è vuoto (augmentation su tutto), non c'è mai una quota da raggiungere.
            """
        if not select_classes:
            return False
        return created_per_class.get(class_name, 0) >= select_classes.get(class_name, 0)

    keywords = ("cropped", "angled_cw", "angled_ccw", "mirrored", "brightened", "noisy")
    counter = 0
    with open(entries_file_path, mode="w", newline="", encoding="utf-8") as entries_out:
        writer = csv.writer(entries_out)
        # Itera tutti i file nella cartella delle immagini
        for file_path in images_path.iterdir():

            # Salta i file che non sono immagini o che sono immagini già modificate
            if (file_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp") and
                    not any(keyword in str(file_path) for keyword in keywords)):
                continue

            # Ricava la riga corrispondente all'immagine considerata
            cropped_file_name = file_path.name.split("cropped")[0]
            row_info = df[df['filename'].str.contains(cropped_file_name, na=False, regex=False)]
            # Se non si trova l'entrata, scarta
            print("I")
            if row_info.empty:
                continue

            # Estrai i dati dell'immagine originale non modificata
            info = row_info.iloc[0].to_dict()
            orig_class = str(info.get('class'))

            # --- FILTRO PER CLASSE ---
            # Se è stata passata una selezione e questa classe non è richiesta, salta l'immagine
            if select_classes and orig_class not in select_classes:
                continue
            # Se la quota di questa classe è già stata raggiunta, salta l'immagine
            if quota_raggiunta(orig_class):
                continue

            # Ricavo i dati per facilità di manipolazione
            xmin, ymin = info.get('xmin'), info.get('ymin')
            xmax, ymax = info.get('xmax'), info.get('ymax')
            width, height = info.get('width'), info.get('height')
            depth = info.get('depth', 3.0)  # default a 3 se non presente

            try:
                with Image.open(file_path) as img:
                    # Nome dell'immagine senza path e senza estensione
                    base_name = file_path.stem

                    # Funzione ausiliaria definita solo nello scope di funzione per salvare le immagini
                    def salva(out_img, suffix, xmin, ymin, xmax, ymax, width, height) -> None:
                        """Salva l'immagine modificata, registra la riga nel csv e aggiorna il contatore di classe."""
                        nonlocal total_new # Guarda lo scope direttamente esterno
                        temp = images_path / f"{base_name}_{suffix}.png"
                        out_img.save(temp)
                        print(temp)
                        writer.writerow([temp.name, orig_class, xmin, ymin, xmax, ymax, width, height, depth])
                        created_per_class[orig_class] = created_per_class.get(orig_class, 0) + 1
                        total_new += 1

                    # Ogni mutazione viene applicata solo se attiva & se la quota della classe non è ancora piena.
                    # In questo modo l'ultima immagine di una classe può fermarsi a metà, appena raggiunta la quota.

                    if modifiers[0] and not quota_raggiunta(orig_class):
                        # 1. Inclina in senso orario (expand=True evita il taglio dei bordi)
                        out = img.rotate(-angle, expand=True)
                        salva(out, "angled_cw", xmin, ymin, xmax, ymax, out.size[0], out.size[1])

                    if modifiers[1] and not quota_raggiunta(orig_class):
                        # 2. Inclina in senso antiorario
                        out = img.rotate(angle, expand=True)
                        salva(out, "angled_ccw", xmin, ymin, xmax, ymax, out.size[0], out.size[1])

                    if modifiers[2] and not quota_raggiunta(orig_class):
                        # 3. Specchia orizzontalmente; xmin/xmax si invertono rispetto alla larghezza (width)
                        out = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        salva(out, "mirrored", width - xmax, ymin, width - xmin, ymax, width, height)

                    if modifiers[3] and not quota_raggiunta(orig_class):
                        # 4. Modifica la luminosità entro il range (evita il fattore neutro 1)
                        brightness_factor = 1
                        while brightness_factor == 1:
                            brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
                        out = ImageEnhance.Brightness(img).enhance(brightness_factor)
                        salva(out, f"brightened_{brightness_factor:.2f}", xmin, ymin, xmax, ymax, width, height)

                    if modifiers[4] and not quota_raggiunta(orig_class):
                        # 5. Aggiunge rumore gaussiano
                        out = add_gaussian_noise(img)
                        salva(out, "noisy", xmin, ymin, xmax, ymax, width, height)

                    counter += 1 # Stampa per mostrare il progresso
                    if counter % 100 == 0:
                        print(f"Salvate {counter} immagini! ")
            except Exception as e:
                print(f"❌ Errore: Modifiche all'immagine '{file_path.name}' non riuscite : {e}")

    # Se è stata salvata almeno una riga
    if total_new > 0:
        df = im.add_entries(df, csv_file, str(entries_file_path))
        print(f"📊 File CSV aggiornato con {total_new} nuove righe")

    print(f"Create {total_new} nuove immagini a partire da {counter} originali!")

    # Avvisa se qualche classe richiesta non ha raggiunto la quota (immagini originali insufficienti)
    for class_name, requested in select_classes.items():
        created = created_per_class.get(class_name, 0)
        if created < requested:
            print(f" Warning: classe '{class_name}' -> create solo {created} su {requested} richieste "
                  f"immagini (immagini originali insufficienti).")

    return None


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":

    dict = {
        "speedlimit":200,
        "green_traffic_light":100
    }
    augment_images(select_classes=dict)
