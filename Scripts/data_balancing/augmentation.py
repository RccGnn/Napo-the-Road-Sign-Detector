import random
from PIL import Image, ImageEnhance
import numpy as np

from Scripts.utilities import utility as u

def add_gaussian_noise(image: Image.Image, sigma: float = 25) -> Image.Image:
    """
    Il 'rumore Gaussiano' è un modo di aggiungere rumore in modo uniforme (distribuito in modo uniforme su tutta l'immagine).

    Per generare un'immagine con rumore Gaussiano, la funzione converte l'immagine in un vettore di pixel, genera un vettore
    di ugual dimensione di 'rumore' normalmente distribuito, lo aggiunge al vettore immagine, esegue il modulo in base 255
    e infine, riconverte il vettore di pixel (rumoroso) in immagine.

    Args:
        image:  Immagine PIL di input.
        sigma:  Deviziazione standard; esprime quanto i valori della distribuzione sono 'lontani' dalla media.

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

def augment_images(angle=15, brightness_range=(0.5, 1.5), modifiers=(True, True, True, True, True)) -> None:
    """
        Applica una serie di mutazioni alle immagini pre-processate,
        permettendo di scegliere quali attivare.

        Le immagini vengono lette dalla cartella 'preprocessed_images' all'interno
        di 'Dataset' e le nuove versioni vengono salvate in 'augmented_images' nella stessa cartella padre.

        Args:
            angle (int | float, opzionale): I gradi di rotazione per l'inclinazione
                in senso orario e antiorario. Di default è 15.
            brightness_range (tuple[float, float], opzionale): Un intervallo (min, max)
                da cui estrarre casualmente il fattore di luminosità. La luminosità originale
                di una foto è 1; quindi per x < 1, l'immagine diventa più scura mentre per x > 1,
                 l'immagine diventa più luminosa. Di default è (0.5, 1.5).
            modifiers (tuple[bool, bool, bool, bool], opzionale): Tupla che abilita o
                disabilita le 5 mutazioni nell'ordine: (Rotazione oraria, Rotazione antioraria,
                Ribaltamento orizzontale, Luminosità, Rumore gaussiano). Di default sono tutte True.

        Returns:
            None: Come effetto collaterale la cartella di destinazione in 'Dataset'
            viene riempita di nuove immagini modificate.
    """
    # Inutile modificare la luminosità se il range non esiste
    if brightness_range[1] == 1 and brightness_range[0] == 1:
        modifiers[3] = False

    dataset_dir = u.get_dataset_dir()
    input_path = dataset_dir / "preprocessed_images"

    output_path = dataset_dir / "augmented_images"
    # Crea cartella di output dove parents=true nel caso in cui la cartella padre ('Dataset') non sia presente
    output_path.mkdir(parents=True, exist_ok=True)

    counter = 0
    # Itera tutti i file nella cartella delle immagini
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() != ".png":
            continue

        try:
            with Image.open(file_path) as img:
                # Nome dell'immagine senza path e senza estensione
                base_name = file_path.stem

                if modifiers[0]:
                    # 1. Inclina immagine in senso orario
                    # expand=True evita che i bordi vengano tagliati a causa della rotazione
                    img_angled_cw = img.rotate(-angle, expand=True)
                    img_angled_cw.save(output_path / f"{base_name}_angled_cw.png")

                if modifiers[1]:
                    # 2. Inclina immagine in senso antiorario
                    img_angled_ccw = img.rotate(angle, expand=True)
                    img_angled_ccw.save(output_path / f"{base_name}_angled_ccw.png")

                if modifiers[2]:
                    # 3. Specchia l'immagine orizzontalmente
                    img_mirrored = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    img_mirrored.save(output_path / f"{base_name}_mirrored.png")

                if modifiers[3]:
                    # 4. Modifica la luminosità di un'immagine (Entro i limiti del range fornito)
                    # Evita il caso in cui la luminosità non viene modificata
                    brightness_factor = 1
                    while brightness_factor == 1:
                        brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
                    enhancer = ImageEnhance.Brightness(img)
                    img_brightness = enhancer.enhance(brightness_factor)
                    img_brightness.save(output_path / f"{base_name}_brightened_{brightness_factor:.2f}.png")

                if modifiers[4]:
                    # 5. Aggiunge rumore distribuito uniformemente all'immagine
                    img_noisy = add_gaussian_noise(img)
                    img_noisy.save(output_path / f"{base_name}_noisy.png")

                counter += 1
        except Exception as e:
            print(f"❌ Errore: Modifiche all'immagine '{file_path.name}' non riuscite : {e}")

    # Per ogni foto sono create più mutazioni
    print(f"Create {counter * len(modifiers)} nuove immagini!")

# ==========================================
# Test
# ==========================================
if __name__ == "__main__":

    augment_images()