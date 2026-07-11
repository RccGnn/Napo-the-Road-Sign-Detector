from Scripts import *
from Dataset import EfficiencyNet,ConvNext
from Scripts.image_manipulation.csv_manipulation import *

m = merge_csv_files("merged.csv", True)

# aggiorna 'm' in memoria in ogni passaggio
m = filter_and_replace_csv("trafficlight", m)

m = delete_entries_flexible(m, colonna="class", valore="speedlimit", n_da_eliminare=500)
m = delete_entries_flexible(m, colonna="class", valore="green_traffic_light", n_da_eliminare=1100)
m = delete_entries_flexible(m, colonna="class", valore="obligation", n_da_eliminare=500)
m = delete_entries_flexible(m, colonna="class", valore="do_not_turn", n_da_eliminare=600)
m = delete_entries_flexible(m, colonna="class", valore="slope", n_da_eliminare=600)

print("Filename unici:", m["filename"].nunique())
print("Righe totali (bounding box):", len(m))

#  Passa lo stesso 'm' aggiornato al preprocessing delle immagini
image_preprocessing_csv(m)

print("\n")
# Ordina le immagini in cartelle per classe
img = u.get_dataset_dir() / "preprocessed_images"
csv = u.get_dataset_dir() / "preprocessed_images.csv"
u.organize_images_by_class(img, csv, "nome", "classe")
u.esegui_split_dataset()