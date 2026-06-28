# Librerie per i grafici matplotlib.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.style.use('_mpl-gallery') # Imposta uno stile grafico di matplotlib.

# Dizionario: ID dataset → (nome dataset, colore del grafico)
dataset_map = {
    1: ("rf1.tensorflow", "royalblue"),
    2: ("rf2.tensorflow", "darkorange"),
    3: ("rf3.tensorflow", "forestgreen"),
    4: ("annotations.csv", "crimson")
}

"""---"""

def build_matrix(classes, data_dict):
    #Trasforma i dati in una matrice: righe = dataset, colonne = classi
    matrix = {i: [0]*len(classes) for i in dataset_map}

    for j, cls in enumerate(classes):
        for ds, val in data_dict[cls]:
            matrix[ds][j] += val

    return matrix

"""---"""

def plot_class_graph(title, classes, data_dict):

    # Crea figura e asse del grafico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Posizioni sull'asse X (una per classe)
    x = np.arange(len(classes))

    # Base iniziale per grafico a barre impilate
    bottom = np.zeros(len(classes))

    # Costruisce la matrice dei dati
    matrix = build_matrix(classes, data_dict)

    # Per ogni dataset (stack delle barre)
    for ds_id, (ds_name, color) in dataset_map.items():

        # Prende i valori del dataset corrente
        values = np.array(matrix[ds_id])

        # Disegna le barre (stacked bar chart)
        ax.bar(
            x,                 # posizione X
            values,            # altezza barre
            bottom=bottom,     # base (per impilare)
            color=color,       # colore barre
            edgecolor="white", # bordo bianco
            linewidth=0.8,     # spessore bordo
            label=ds_name      # nome legenda
        )

        # Aggiorna la base per il prossimo stack
        bottom += values

    # Imposta le etichette sull’asse X
    ax.set_xticks(x)

    # Mostra i nomi delle classi ruotati per leggibilità
    ax.set_xticklabels(classes, rotation=25, ha="right")

    # Titolo del grafico
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Etichetta asse Y
    ax.set_ylabel("Numero di segnali")

    # Scrive il totale sopra ogni barra
    for i, total in enumerate(bottom):
        ax.text(i, total, str(int(total)), ha="center", va="bottom")

    ax.set_ylim(0, max(bottom) * 1.15)  # Più margine superiore del grafo (asse y, ordinate).
    # ax.set_ylim(0, 10000) # Visualizzazione con una scala fissa.

    # Crea legenda manuale con colori corretti
    handles = [
        Patch(facecolor=dataset_map[i][1], label=dataset_map[i][0])
        for i in dataset_map
    ]

    # Mostra legenda sul grafico
    ax.legend(handles=handles, title="Dataset")

    # Sistema automaticamente gli spazi
    plt.tight_layout()

    # Mostra il grafico
    plt.show()

"""---"""

# GRAFICO 1
plot_class_graph(
    "Segnali d'obbligo",
    ["Turn Left", "Turn Right", "Straight", "Roundabout"],
    {
        "Turn Left": [(1, 6132), (2, 732)],
        "Turn Right": [(2, 6190)],
        "Straight": [(2, 6132), (3, 732)],
        "Roundabout": [(1, 327)]
    }
)

# GRAFICO 2
plot_class_graph(
    "Segnali di velocità",
    ["Non classificati", "Speed 30", "Speed 60", "Speed 70", "Speed 80", "Speed 100"],
    {
        "Non classificati": [(4, 783)],
        "Speed 30": [(1, 204)],
        "Speed 60": [(1, 264)],
        "Speed 70": [(1, 219)],
        "Speed 80": [(1, 180)],
        "Speed 100": [(1, 132)]
    }
)

# GRAFICO 3
plot_class_graph(
    "Segnali di divieto",
    ["No Entry", "No Right", "No Straight", "No Left"],
    {
        "No Entry": [(1, 724)],
        "No Right": [(1, 150), (3, 696)],
        "No Straight": [(3, 768)],
        "No Left": [(1, 168)]
    }
)

# GRAFICO 4
plot_class_graph(
    "Segnali di precedenza",
    ["Yield", "Stop"],
    {
        "Stop": [(1, 427), (2, 3054), (4, 91)],
        "Yield": [(1, 321), (2, 8648)]
    }
)

# GRAFICO 5
plot_class_graph(
    "Segnaletica luminosa",
    ["Non classificati", "Green", "Yellow", "Red"],
    {
        "Non classificati": [(4, 170)],
        "Red": [(1, 489), (3, 714)],
        "Yellow": [(1, 233)],
        "Green": [(1, 1741), (3, 678)]
    }
)

# GRAFICO 6
plot_class_graph(
    "Segnali di pericolo",
    ["Hazard", "Workers Ahead", "Speed Bump", "Slope", "Cross Walk"],
    {
        "Hazard": [(1, 507)],
        "Workers Ahead": [(2, 646)],
        "Speed Bump": [(1, 300)],
        "Slope": [(3, 1428)],
        "Cross Walk": [(4, 200)]
    }
)

# GRAFICO 7
plot_class_graph(
    "Segnali di indicazione",
    ["Parking"],
    {
        "Parking": [(3, 678)]
    }
)