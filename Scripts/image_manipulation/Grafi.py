# Librerie per i grafici matplotlib.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.style.use('_mpl-gallery') # Imposta uno stile grafico di matplotlib.

# Dizionario: ID dataset → (nome dataset, colore del grafico)
dataset_map = {
    1: ("rf1.tensorflow", "royalblue"),
    3: ("rf3.tensorflow", "forestgreen"),
    4: ("annotations.csv", "pink")
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

    # Posizioni sull'asse Y (una per classe)
    y = np.arange(len(classes))

    # Base iniziale per barre orizzontali impilate
    left = np.zeros(len(classes))

    # Costruisce la matrice dei dati
    matrix = build_matrix(classes, data_dict)

    # Disegna uno stack per ogni dataset
    for ds_id, (ds_name, color) in dataset_map.items():

        values = np.array(matrix[ds_id])

        ax.barh(
            y,
            values,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=ds_name
        )

        left += values

    # Etichette asse Y
    ax.set_yticks(y)
    ax.set_yticklabels(classes)

    # Titolo
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Etichetta asse X
    ax.set_xlabel("Numero di segnali")

    # Totale alla fine di ogni barra
    for i, total in enumerate(left):
        ax.text(
            total,
            i,
            str(int(total)),
            va="center",
            ha="left"
        )

    # Margine destro
    ax.set_xlim(0, max(left) * 1.15)

    # Legenda
    handles = [
        Patch(facecolor=dataset_map[i][1], label=dataset_map[i][0])
        for i in dataset_map
    ]

    ax.legend(handles=handles, title="Dataset")

    plt.tight_layout()
    plt.show()

"""---"""

# GRAFICO 1
plot_class_graph(
    "Segnali d'obbligo",
    ["Turn Left", "Straight", "Roundabout"],
    {
        "Turn Left": [(3, 732)],
        "Straight": [(3, 732)],
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
        "No Right": [(1, 150), (3, 300)],
        "No Straight": [(3, 300)],
        "No Left": [(1, 168)]
    }
)

# GRAFICO 4
plot_class_graph(
    "Segnali di precedenza",
    ["Yield", "Stop"],
    {
        "Stop": [(1, 427), (4, 91)],
        "Yield": [(1, 321)]
    }
)

# GRAFICO 5
plot_class_graph(
    "Segnaletica luminosa",
    ["Green", "Yellow", "Red"],
    {
        "Red": [(1, 489), (3, 26)],
        "Yellow": [(1, 233), (4,1)],
        "Green": [(1, 700), (4, 15)]
    }
)

# GRAFICO 6
plot_class_graph(
    "Segnali di pericolo",
    ["Hazard", "Speed Bump", "Slope Down", "Slope Up", "Cross Walk"],
    {
        "Hazard": [(1, 507)],
        "Speed Bump": [(1, 300)],
        "Slope Down": [(3, 336)],
        "Slope Up": [(3, 378)],
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