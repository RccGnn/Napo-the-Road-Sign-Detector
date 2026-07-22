# Napo-the-Road-Sign-Detector
Repository per il progetto del corso di Fondamenti di Intelligenza Artificiale, A.A. 2025-2026

# Obiettivi del progetto / Project goals
Il progetto Napo si pone due obiettivi principali:
1. Addestrare un modello di machine learning e mettere in produzione un'applicazione in grado di classificare i segnali stradali partendo da un'immagine in input passata dall'utente;
2. Mettere a confronto le performance del modello addestrato nella fase precedente contro un dataset di immagini reali (foto di segnali stradali in diverse condizioni, diversi livelli di esposizione, diverse angolazioni delle foto...).

# Struttura (semplificata) della repository / Repository (simplified) structure
.\
├── Application           # applicativo\
├── Dataset/              # dataset (.zip) e model training scripts\
├── Scripts/\
│   └── data/                   # scripts about data\
│   └── image_manipulation      # scripts about parsing the dataset\
│   └── utility                 # general purpose scripts\
└── README.md\

# Dipendenze / Dependencies
Il progetto richiede Python 3.12 ed i seguenti moduli:
pandas
pillow
numpy
tqdm
matplotlib
torch
torchvision
scikit-learn
streamlit

# Istruzioni per la riproduzione / Imitations steps
Di seguito sono riportati i passaggi necessari per replicare i risultati ottenuti dal progetto:
1. Installare le dipendenze;
2. eseguire lo script Scripts/image_manipulation/csv_manipulation.py;
3. eseguire lo script Dataset/EfficiencyNet.py ed attendere la fine dell'addestramento;
4. eseguire lo script Dataset/ConvNext.py ed attendere la fine dell'addestramento;

A questo punto:
* Per eseguire l'applicazione:\
muovendosi nella cartella Application, eseguire da shell il comando 

* Per visualizzare le metriche sul dataset reale:\
eseguire lo script Dataset/RealDatasetTest.py

# Dataset Credits & Attribution

This project — Napo the Road Sign Detector — was trained using image data sourced from
Roboflow Universe, an open repository of computer vision
datasets hosted by Roboflow and Kaggle. The datasets below are credited in
accordance with their respective licenses.


## 1. Traffic Sign Computer Vision Dataset — by PRASHANT


**Source**: Roboflow Universe — traffic-sign [(Roboflow Dataset Page)](https://universe.roboflow.com/prashant-qp3sw/traffic-sign-yh4bz)

**Author**: PRASHANT [(Roboflow Author Page)](https://universe.roboflow.com/prashant-qp3sw)

**License**: CC BY 4.0




## 2. Road Sign Computer Vision Model (v3) — by Big Diggers


**Source**: Roboflow Universe — road-sign [(Roboflow Dataset Page)](https://universe.roboflow.com/big-diggers/road-sign-mcfu1)

**Author**: Big Diggers [(Roboflow Author Page)](https://universe.roboflow.com/big-diggers)

**License**: CC BY 4.0


## 3. Road Sign Detection - by Larxel


**Source**: Kaggle - road-sign-detection [(Kaggle Dataset Page)](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) 

**Author**: Larxel [(Kaggle Author Page)](https://universe.roboflow.com/big-diggers)

**License**: CC0: Public Domain
