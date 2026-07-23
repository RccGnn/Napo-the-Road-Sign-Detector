# Napo-the-Road-Sign-Detector
Repository per il progetto del corso di Fondamenti di Intelligenza Artificiale, A.A. 2025-2026

# Obiettivi del progetto / Project goals
Il progetto Napo si pone due obiettivi principali:
1. Addestrare un modello di machine learning e mettere in produzione un'applicazione in grado di classificare i segnali stradali partendo da un'immagine in input passata dall'utente;
2. Mettere a confronto le performance del modello addestrato nella fase precedente contro un dataset di immagini reali (foto di segnali stradali in diverse condizioni, diversi livelli di esposizione, diverse angolazioni delle foto...).

# Struttura (semplificata) della repository / Repository (simplified) structure
├── Application           # applicativo\
├── Dataset/              # dataset (.zip) e model training scripts\
├── Scripts/\
│   └── data/                   # scripts about data\
│   └── image_manipulation      # scripts about parsing the dataset\
│   └── utility                 # general purpose scripts\
└── README.md\

# Dipendenze / Dependencies
Il progetto richiede Python 3.12 ed i seguenti moduli:
* pandas
* pillow
* numpy
* tqdm
* matplotlib
* torch
* torchvision
* scikit-learn
* streamlit
* seaborn

Di seguito sono riportate le dipendenze di ogni singolo pacchetto.
| Dipendenze | Versione |
| --- | --- |
| altair | 6.2.2 |
| anyio | 4.14.2 |
| attrs | 26.1.0 |
| blinker | 1.9.0 |
| certifi | 2026.7.22 |
| charset-normalizer | 3.4.9 |
| click | 8.4.2 |
| colorama | 0.4.6 |
| contourpy | 1.3.3 |
| cycler | 0.12.1 |
| filelock | 3.32.0 |
| fonttools | 4.63.0 |
| fsspec | 2026.6.0 |
| gitdb | 4.0.12 |
| GitPython | 3.1.55 |
| h11 | 0.16.0 |
| httptools | 0.8.0 |
| idna | 3.18 |
| itsdangerous | 2.2.0 |
| Jinja2 | 3.1.6 |
| joblib | 1.5.3 |
| jsonschema | 4.26.0 |
| jsonschema-specifications | 2025.9.1 |
| kiwisolver | 1.5.0 |
| MarkupSafe | 3.0.3 |
| matplotlib | 3.11.1 |
| mpmath | 1.3.0 |
| narwhals | 2.24.0 |
| networkx | 3.6.1 |
| numpy | 2.5.1 |
| packaging | 26.2 |
| pandas | 3.0.3 |
| pandas-stubs | 3.0.3.260530 |
| pillow | 12.3.0 |
| protobuf | 7.35.1 |
| pyarrow | 24.0.0 |
| pydeck | 0.9.3 |
| pyparsing | 3.3.2 |
| python-dateutil | 2.9.0.post0 |
| python-multipart | 0.0.32 |
| referencing | 0.37.0 |
| requests | 2.34.2 |
| rpds-py | 2026.6.3 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.0 |
| seaborn | 0.13.2 |
| setuptools | 83.0.0 |
| six | 1.17.0 |
| smmap | 5.0.3 |
| starlette | 1.3.1 |
| streamlit | 1.60.0 |
| streamlit-option-menu | 0.4.0 |
| sympy | 1.14.0 |
| tenacity | 9.1.4 |
| threadpoolctl | 3.6.0 |
| toml | 0.10.2 |
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| tqdm | 4.69.0 |
| typing_extensions | 4.16.0 |
| tzdata | 2026.3 |
| urllib3 | 2.7.0 |
| uvicorn | 0.51.0 |
| watchdog | 6.0.0 |
| websockets | 16.1.1 |

# Istruzioni per la riproduzione / Imitations steps
Di seguito sono riportati i passaggi necessari per replicare i risultati ottenuti dal progetto:
1. Installare le dipendenze;
2. eseguire lo script Scripts/image_manipulation/csv_manipulation.py;
3. eseguire lo script Dataset/EfficiencyNet.py ed attendere la fine dell'addestramento;
4. eseguire lo script Dataset/ConvNext.py ed attendere la fine dell'addestramento;

A questo punto:
* Per eseguire l'applicazione:\
muovendosi nella cartella Application, eseguire da shell il comando `streamlit run app.py` 

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
