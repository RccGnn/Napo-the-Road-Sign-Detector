import pandas as pd


def visualizza_csv_in_pycharm(percorso_file: str):
    try:
        # Carica il dataset
        df = pd.read_csv("_annotations.csv")

        # --- CONFIGURAZIONI PER IL TERMINALE DI PYCHARM ---
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

        print("\n--- INFORMAZIONI SUL DATASET ---")
        print(f"Totale righe: {len(df)}")
        lista_classi = df['class'].unique()
        print(f"Classi uniche trovate: {lista_classi}")

        # Numero di elementi per classe
        i = 0
        for classi in lista_classi:
            i += len(df[(df["class"] == classi)])
            print(classi +": " + str(len(df[(df["class"] == classi)])))

        # Totale
        print(f"Totale: {len(df)} - Calcolati: {i}\n")

    except FileNotFoundError:
        print(f"Errore: Il file '{"_annotations.csv"}' non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")


# Sostituisci con il nome reale del tuo file CSV
mio_file_csv = 'dataset_segnali.csv'
visualizza_csv_in_pycharm(mio_file_csv)