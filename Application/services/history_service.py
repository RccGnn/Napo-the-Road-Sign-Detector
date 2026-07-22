import streamlit as st
from datetime import datetime

def initialize_history():
    if "history" not in st.session_state:
        st.session_state.history = []

"""---"""

def save_prediction(results, model, source):
    best_class = results[0][0]
    confidence = results[0][1]

    entry = {
        "Ora": datetime.now().strftime("%H:%M:%S"),
        "Segnale": best_class,
        "Probabilità (%)": round(confidence, 2),
        "Modello": model,
        "Sorgente": source
    }

    st.session_state.history.insert(0, entry)