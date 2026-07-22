import streamlit as st
from PIL import Image

"""---"""

def select_input():
    input_mode = st.radio(
        "📸 Scegli la sorgente dell'immagine:",
        [
            "Carica immagine",
            "Usa telecamera",
            "Telecamera live"
        ],
        horizontal=True
    )

    image = None

    # Caricamento immagine
    if input_mode == "Carica immagine":
        uploaded_file = st.file_uploader(
            "Scegli un'immagine",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

        if uploaded_file is not None:
            image = Image.open(
                uploaded_file
            )

    # Scatto singolo
    elif input_mode == "Usa telecamera":
        camera_image = st.camera_input(
            "Scatta una foto del segnale"
        )

        if camera_image is not None:
            image = Image.open(
                camera_image
            )

    return input_mode, image