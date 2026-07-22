import streamlit as st
import pandas as pd

from components.header import show_header
from components.model_selector import select_model
from components.input_selector import select_input
from components.result_view import show_result
from components.history_view import show_history

from services.history_service import (
    initialize_history
)

from camera_predictor import (
    run_camera,
    history_queue
)

from streamlit_autorefresh import (
    st_autorefresh
)

from utils.layout import (
    centered_container
)

# Per runnare l'applicazione:
# 1) Posizionarsi nella cartella (scegliere un percorso breve per testare il progetto): cd Application
# 2) streamlit run app.py

# Configurazione del tab
st.set_page_config(
    page_title="Road Sign Detector",
    page_icon="🚦"
)

initialize_history()

show_header()

selected_model = select_model()

show_history()

input_mode, image = select_input()

"""---"""

# Telecamera live. La scelta di non utilizzare un package differente per la telecamera live è stata determinata da vincoli di software.

if input_mode == "Telecamera live":
    st_autorefresh(
        interval=3000,
        key="live_history_refresh"
    )

    col_video, col_history = st.columns(
        [2, 1]
    )

    with col_video:
        st.subheader(
            "🚗 Riconoscimento in tempo reale"
        )
        run_camera(selected_model)

    with col_history:
        st.subheader(
            "📜 Lista in diretta"
        )
        while not history_queue.empty():
            item = history_queue.get()

            st.session_state.history.insert(
                0,
                item
            )

        live_history = [
            item for item in st.session_state.history
            if item["Sorgente"] == "Telecamera Live"
        ]

        if len(live_history) > 0:
            live_df = pd.DataFrame(
                live_history[:10]
            )
            st.dataframe(
                live_df,
                hide_index=True,
                use_container_width=True
            )

        else:
            st.info(
                "In attesa di segnali dalla telecamera live..."
            )

# Predizione
elif image is not None:

    show_result(
        image,
        selected_model,
        input_mode
    )