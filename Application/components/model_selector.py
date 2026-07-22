import streamlit as st
from streamlit_option_menu import option_menu

"""---"""

def select_model():

    with st.columns([1, 2, 1])[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🧠 Seleziona il modello AI")

    selected_model = option_menu(
        menu_title=None,
        options=[
            "EfficientNet",
            "ConvNeXt"
        ],orientation="horizontal",
    )


    if selected_model == "EfficientNet":
        st.markdown(
            """
            <div style="
                text-align: justify;
                font-size: 14px;
                color: #667;
                line-height: 1.6;
                margin-bottom: 15px;
            ">
            EfficientNetV2-M è una rete neurale convoluzionale (CNN) progettata per offrire un buon 
            compromesso tra accuratezza ed efficienza computazionale. Grazie a un'architettura ottimizzata, garantisce 
            tempi di addestramento ridotti e prestazioni elevate in compiti di classificazione delle immagini e transfer learning.
            </div>            
            """,
            unsafe_allow_html=True
        )


    elif selected_model == "ConvNeXt":
        st.markdown(
            """
            <div style="
                text-align: justify;
                font-size: 14px;
                color: #667;
                line-height: 1.6;
                margin-bottom: 15px;
            ">
            ConvNeXt-Base è una CNN di nuova generazione che integra soluzioni architetturali ispirate ai Vision 
            Transformer mantenendo una struttura interamente convoluzionale. Si distingue per l'elevata capacità di 
            estrazione delle caratteristiche e per le ottime prestazioni in attività di classificazione, segmentazione e rilevamento di oggetti.
            </div>
            """,
            unsafe_allow_html=True
        )

    return selected_model