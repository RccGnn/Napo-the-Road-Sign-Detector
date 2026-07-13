import streamlit as st
from streamlit_option_menu import option_menu


def select_model():

    with st.columns([1, 2, 1])[1]:

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        st.subheader(
            "🧠 Seleziona il modello AI"
        )


    selected_model = option_menu(
        menu_title=None,
        options=[
            "EfficientNet",
            "ConvNeXt"
        ],
        orientation="horizontal",
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
            EfficientNetV2-M è una rete convoluzionale progettata per ottenere un elevato compromesso tra accuratezza ed efficienza computazionale. 
            La sua architettura combina blocchi Fused-MBConv e MBConv, integrando meccanismi di Squeeze-and-Excitation per migliorare la selezione delle caratteristiche più informative. 
            Utilizza una strategia di compound scaling che bilancia profondità, ampiezza e risoluzione dell’immagine. 
            Grazie alla riduzione dei costi di addestramento e al numero contenuto di parametri, è particolarmente adatta per applicazioni dove sono richieste elevate prestazioni con risorse limitate.
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
            ConvNeXt-Base è una moderna architettura convoluzionale sviluppata per portare le CNN tradizionali a prestazioni paragonabili ai Vision Transformer. 
            Il modello introduce modifiche strutturali ispirate ai Transformer, come Layer Normalization, convoluzioni depthwise con kernel di grandi dimensioni e blocchi con attivazione GELU. 
            La rete presenta una struttura gerarchica a quattro stadi che consente di estrarre caratteristiche visive sempre più astratte e globali. 
            Grazie alla sua elevata capacità rappresentativa, è particolarmente indicata per compiti di classificazione, rilevamento e segmentazione di immagini ad alta precisione.
            </div>
            """,
            unsafe_allow_html=True
        )


    return selected_model