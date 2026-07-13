import streamlit as st


def show_header():

    st.markdown(
        """
        <h1 style="
            text-align:center;
            margin-bottom:0px;
            font-size:42px;
        ">
        🚦 Riconoscimento Segnali Stradali
        </h1>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    st.markdown(
        """
    <div style="
        padding:25px 10px;
        border-bottom:2px solid #e6e6e6;
        text-align: center;
    ">

    <span style="
        font-size:30px;
        font-weight:600;
    ">
    📸 Carica un'immagine del segnale
    </span>

    <span style="
        color:#999;
        font-size:20px;
    ">
    Il nostro sistema offre la predizione dei modelli EfficientNet e ConvNeXt.
    </span>

    </div>
    """,
        unsafe_allow_html=True
    )