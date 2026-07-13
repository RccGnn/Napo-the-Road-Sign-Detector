import streamlit as st
import pandas as pd

from predictor import predict

from services.history_service import save_prediction

from model_info import (
    get_model_size,
    get_parameters,
    get_accuracy
)


def show_result(
        image,
        selected_model,
        input_mode
):

    with st.columns([1, 2, 1])[1]:

        st.image(
            image,
            caption="Immagine acquisita",
            width=325
        )


    col1, col2, col3 = st.columns(
        [2.75, 4, 3]
    )


    with col2:

        recognize = st.button(
            "Riconosci segnale",
            use_container_width=True
        )


    if recognize:

        results, inference_time = predict(
            image,
            selected_model
        )


        save_prediction(
            results,
            selected_model,
            input_mode
        )


        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #00b09b, #96c93d);
                padding: 15px;
                border-radius: 15px;
                color: white;
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            ">
                🚦 Risultato ottenuto con {selected_model}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        st.subheader(
            "🎯 Probabilità delle predizioni"
        )


        df_results = pd.DataFrame(
            results,
            columns=[
                "Classe",
                "Probabilità (%)"
            ]
        )


        df_results.loc[0, "Classe"] = (
            "🏆 "
            +
            df_results.loc[0, "Classe"]
        )


        df_results.index = (
            df_results.index + 1
        )


        st.dataframe(
            df_results,
            use_container_width=True
        )


        st.divider()


        st.subheader(
            "📊 Informazioni modello"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Parametri",
                f"{get_parameters(selected_model)} M"
            )


        with col2:

            st.metric(
                "Dimensione",
                f"{get_model_size(selected_model)} MB"
            )


        with col3:

            st.metric(
                "Accuracy test",
                get_accuracy(selected_model)
            )


        st.info(
            f"⏱️ Tempo di inferenza per riconoscere l'immagine: "
            f"{inference_time * 1000:.2f} ms"
        )


        st.divider()


        st.subheader(
            "📜 Storico completo dei riconoscimenti"
        )


        if len(st.session_state.history) > 0:

            history_df = pd.DataFrame(
                reversed(
                    st.session_state.history
                )
            )


            history_df.index = (
                history_df.index + 1
            )


            st.dataframe(
                history_df,
                use_container_width=True
            )


        else:

            st.info(
                "Nessun segnale riconosciuto ancora."
            )