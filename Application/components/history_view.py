import streamlit as st
import pandas as pd


def show_history():

    with st.expander(
        "📜 Storico completo dei riconoscimenti fino all'immagine precedentemente inserita"
    ):

        if len(st.session_state.history) > 0:

            history_df = pd.DataFrame(
                st.session_state.history
            )


            st.dataframe(
                history_df,
                hide_index=True,
                use_container_width=True
            )


        else:

            st.info(
                "Nessun riconoscimento effettuato."
            )