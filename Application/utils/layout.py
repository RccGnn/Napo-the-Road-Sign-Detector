import streamlit as st

# Funzione per centrare elementi
def centered_container():

    return st.columns(
        [1, 2, 1]
    )[1]