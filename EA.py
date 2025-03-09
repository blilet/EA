import streamlit as st
st.set_page_config(
        page_title="EA Recherche - Bienvenue",layout="wide")

st.title("EA Recherche")
st.write("#### Application de simulation:")
st.markdown(
    r"$$\blacktriangleright$$ L'onglet théorie contient les différentes formules utilisées."
)
st.markdown(
    r"$$\blacktriangleright$$ La deuxième page permet de régler les paramètres de simulations ainsi que le méthode (par EDP ou Monte-Carlo) pour obtenir le contrôle optimal."
)
st.markdown(
    r"$$\blacktriangleright$$ La troisième page permet de simuler plusieurs trajectoires de tous les processus et d'afficher des histogrammes de richesse."
)
st.markdown(
    r"$$\blacktriangleright$$ La dernière page permet de simuler une trajectoire de variance ainsi que le contrôle optimale."
)
st.write("Par XU Zhenyu, BLILET Hatim, QRICIHI ANIBA Adam sous encadrement de M. Nicolas Baradel.")
