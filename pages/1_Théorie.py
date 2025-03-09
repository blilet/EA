import streamlit as st
import numpy as np
import scipy.interpolate as interp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="EA Recherche - Théorie", layout="wide"
)
if "N_second" not in st.session_state:
    N_second = 100
if "X0" not in st.session_state:
    st.session_state.X0 = 100
if "Y0" not in st.session_state:
    st.session_state.Y0 = 10
if "V0" not in st.session_state:
    st.session_state.V0 = 0.04
if "X" not in st.session_state:
    st.session_state.X = None
if "Y" not in st.session_state:
    st.session_state.Y = None
if "V" not in st.session_state:
    st.session_state.V = None
if "Vp" not in st.session_state:
    st.session_state.Vp = None
if "N__" not in st.session_state:
    st.session_state.N__ = 100
if "second_session" not in st.session_state:
    st.session_state.second_session = False
if "third_session" not in st.session_state:
    st.session_state.third_session = True

st.title("Théorie et simulations préliminaires")
st.write("## Théorie")
st.write("##### Définitions")
st.write("On rappelle dans cette partie la base théorique de notre projet de recherche:")
st.write(r"On se fixe $T>0$. On considère deux mouvements brownien $(W_t^1)_{t\in [0,T]}$ et $(W_t^1)_{t\in [0,T]}$.")
st.write(r" On définit l'actif $(Y_t)$ par l'équation différentille stochastique suivante: $$ \text{d}Y_t = \mu Y_t\text{d}t + \sqrt{V}_t Y_t \text{d}W_t^1$$")
st.write(r"La variance vérifie l'EDS suivante: $\text{d}V_t = -a(V_t-\sigma^2)\text{d}t + \xi \sqrt{V}_t\bigg[\rho\text{d}W_t^1 + \sqrt{1-\rho^2}\text{d}W_t^2\bigg]$")
st.write(r"On considère un portfeuille autofinancé vérifiant l'équation suivante où $x$ est la richesse initialle et $\alpha$ un contrôle laissant $X^{t,x,v,\alpha}$ uniformément bornée:")
st.write(r"$$X_s^{t,x,y,v,\alpha} = x+ \int_{t}^s \dfrac{\alpha_u}{Y_{u}^{t,y,v}}\text{d}Y_u^{t,y,v} =x+ \int_{t}^s\alpha_u(\mu \text{d}u + \sqrt{V_u^{t,v}}\text{d}W_u^1)$$")
st.write("##### Enoncé du problème")
st.write(r"Pour la fonction utilité $U:x\mapsto U(x) = -\text{exp}(-\eta x)$ et un contrôle $\alpha$, on définit: $J(t,x,v,\alpha) = \textbf{E}(U(X_{T}^{t,x,v,\alpha})$")
st.write(r"Enfin, on note $\text{v}(t,x,v) = \sup_{\alpha} J(t,x,v,\alpha)$. Le but de ce projet est d'approcher $\text{argmax} \text{v}(t,x,v)$ pour tout triplet $(t,x,v)\in \textbf{R}^+\times \textbf{R}\times\textbf{R}^+$")
st.write("##### Propriétés utilisés en simulation")
st.write(r"Par définition de la fonction d'utilité on montre que $\text{v}(t,x,v) = e^{-\eta x}\text{v}_\circ(t,v)$ où $\text{v}_{\circ}(t,v) = \text{v}(t,0,v)$")
st.write(r"On montre (voir rapport sur GitHub) que $\text{v}_\circ$ vérifie une équation de programmation dynamique dont une conséquence et que:")
st.write(r"$$\text{v}_\circ(t,v) = \sup_{\alpha\in \textbf{R}}\textbf{E}(e^{-\eta X_{t+\Delta t}^{t,0,v,\alpha}} \text{v}_\circ(t+\Delta t, V_{t+\Delta t}^{t,v}))$$")
st.write(r"C'est cette équation utilisée pour le schéma Monte - Carlo.")
st.write(r"On montre aussi sous reserve de régularité que $\text{v}_\circ$ vérifie l'ED suivante")
st.write(r"$$\partial_t\text{v}_\circ -a(v-\sigma^2)   + \dfrac{1}{2}+ \partial_{v,v}\text{v}_\circ \zeta^2 - \dfrac{(\mu\partial_v\text{v}_\circ - \eta \partial_v\text{v}_\circ \zeta \rho v)^2}{2\eta v\text{v}_\circ} = 0 $$")
st.write(r"Cette équation est utilisé pour le schéma EDP en simulation (page 2)")
