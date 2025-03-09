import streamlit as st
import scipy.interpolate as interp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
        page_title="EA Recherche - Simulation de contrôle",layout="wide"
    )

if "eta" not in st.session_state:
    st.session_state.eta = 0.2
if "rho" not in st.session_state:
    st.session_state.rho = -0.7
if "mu" not in st.session_state:
    st.session_state.mu = 0.07
if "sigma2" not in st.session_state:
    st.session_state.sigma2 = 0.04
if "a" not in st.session_state:
    st.session_state.a = 2
if "ksi" not in st.session_state:
    st.session_state.ksi = 0.25

if "T" not in st.session_state:
    st.session_state.T = 1
if "dt" not in st.session_state:
    st.session_state.dt = 0.004
if "V_min" not in st.session_state:
    st.session_state.V_min = 0.01
if "V_max" not in st.session_state:
    st.session_state.V_max = 0.51
if "N_V" not in st.session_state:
    st.session_state.N_V = 101

if "sim_T" not in st.session_state:
    st.session_state.sim_T = 1
if "sim_dt" not in st.session_state:
    st.session_state.sim_dt = 0.004
if "sim_V_min" not in st.session_state:
    st.session_state.sim_V_min = 0.01
if "sim_V_max" not in st.session_state:
    st.session_state.sim_V_max = 0.51
if "sim_N_V" not in st.session_state:
    st.session_state.sim_N_V = 101

if "sim_type" not in st.session_state:
    st.session_state.sim_type= "Monte-Carlo"
if "N_sim" not in st.session_state:
    st.session_state.N_sim = 250

if "has_sim" not in st.session_state:
    st.session_state.has_sim = False
if "v_star" not in st.session_state:
    st.session_state.v_star= None
if "control_opt" not in st.session_state:
    st.session_state.control_opt = None

if "choix_instant" not in st.session_state:
    st.session_state.choix_instant = 0
if "choix_var" not in st.session_state:
    st.session_state.choix_var = 0.2


st.title("Simulation de contrôle")
cols = st.columns(2)
cols[0].write("### Choix paramètres stochastiques")
cols[1].write("### Choix paramètres de simulations ")

c = cols[0].columns(4)
cp = cols[1].columns(4)
st.session_state.eta = c[0].text_input("Entrer une valeur pour $\eta$:", value=st.session_state.eta)
st.session_state.rho = c[1].text_input(r"Entrer une valeur pour $\rho$:", value=st.session_state.rho)
st.session_state.mu = c[2].text_input("Entrer une valeur pour $\mu$:", value=st.session_state.mu)
st.session_state.sigma2 = c[3].text_input("Entrer une valeur pour $\sigma^2$:", value=st.session_state.sigma2)
st.session_state.a = c[0].text_input("Entrer une valeur pour $a$:", value=st.session_state.a)
st.session_state.ksi = c[1].text_input(r"Entrer une valeur pour $\xi$:", value=st.session_state.ksi)
st.session_state.T = c[2].text_input("Entrer une valeur pour $T$:", value=st.session_state.T)

st.session_state.dt = cp[0].text_input(r"Entrer une valeur pour $\text{d}t$:", value=st.session_state.dt)
st.session_state.V_min = cp[1].text_input(r"Entrer $\sqrt{V_\text{min}}$:", value=st.session_state.V_min)
st.session_state.V_max = cp[2].text_input(r"Entrer  $\sqrt{V_\text{max}}$:", value=st.session_state.V_max)
st.session_state.N_V = cp[3].text_input(r"Entrer une valeur pour $N_V$ le nombre de points en $V$:", value=st.session_state.N_V)
st.session_state.sim_type = cp[0].radio("Choisir mode de simulation", ('Monte-Carlo', 'EDP'))

if st.session_state.sim_type == "Monte-Carlo":
    st.session_state.N_sim = cp[1].text_input("Entrer un nombre d'échantillons de simulation:", value=st.session_state.N_sim)
if cp[2].button("Simuler"):
    st.session_state.has_sim = True
    eta = float(st.session_state.eta)
    rho = float(st.session_state.rho)
    mu = float(st.session_state.mu)
    sigma2 = float(st.session_state.sigma2)
    a = float(st.session_state.a)
    ksi = float(st.session_state.ksi)
    T = float(st.session_state.T)
    dt = float(st.session_state.dt)
    st.session_state.sim_T = T
    st.session_state.dt = dt
    N_t = int(T / dt) + 1
    V_values = np.linspace(float(st.session_state.V_min), float(st.session_state.V_max), int(st.session_state.N_V))
    V_grid = V_values ** 2
    bn = np.sqrt(dt)
    v_star = np.zeros((N_t, len(V_grid)))
    control_opt = np.zeros((N_t, len(V_grid)))
    v_star[-1, :] = -1
    st.session_state.sim_N_V = int(st.session_state.N_V)
    st.session_state.sim_V_min = float(st.session_state.V_min)
    st.session_state.sim_V_max = float(st.session_state.V_max)

    if st.session_state.sim_type == "Monte-Carlo":
        progress_bar = cols[1].progress(0)
        N_sim = int(st.session_state.N_sim)
        for t in range(N_t - 2, -1, -1):  # rétrograde de T-Δt
            v_interp = interp.interp1d(V_grid, v_star[t + 1], fill_value="extrapolate", kind="linear")  ## pour être sur
            for i, v in enumerate(V_grid):
                best_valeur = -np.inf
                best_alpha = 0

                dW1_ = np.random.normal(0, bn, N_sim)
                dW2_ = np.random.normal(0, bn, N_sim)
                V_next = v - a * (v - sigma2) * dt + ksi * np.sqrt(v) * (rho * dW1_ + np.sqrt(1 - rho ** 2) * dW2_)
                V_next_ant = v - a * (v - sigma2) * dt - ksi * np.sqrt(v) * (rho * dW1_ + np.sqrt(1 - rho ** 2) * dW2_)
                v_t_dt = v_interp(V_next)
                v_t_dt_ant = v_interp(V_next_ant)
                X_prime = mu * dt + np.sqrt(v) * dW1_
                X_prime_ant = mu * dt - np.sqrt(v) * dW1_
                for alpha in range(101):
                    valeur = ((np.exp(-eta * alpha * X_prime) @ v_t_dt / N_sim) +
                              (np.exp(-eta * alpha * X_prime_ant) @ v_t_dt_ant / N_sim)) / 2
                    if valeur > best_valeur :
                        best_valeur = valeur
                        best_alpha = alpha

                v_star[t, i] = best_valeur
                control_opt[t, i] = best_alpha
            progress = (N_t - t - 1) / (N_t - 1)
            progress_bar.progress(progress)
        st.session_state.control_opt = control_opt
        st.session_state.v_star = v_star
        st.rerun()

if st.session_state.v_star is not None:
    st.write(f"### Résultats simulations - {st.session_state.sim_type}")
    col_prime = st.columns(2)
    t= col_prime[0].slider("Choisir l'instant $t$: ", 0.0, float(st.session_state.sim_T), step = st.session_state.sim_dt)
    step_V =(- float(st.session_state.sim_V_min)+ float(st.session_state.sim_V_max))/st.session_state.sim_N_V
    sqrt_v = col_prime[1].slider("Choisir la variance $\sqrt{V}$: ", float(st.session_state.sim_V_min), float(st.session_state.sim_V_max), step = step_V)
    t_values = np.linspace(0.0, float(st.session_state.sim_T),int( float(st.session_state.sim_T)/st.session_state.sim_dt))
    V_values = np.linspace(float(st.session_state.sim_V_min), float(st.session_state.sim_V_max),st.session_state.sim_N_V)
    V_grid= V_values**2
    t_index = int(t/st.session_state.sim_dt)
    v_fixed = (sqrt_v)**2
    # Ensure v_fixed is formatted correctly first
    subplot_title_2 = r"Contrôle optimal en temps"

    # Use the formatted title for subplot_titles
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Contrôle optimal en racine(V)", subplot_title_2))

    # Plot control_opt(v) at fixed t
    fig.add_trace(
        go.Scatter(x=V_values, y=st.session_state.control_opt[t_index, :], mode='lines',
                   name=f"control_opt avec t={t_values[t_index]:.2f}"),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Racine de variance ", row=1, col=1)
    fig.update_yaxes(title_text=r"Contrôle optimal", row=1, col=1)

    # Plot control_opt(t) at fixed v
    index_v = np.argmin(np.abs(V_grid - v_fixed))
    fig.add_trace(
        go.Scatter(x=t_values, y=st.session_state.control_opt[:, index_v], mode='lines',
                   name=f"Contrôle pour sqrt(V)={v_fixed:.2f}"),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Temps", row=1, col=2)
    fig.update_yaxes(title_text=r"Contrôle optimal", row=1, col=2)
    fig.update_yaxes(range=[0, 110], row=1, col=2)  # Limiting the y-axis range

    fig.update_layout(
        showlegend=True,
        height=600,
        width=1500
    )
    st.write(r"###### Contrôle optimal (fig 1: $\sqrt{V}\mapsto \hat{\alpha}[t,\sqrt{V}$], fig 2: $t\mapsto \hat{\alpha}[t,\sqrt{V}$] )")


    st.plotly_chart(fig)
