import streamlit as st
import numpy as np
import plotly.graph_objects as go
import scipy.interpolate as interp

st.set_page_config(
    page_title="EA Recherche - Simulation de trajectoire de variance et contrôle optimal", layout="wide"
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
if not st.session_state.has_sim:
    st.write(st.session_state.N_V)
    st.error('Pas de simulation de contrôle fait en avance (voir page 3)')
else:
    st.title("Simulation de trajectoires de variance et du contrôle optimal")
    st.write(r"### Simulation de $(V_t)$ et $t\mapsto \alpha[t,\sqrt{V_t}]$")
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
    v_star[-1, :] = -1
    st.session_state.sim_N_V = int(st.session_state.N_V)
    st.session_state.sim_V_min = float(st.session_state.V_min)
    st.session_state.sim_V_max = float(st.session_state.V_max)
    V0 = float(st.text_input('Entrer valeur pour $(V_0)$:', value= st.session_state.V0))
    if st.button('Simuler une trajectoire'):
        st.session_state.third_session = True
        V = np.zeros((N_t, 1))
        alpha_sim = np.zeros((N_t, 1))
        V[0, :] = V0
        for t in range(N_t - 1):
            V[t + 1, :] = V[t, :] - a * (V[t, :] - sigma2) * dt + ksi * np.sqrt(V[t, :]) * (
                        rho * np.random.randn(1) + np.sqrt(1 - rho ** 2) * np.random.randn(1))
            V[t + 1, :] = np.clip(V[t + 1, :], np.min(V_grid), np.max(V_grid))
            v_interp = interp.interp1d(V_grid, st.session_state.control_opt[t], fill_value="extrapolate")
            alpha_sim[t, :] = v_interp(V[t, :])
        st.session_state.Vp = V
        print(V)
    if st.session_state.third_session: 
        t_values = np.arange(N_t) * dt
        st.write(r"##### Simulation de Variance $V_t$ avec controle optimal $(\alpha[t,\sqrt{V_t}])$")
        fig5 = go.Figure()
        V_p= st.session_state.Vp.T
        L = np.zeros(len(t_values))
        for i in range(int( float(st.session_state.T)/st.session_state.dt)):
            v_fixed = V_p[0,i]
        
            index_v = np.argmin(np.abs(V_grid - v_fixed))
            L[i] = st.session_state.control_opt[i, index_v]
        print(V_p)
        fig5.add_trace(go.Scatter(x=t_values, y=100*np.sqrt(V_p[0,:]), mode='lines', name=f'sqrt(V_{1})*100'))
        fig5.add_trace(go.Scatter(x=t_values, y=L, mode='lines', name=f'Contrôle (alpha)'))

        fig5.update_layout(xaxis_title="Temps t", yaxis_title="Variance en % / Contrôle")
        st.plotly_chart(fig5)