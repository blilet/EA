import streamlit as st
import scipy.interpolate as interp
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="EA Recherche - Simulation de trajectoire sous contrôle optimal", layout="wide"
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
if "N__" not in st.session_state:
    st.session_state.N__ = 100
if "second_session" not in st.session_state:
    st.session_state.second_session = False

if not st.session_state.has_sim:
    st.write(st.session_state.N_V)
    st.error('Pas de simulation de contrôle fait en avance (voir page 3)')
else:
    st.title("Simulation de trajectoires sous contrôle optimal et P&L")
    st.write("### Simulation de $(Y_t), (V_t)$ et $(X_t)$ sous contrôle optimal")
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
    st.write("Choisir valeurs initiales pour: ")
    cols = st.columns(5)
    st.session_state.X0 = float(cols[0].text_input(" $X_0$:", value=st.session_state.X0))
    st.session_state.Y0 = float(cols[1].text_input(" $Y_0$:", value=st.session_state.Y0))
    st.session_state.V0 = float(cols[2].text_input(" $V_0$:", value=st.session_state.V0))
    c = cols[4].columns(2)
    N = int(c[0].text_input(r" $N_\text{sim}$:", value=100))
    st.session_state.N_second = int(cols[3].text_input(r" Trajectoires à visualiser (<$N_{\text{sim}}$):", value=2))

    st.write('Pour rappel les autres paramètres sont fixées à: ')
    st.write(f'$\eta = {st.session_state.eta}$, $a = {st.session_state.a}$ ,' + r' $\rho = $ ' + str(st.session_state.rho) +  f', $\sigma^2 = {st.session_state.sigma2}$, ' + r' $\xi = $' + str(st.session_state.ksi) + f', $T = {st.session_state.T}$ et ' + r'$\text{d}t = $'  + str(st.session_state.dt))
    X0= st.session_state.X0
    Y0 = st.session_state.Y0
    V0 = st.session_state.V0
    t_values = np.arange(N_t) * dt
    c[1].write(" ")
    c[1].write(" ")
    if c[1].button("Simuler"):
        st.session_state.second_session = True
        Y = np.zeros((N_t, N))
        V = np.zeros((N_t, N))
        X = np.zeros((N_t, N))
        alpha_sim = np.zeros((N_t, N))

        Y[0, :] = Y0
        V[0, :] = V0
        X[0, :] = X0

        for t in range(N_t - 1):
            V[t + 1, :] = V[t, :] - a * (V[t, :] - sigma2) * dt + ksi * np.sqrt(V[t, :]) * (
                        rho * np.random.randn(N) + np.sqrt(1 - rho ** 2) * np.random.randn(N))
            V[t + 1, :] = np.clip(V[t + 1, :], np.min(V_grid), np.max(V_grid))

            v_interp = interp.interp1d(V_grid, st.session_state.control_opt[t], fill_value="extrapolate")
            alpha_sim[t, :] = v_interp(V[t, :])

            Y[t + 1, :] = Y[t, :] * np.exp((mu - 0.5 * V[t, :]) * dt + np.sqrt(V[t, :]) * np.sqrt(dt) * np.random.randn(N))
            X[t + 1, :] = X[t, :] + alpha_sim[t, :] * (mu * dt + np.sqrt(V[t, :]) * np.sqrt(dt) * np.random.randn(N))
        st.session_state.X = X
        st.session_state.Y = Y
        st.session_state.V = V
        st.rerun()
    if st.session_state.second_session:

        fig1 = go.Figure()
        for i in range(st.session_state.N_second):  # Adjust range for the number of series you want to plot
            fig1.add_trace(go.Scatter(x=t_values, y=st.session_state.Y[:, i], mode='lines', name=f'Y_{i + 1}'))
        fig1.update_layout(title="Simulation de prix Y_t", xaxis_title="Temps t", yaxis_title="Prix d'actif Y_t")

        fig2 = go.Figure()
        for i in range(st.session_state.N_second):
            fig2.add_trace(go.Scatter(x=t_values, y=st.session_state.V[:, i], mode='lines', name=f'V_{i + 1}'))
        fig2.update_layout(title="Simulation de variance V_t", xaxis_title="Temps t", yaxis_title="Variance V_t")

        fig3 = go.Figure()
        for i in range(st.session_state.N_second):
            fig3.add_trace(go.Scatter(x=t_values, y=st.session_state.X[:, i], mode='lines', name=f'X_{i + 1}'))
        fig3.update_layout(title="Simulation de richesse X_t", xaxis_title="Temps t", yaxis_title="Richesse X_t")

        # Display the plots in Streamlit
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        st.plotly_chart(fig3)

        # Histogramme avec Plotly
        bins = 0
        if N>=100:
            bins = 100
        else:
            bins = N//10
        if N>=10:
            final_returns = (st.session_state.X[-1, :] - X0) / X0
            fig4 = go.Figure(data=[go.Histogram(x=final_returns, nbinsx=bins, histnorm='probability')])
            fig4.update_layout(title="Histogramme de performance du portefeuille", xaxis_title="Revenue portefeuille",
                               yaxis_title="Densité")

            # Display the histogram in Streamlit
            st.plotly_chart(fig4)
            st.write(f'### Moyenne empirique de la performance: {np.mean(final_returns):.3f} $\pm$ { np.std(final_returns)/np.sqrt(N):.5f}')



