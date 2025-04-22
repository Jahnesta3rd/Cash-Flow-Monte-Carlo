import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv('CPW 2022-2025 V2.csv')
df['Date'] = pd.to_datetime(df['Date'])
numeric_df = df.select_dtypes(include=[np.number])

# Classify inflows and outflows
avg_values = numeric_df.mean()
inflow_cols = avg_values[avg_values > 0].index.tolist()
outflow_cols = avg_values[avg_values < 0].index.tolist()

# Sidebar Controls
st.sidebar.title("Stress Test Controls")
volatility_shock = st.sidebar.slider("Volatility Shock (%)", 0, 100, 0)
inflow_reduction = st.sidebar.slider("Inflow Reduction (%)", 0, 50, 0)
outflow_increase = st.sidebar.slider("Outflow Increase (%)", 0, 50, 0)
confidence_level = st.sidebar.selectbox("Confidence Level", ["90%", "95%"])

# Monte Carlo Simulation Setup
n_sim = 10000

def simulate_category(data, shock, adjust, is_inflow=True):
    mu, sigma = np.mean(data), np.std(data)
    sigma *= (1 + shock / 100)
    sim_data = np.random.normal(mu, sigma, n_sim)
    if is_inflow:
        sim_data *= (1 - adjust / 100)
    else:
        sim_data *= (1 + adjust / 100)
    return sim_data

# Simulate
sim_inflows = np.zeros((n_sim, len(inflow_cols)))
sim_outflows = np.zeros((n_sim, len(outflow_cols)))

for idx, col in enumerate(inflow_cols):
    sim_inflows[:, idx] = simulate_category(numeric_df[col], volatility_shock, inflow_reduction, True)

for idx, col in enumerate(outflow_cols):
    sim_outflows[:, idx] = simulate_category(numeric_df[col].abs(), volatility_shock, outflow_increase, False)

net_cash_flow = sim_inflows.sum(axis=1) - sim_outflows.sum(axis=1)

# Determine Threshold
conf_val = 0.90 if confidence_level == "90%" else 0.95
percentile_cutoff = (1 - conf_val) * 100
net_threshold = np.percentile(net_cash_flow, percentile_cutoff)

# Display Results
st.title("Monte Carlo Cash Flow Dashboard")
st.metric("Net Cash Flow Threshold", f"${net_threshold:,.0f}")
st.metric("Target", "$50,000,000")
st.metric("Meets Target", "Yes" if net_threshold >= 50_000_000 else "No")

# Plot Distribution
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(net_cash_flow, bins=50, color='skyblue', edgecolor='black')
ax.axvline(x=50_000_000, color='red', linestyle='--', label='$50M Target')
ax.set_title('Net Cash Flow Distribution')
ax.set_xlabel('Net Cash Flow ($)')
ax.set_ylabel('Frequency')
ax.legend()
st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.write("Adjust stress parameters to see impact on cash flow risk.")
