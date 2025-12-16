import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Multi-Factor Trading Strategy (MFT) Web App")

# 1️⃣ Upload CSV file
uploaded_file = st.file_uploader("Upload CSV with Date, Close, Volume", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')

    # 2️⃣ Clean numeric columns
    for col in ['Close', 'Volume']:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
            .str.replace('$', '', regex=False)
        )
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna()  # Drop rows with invalid data

    # 3️⃣ Factor Calculation
    data['Momentum'] = data['Close'].pct_change(20)
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = 1 / (data['Returns'].rolling(20).std() + 1e-9)
    data['VolumeFactor'] = data['Volume'].pct_change(20)
    data = data.dropna()

    # 4️⃣ Winsorization
    def winsorize(series, limits=0.01):
        lower = series.quantile(limits)
        upper = series.quantile(1 - limits)
        return np.clip(series, lower, upper)

    for col in ['Momentum', 'Volatility', 'VolumeFactor']:
        data[col] = winsorize(data[col])

    # 5️⃣ Z-score
    for col in ['Momentum', 'Volatility', 'VolumeFactor']:
        data[col + '_z'] = (data[col] - data[col].mean()) / data[col].std()

    # 6️⃣ Composite Score
    data['MFT_Score'] = (data['Momentum_z'] + data['Volatility_z'] + data['VolumeFactor_z']) / 3

    # 7️⃣ Signals
    upper = data['MFT_Score'].quantile(0.7)
    lower = data['MFT_Score'].quantile(0.3)
    data['Signal'] = 0
    data.loc[data['MFT_Score'] > upper, 'Signal'] = 1
    data.loc[data['MFT_Score'] < lower, 'Signal'] = -1
    data['Signal_Label'] = data['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})

    # 8️⃣ Strategy Returns
    data['StrategyReturn'] = data['Signal'].shift(1) * data['Returns']
    data['CumulativeReturn'] = (1 + data['StrategyReturn']).cumprod()

    # 9️⃣ Performance Metrics
    N = len(data)
    ending_value = data['CumulativeReturn'].iloc[-1]
    ann_return = ending_value ** (252 / N) - 1
    ann_vol = data['StrategyReturn'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    st.subheader("Performance Metrics")
    st.write(f"Annualized Return: {ann_return:.4%}")
    st.write(f"Annualized Volatility: {ann_vol:.4%}")
    st.write(f"Sharpe Ratio: {sharpe:.4f}")

    # 🔟 Display Sample Data
    st.subheader("Sample Data with Signals")
    st.dataframe(data.tail(20))

    # 1️⃣1️⃣ Plot Buy/Sell signals
    st.subheader("Stock Price with Buy/Sell Signals")

    fig = go.Figure()

    # Close Price
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        mode='lines',
        name='Close Price'
    ))

    # Buy signals
    fig.add_trace(go.Scatter(
        x=data[data['Signal'] == 1].index,
        y=data[data['Signal'] == 1]['Close'],
        mode='markers',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        name='Buy Signal'
    ))

    # Sell signals
    fig.add_trace(go.Scatter(
        x=data[data['Signal'] == -1].index,
        y=data[data['Signal'] == -1]['Close'],
        mode='markers',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        name='Sell Signal'
    ))

    fig.update_layout(height=500, width=900, template='plotly_white')
    st.plotly_chart(fig)

    # 1️⃣2️⃣ Option to download processed CSV
    st.subheader("Download Processed Data with Signals")
    csv = data.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="processed_stock_data.csv",
        mime="text/csv"
    )
