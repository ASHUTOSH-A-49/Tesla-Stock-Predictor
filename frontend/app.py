import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path to import from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.data_processor import (
    load_and_wrangle, run_correlation_test, run_volume_ttest, 
    run_normality_test, get_outlier_bounds
)
from backend.model_trainer import prepare_and_train_model

# Page Config
st.set_page_config(page_title="Tesla Stock Price Prediction", page_icon="🚗", layout="wide")

st.title("🚗 Tesla Stock Price Prediction")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload TSLA.csv", type=["csv"])

epochs = st.sidebar.slider("Epochs", min_value=5, max_value=50, value=20, step=1)
batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)
test_split = st.sidebar.slider("Test Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

short_ma_window = st.sidebar.slider("Short MA Window", min_value=5, max_value=50, value=20)
long_ma_window = st.sidebar.slider("Long MA Window", min_value=50, max_value=200, value=50)

model_selection = st.sidebar.selectbox("Model Selection", ["LSTM", "RNN", "Both"])

if not uploaded_file:
    st.warning("Please upload a TSLA.csv file in the sidebar to proceed.")
    st.stop()

@st.cache_data
def cached_load_and_wrangle(file, short_ma, long_ma):
    return load_and_wrangle(file, short_ma, long_ma)

try:
    df = cached_load_and_wrangle(uploaded_file, short_ma_window, long_ma_window)
except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Overview", "EDA & Charts", "Hypothesis Testing", 
    "Feature Engineering", "Model Training"
])

with tab1:
    st.header("Data Overview")
    st.subheader("Data Head")
    st.dataframe(df.head())
    
    st.subheader("Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    
    st.subheader("Null Value Check")
    st.write("Since NAs are dropped in wrangling, missing values should be 0.")
    st.dataframe(df.isnull().sum().rename("Null Count"))

with tab2:
    st.header("EDA & Charts")
    
    st.subheader("Close Price & Moving Averages")
    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
    fig_close.add_trace(go.Scatter(x=df['Date'], y=df['Short_MA'], mode='lines', name=f'{short_ma_window}-Day MA'))
    fig_close.add_trace(go.Scatter(x=df['Date'], y=df['Long_MA'], mode='lines', name=f'{long_ma_window}-Day MA'))
    fig_close.update_layout(xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig_close, use_container_width=True)
    
    st.subheader("Daily Volume")
    fig_vol = px.bar(df, x='Date', y='Volume', title="Daily Trading Volume", template="plotly_dark")
    st.plotly_chart(fig_vol, use_container_width=True)
    
    st.subheader("Candlestick Chart")
    fig_candle = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'])])
    fig_candle.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig_candle, use_container_width=True)
    
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig_corr)

with tab3:
    st.header("Hypothesis Testing")
    
    st.subheader("1. Pearson Correlation (Volume vs Close)")
    corr, p_corr = run_correlation_test(df)
    st.write(f"**Correlation Coefficient:** {corr:.4f}")
    st.write(f"**p-value:** {p_corr:.4e}")
    if p_corr < 0.05:
        st.success("Reject Null Hypothesis: Significant correlation exists between Volume and Close price.")
    else:
        st.error("Fail to Reject Null Hypothesis: No significant correlation observed.")
        
    st.subheader("2. Independent T-Test (Volume across Years)")
    years_avail = sorted(df['Date'].dt.year.unique())
    if len(years_avail) >= 2:
        col_y1, col_y2 = st.columns(2)
        year1 = col_y1.selectbox("Year 1", years_avail, index=0)
        year2 = col_y2.selectbox("Year 2", years_avail, index=len(years_avail)-1)
        
        t_stat, p_ttest = run_volume_ttest(df.copy(), year1, year2)
        if t_stat is not None:
            st.write(f"**T-Statistic:** {t_stat:.4f}")
            st.write(f"**p-value:** {p_ttest:.4e}")
            if p_ttest < 0.05:
                st.success(f"Reject Null Hypothesis: Significant difference in Volume between {year1} and {year2}.")
            else:
                st.error(f"Fail to Reject Null Hypothesis: No significant difference in Volume.")
        else:
            st.warning("Insufficient data for the selected years.")
    else:
        st.warning("Not enough years in data for T-Test comparison.")
        
    st.subheader("3. Normality Test (Shapiro-Wilk on Daily Returns)")
    stat_norm, p_norm = run_normality_test(df)
    st.write(f"**Test Statistic:** {stat_norm:.4f}")
    st.write(f"**p-value:** {p_norm:.4e}")
    if p_norm < 0.05:
        st.success("Reject Null Hypothesis: Daily Returns are NOT normally distributed.")
    else:
        st.error("Fail to Reject Null Hypothesis: Daily Returns appear normally distributed.")

with tab4:
    st.header("Feature Engineering")
    
    st.subheader("Engineered Columns Preview")
    features_to_show = ['Date', 'Close', 'Short_MA', 'Long_MA', 'Daily_Return', 'Volume_Change', 'High_Low_Diff', 'Log_Close', 'Is_High_Value']
    avail_cols = [col for col in features_to_show if col in df.columns]
    st.dataframe(df[avail_cols].head(15))
    
    st.subheader("Outlier Bounds (Close Price)")
    Q1, Q3, IQR, lower_bound, upper_bound = get_outlier_bounds(df)
    col_out1, col_out2, col_out3 = st.columns(3)
    col_out1.metric("Q1", f"${Q1:.2f}")
    col_out2.metric("Q3", f"${Q3:.2f}")
    col_out3.metric("IQR", f"${IQR:.2f}")
    
    st.write(f"**Lower Bound:** ${lower_bound:.2f}")
    st.write(f"**Upper Bound:** ${upper_bound:.2f}")
    
    outliers = df[(df['Close'] < lower_bound) | (df['Close'] > upper_bound)]
    st.write(f"Number of outliers detected: **{len(outliers)}**")
    if len(outliers) > 0:
        st.dataframe(outliers[['Date', 'Close']].head(10))

with tab5:
    st.header("Model Training")
    st.write("Train deep learning models to predict Tesla Stock Price.")
    
    if st.button("Train Models"):
        models_to_train = [model_selection] if model_selection in ["LSTM", "RNN"] else ["LSTM", "RNN"]
        
        for m_type in models_to_train:
            st.subheader(f"Training {m_type} Model...")
            with st.spinner(f"Training {m_type}... This might take a moment."):
                results = prepare_and_train_model(
                    df, model_type=m_type, epochs=epochs, 
                    batch_size=batch_size, test_split=test_split
                )
                
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("RMSE", f"{results['rmse']:.4f}")
            col_m2.metric("MAE", f"{results['mae']:.4f}")
            
            # Plot Loss Curve
            st.write(f"**{m_type} Loss Curve (Train vs Val)**")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=results['history']['loss'], mode='lines', name='Train Loss'))
            if 'val_loss' in results['history']:
                fig_loss.add_trace(go.Scatter(y=results['history']['val_loss'], mode='lines', name='Val Loss'))
            fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="MSE Loss", template="plotly_dark")
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Plot Actual vs Predicted
            st.write(f"**{m_type} Actual vs Predicted Prices (Test Set)**")
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(y=results['actuals'], mode='lines', name='Actual Price'))
            fig_pred.add_trace(go.Scatter(y=results['predictions'], mode='lines', name='Predicted Price'))
            fig_pred.update_layout(xaxis_title="Time Steps", yaxis_title="Price", template="plotly_dark")
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.markdown("---")
