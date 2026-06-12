import pandas as pd
import numpy as np
from scipy import stats

def load_and_wrangle(filepath, short_ma_window=20, long_ma_window=50):
    # Read TSLA CSV
    df = pd.read_csv(filepath)
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate Moving Averages
    df['Short_MA'] = df['Close'].rolling(window=short_ma_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_ma_window).mean()
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # High-Low differences
    df['High_Low_Diff'] = df['High'] - df['Low']
    
    # Log closes
    df['Log_Close'] = np.log(df['Close'])
    
    # Target variable Is_High_Value
    df['Is_High_Value'] = (df['Close'] > df['Short_MA']).astype(int)
    
    # Drop NAs
    df = df.dropna().reset_index(drop=True)
    return df

def run_correlation_test(df):
    # Pearson correlation between Volume and Close
    corr, p_value = stats.pearsonr(df['Volume'], df['Close'])
    return corr, p_value

def run_volume_ttest(df, year1, year2):
    # Scipy T-Test independent between two selectable years
    df['Year'] = df['Date'].dt.year
    vol_year1 = df[df['Year'] == year1]['Volume']
    vol_year2 = df[df['Year'] == year2]['Volume']
    
    if len(vol_year1) == 0 or len(vol_year2) == 0:
        return None, None
        
    t_stat, p_value = stats.ttest_ind(vol_year1, vol_year2, equal_var=False)
    return t_stat, p_value

def run_normality_test(df):
    # Shapiro-Wilk test on a sample of daily returns
    sample = df['Daily_Return'].dropna()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    stat, p_value = stats.shapiro(sample)
    return stat, p_value

def get_outlier_bounds(df):
    # Calculate Q1, Q3, and IQR boundaries for Close price
    Q1 = df['Close'].quantile(0.25)
    Q3 = df['Close'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return Q1, Q3, IQR, lower_bound, upper_bound
