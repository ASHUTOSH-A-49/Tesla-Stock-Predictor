@echo off
echo ========================================================
echo Tesla Stock Price Prediction - Setup ^& Run
echo ========================================================

echo.
echo [1/4] Checking for virtual environment...
IF NOT EXIST venv (
    echo Creating a new Python virtual environment...
    python -m venv venv
) ELSE (
    echo Virtual environment already exists.
)

echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/4] Upgrading pip and installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [4/4] Starting Streamlit Application...
streamlit run frontend\app.py

pause
