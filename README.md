# ⚡ Electricity Load Forecasting Dashboard

A machine-learning project that forecasts short-term electricity demand
using historical load, time patterns (hour/day/season), and temperature.
Built as a beginner-friendly EEE academic project — no hardware required.

## Project structure

```
load_forecast/
├── data_generator.py     # Creates a realistic synthetic hourly load dataset
├── train_model.py        # Feature engineering + trains RandomForestRegressor
├── app.py                # Streamlit dashboard (the deployed app)
├── requirements.txt       # Python dependencies
├── load_data.csv         # Generated dataset (created by data_generator.py)
├── model.joblib          # Trained model (created by train_model.py)
└── README.md
```

## How it works (for your report/viva)

1. **Data**: Hourly electricity load (MW) and temperature (°C) for one year.
   Patterns baked in: daily peaks (morning ~9am, evening ~7pm), lower
   weekend load, seasonal swings, and temperature-driven demand (AC/heating).
2. **Features engineered**: hour of day, day of week, month, weekend flag,
   temperature, load 1 hour ago, load same hour yesterday (lag features —
   these are what let the model "remember" recent trends).
3. **Model**: Random Forest Regressor (100s of decision trees averaged
   together) — chosen because it handles non-linear patterns well and
   needs no scaling, making it easy to explain in a viva.
4. **Evaluation**: MAE, RMSE, R² on a held-out time period (last 20% of
   the year, kept in chronological order — never shuffle time series data).
5. **Dashboard**: Streamlit app lets a user pick a forecast horizon (6–72
   hours) and a temperature "what-if" adjustment, then see the forecast
   plotted against recent history.

## Run it locally

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python3 data_generator.py

# 4. Train the model
python3 train_model.py

# 5. Launch the dashboard
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

## Using a real dataset instead of synthetic data

To make this project stronger, swap in a real dataset once the pipeline
works:
- **Kaggle**: "Hourly Energy Consumption" (PJM Interconnection, USA)
- **UCI ML Repository**: "Individual household electric power consumption"

Just make sure your CSV has a datetime column and a load/consumption
column, then rename the columns to `datetime` and `load_mw` (or update
those names in `train_model.py` and `app.py`).

## Deployment (Streamlit Community Cloud — free, ~10 minutes)

1. **Create a GitHub repo** and push this whole folder to it:
   ```bash
   cd load_forecast
   git init
   git add .
   git commit -m "Initial commit: load forecasting project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   Important: also commit `load_data.csv` and `model.joblib` (don't
   .gitignore them) so the deployed app has data/model to load without
   re-running the scripts.

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in
   with your GitHub account.

3. Click **"New app"**, select your repo, branch (`main`), and set the
   main file path to `app.py`.

4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and give you a public URL like:
   `https://your-app-name.streamlit.app`

5. Share that link — this is what you submit/demo for your project.

### Alternative: deploy on Render / Railway (if you prefer a Flask-style host)
If your college asks for a "web app" broadly rather than specifically
Streamlit, Streamlit Cloud is still the fastest path. Render/Railway are
better suited if you later convert this into a Flask app — happy to help
with that conversion if needed.

## Possible extensions (if you want to go further for extra marks)

- Add an LSTM/GRU deep learning model and compare its accuracy to the
  Random Forest baseline.
- Add anomaly detection to flag unusual demand spikes.
- Connect to a live weather API for real temperature forecasts instead
  of the "what-if" slider.
- Add a PDF/Word report generator summarizing forecast accuracy.
