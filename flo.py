import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(layout='wide')
st.title("Hadejia-Jama'are Basin: Flood Forecasting Dashboard")

st.sidebar.header("1. Upload Your Data")

hydro_file = st.sidebar.file_uploader("Upload Hydrology CSV", type="csv")
lulc_file = st.sidebar.file_uploader("Upload LULC/Soil CSV", type="csv")

def validate_and_load(file, required_cols, date_col='date'):
    """Load CSV and check required columns exist"""
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        # Make column names lowercase for flexibility
        df.columns = df.columns.str.lower().str.strip()
        required_cols = [c.lower() for c in required_cols]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns in {file.name}: {missing}")
            st.info(f"Required: {required_cols}")
            st.stop()

        # Parse dates - try multiple formats
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=False)
        if df[date_col].isna().any():
            st.error(f"Could not parse some dates in {file.name}. Use YYYY-MM-DD format.")
            st.stop()

        return df
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        st.stop()

# Required columns for your project
hydro_required = ['date', 'station', 'discharge_m3s', 'stage_m']
lulc_required = ['date', 'station', 'lulc_urban_pct', 'soil_infiltration_mmhr']

hydro_df = validate_and_load(hydro_file, hydro_required)
lulc_df = validate_and_load(lulc_file, lulc_required)

if hydro_df is None or lulc_df is None:
    st.info("👆 Upload both CSV files to start.")
    with st.expander("Required CSV format"):
        st.write("**Hydrology CSV must have:** `date, station, discharge_m3s, stage_m`")
        st.write("**LULC/Soil CSV must have:** `date, station, lulc_urban_pct, soil_infiltration_mmhr`")
        st.write("Optional: `is_forecast, flood_severity, channel_width_m, sinuosity_index, river, latitude, longitude`")
    st.stop()

# Handle missing 'is_forecast' column - infer from year if needed
if 'is_forecast' not in hydro_df.columns:
    current_year = datetime.now().year
    hydro_df['is_forecast'] = hydro_df['date'].dt.year >= current_year
    st.sidebar.warning(f"No 'is_forecast' column found. Assumed dates >= {current_year} are forecasts.")

# Handle missing 'flood_severity' - we can calc it later
if 'flood_severity' not in hydro_df.columns:
    hydro_df['flood_severity'] = 0

# Sidebar: Scenario controls
st.sidebar.header("2. 2030 Forecast Scenario")
stations = sorted(hydro_df['station'].unique())
station = st.sidebar.selectbox("Station", stations)

# Get urban % range from actual data
urban_min = int(lulc_df['lulc_urban_pct'].min())
urban_max = int(lulc_df['lulc_urban_pct'].max() * 1.5)
urban_current = int(lulc_df[lulc_df['station']==station]['lulc_urban_pct'].iloc[-1])

urban_scenario = st.sidebar.slider("Urban % in 2030",
                                   min_value=max(1, urban_min),
                                   max_value=min(50, urban_max),
                                   value=urban_current)
rainfall_factor = st.sidebar.slider("Rainfall Intensity Factor", 0.8, 1.5, 1.15)

# Filter to selected station
hist = hydro_df[(hydro_df['station']==station) & (hydro_df['is_forecast']==False)].copy()
forecast = hydro_df[(hydro_df['station']==station) & (hydro_df['is_forecast']==True)].copy()
lulc_station = lulc_df[lulc_df['station']==station].copy()

if hist.empty:
    st.error(f"No historical data for station '{station}'. Check your 'is_forecast' column.")
    st.stop()
if forecast.empty:
    st.warning(f"No forecast data for station '{station}'. Showing historical only.")
    forecast = hist.tail(365).copy() # fake 1 year for display
    forecast['is_forecast'] = True

# Simple scenario model: discharge scales with urban_pct and rainfall
base_urban = lulc_station['lulc_urban_pct'].iloc[-365:].mean()
urban_change = urban_scenario / base_urban if base_urban > 0 else 1
forecast['discharge_m3s'] = forecast['discharge_m3s'] * urban_change * rainfall_factor

# Update stage using rating curve if stage exists
if 'stage_m' in forecast.columns:
    forecast['stage_m'] = 0.3 + (forecast['discharge_m3s'] / 50)**0.6

# Recalc severity based on new discharge - Objective 3
q85, q95, q98, q995 = np.percentile(hist['discharge_m3s'], [85,95,98,99.5])
forecast['flood_severity'] = 0
forecast.loc[forecast['discharge_m3s'] > q85, 'flood_severity'] = 1
forecast.loc[forecast['discharge_m3s'] > q95, 'flood_severity'] = 2
forecast.loc[forecast['discharge_m3s'] > q98, 'flood_severity'] = 3
forecast.loc[forecast['discharge_m3s'] > q995, 'flood_severity'] = 4

plot_df = pd.concat([hist.tail(365*5), forecast])

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Discharge Forecast: {station}")
    fig = px.line(plot_df, x='date', y='discharge_m3s', color='is_forecast',
                  labels={'is_forecast':'Forecast Period', 'discharge_m3s':'Discharge m³/s'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Flood Severity Days per Year")
    severity_yearly = plot_df.groupby(plot_df['date'].dt.year)['flood_severity'].value_counts().unstack().fillna(0)
    severity_cols = ['None','Minor','Moderate','Major','Extreme']
    for i in range(len(severity_cols)):
        if i not in severity_yearly.columns:
            severity_yearly[i] = 0
    severity_yearly.columns = severity_cols
    st.bar_chart(severity_yearly[['Moderate','Major','Extreme']])

col1, col2, col3 = st.columns(3)
extreme_days = int(forecast[forecast['flood_severity']==4].shape[0])
hist_extreme = int(hist[hist['date'].dt.year >= hist['date'].dt.year.max()-5]['flood_severity'].eq(4).sum())
col1.metric("Forecast: Extreme Flood Days", extreme_days,
          delta=f"{extreme_days - hist_extreme} vs last 5 yrs")

# Objective 2: Infiltration check
recent_infil = lulc_station.tail(365*5)['soil_infiltration_mmhr'].mean()
col2.metric("Avg Soil Infiltration", f"{recent_infil:.1f} mm/hr",
            delta=f"{urban_change:.2f}x urban factor", delta_color="inverse")

# Objective 1: Channel width if available
if 'channel_width_m' in forecast.columns and 'channel_width_m' in hist.columns:
    width_change = forecast['channel_width_m'].iloc[-1] - hist['channel_width_m'].iloc[-1]
    col3.metric("Channel Width 2030", f"{forecast['channel_width_m'].iloc[-1]:.1f} m",
                delta=f"{width_change:+.1f} m")
else:
    col3.metric("Channel Width", "N/A", help="Add 'channel_width_m' to your CSV for Obj 1")

with st.expander("Data preview + debug info"):
    st.write("**Hydrology rows:**", len(hydro_df), "| **LULC rows:**", len(lulc_df))
    st.write("**Date range:**", hydro_df['date'].min().date(), "to", hydro_df['date'].max().date())
    st.write("**Stations found:**", list(stations))
    st.write("**Hydrology columns:**", list(hydro_df.columns))
    st.write("**LULC columns:**", list(lulc_df.columns))
