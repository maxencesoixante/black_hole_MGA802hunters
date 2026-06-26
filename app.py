"""
app.py  —  Streamlit Interface for Automated Astronomical Anomaly Detector
MGA 802 : Introduction to Programming with Python

This script provides a Streamlit-based web interface for the anomaly detection pipeline.

How to run:
    streamlit run app.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import streamlit as st
from astro_module.data_handler import AstroFetcher, SignalCleaner
from astro_module.anomaly_engine import AnomalyDetector
from astro_module.visualizer import AstroPlotter

# ── Pipeline configuration ─────────────────────────────────────────────────────
# Default values for the pipeline parameters
default_window_streamlit = 200
default_baseline_sigma_streamlit = 1.0
default_min_duration_days_streamlit = 0.1
default_trim_edges_days_streamlit = 0.5
default_flare_asymmetry_threshold = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Astronomical anomaly detector",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 Automated astronomical anomaly detector 🌠")
st.markdown("""
    **MGA 802 — Session Project (Maxence, Jules, Alexandre)**

    This app detects transits in astronomical light curves using a rolling-window baseline departure method.
""")

# Sidebar for user input
st.sidebar.header("Input Parameters")

# Options for demo
options_demo = ['KIC 11904151','KIC 11446443', 'HAT-P-7', 'Kepler-8', 'TIC 100100827', 'TIC 261136679']

# Exclusive choice between manual input and dropdown menu for mission
input_method = st.sidebar.radio(
    "Choose input method for mission:",
    ["Manual Input", "Dropdown Menu preset examples"]
)

if input_method == "Manual Input":
    target_id = st.sidebar.text_input(
        "Enter the star identifier to analyse (e.g., KIC 11904151):",
        value="KIC 11904151"
    )
else:
    choix_demo = st.sidebar.selectbox(
        "Choose a mission for the demo:", options_demo
    )
    target_id = choix_demo

# Allow the user to adjust parameters
window_streamlit = st.sidebar.slider("Rolling-window size (data points)", 50, 500, default_window_streamlit,
                                     key="rolling_window_slider")
baseline_sigma_streamlit = st.sidebar.slider("Baseline sigma threshold", 0.5, 3.0, default_baseline_sigma_streamlit,
                                             key="sigma_slider")
min_duration_days_streamlit = st.sidebar.slider("Minimum duration (days)", 0.05, 1.0,
                                                default_min_duration_days_streamlit, key="duration_slider")
trim_edges_days_streamlit = st.sidebar.slider("Trim edges (days)", 0.0, 2.0, default_trim_edges_days_streamlit,
                                              key="trim_slider")
flare_asymmetry_threshold_streamlit = st.sidebar.slider("Flare asymmetry threshold", -1.0, 1.0,
                                                        default_flare_asymmetry_threshold, key="flare_slider")

# Add the possibility to upload a local CSV file
uploaded_file = st.sidebar.file_uploader("Upload a local CSV file (optional)", type=["csv"])

def reset_sliders():
    st.session_state.rolling_window_slider = default_window_streamlit
    st.session_state.sigma_slider = default_baseline_sigma_streamlit
    st.session_state.duration_slider = default_min_duration_days_streamlit
    st.session_state.trim_slider = default_trim_edges_days_streamlit
    st.session_state.flare_slider = default_flare_asymmetry_threshold

reset_default_values_button = st.sidebar.button("Reset the sliders to the default values",
                                        on_click= reset_sliders)

# Main content area
if st.sidebar.button("Run Analysis"):
    st.markdown("---")
    st.header("Analysis Results")

    # Step 1: Fetch and clean data
    st.subheader("Step 1/3: Fetching and cleaning data...")
    try:
        if uploaded_file is not None:
            # Load the local CSV file
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            if 'time' not in df.columns or 'flux' not in df.columns:
                st.error("Error: The CSV file must contain 'time' and 'flux' columns.")
                st.stop()
            clean_data = SignalCleaner(df).process_data(trim_edges_days=trim_edges_days_streamlit)
            st.success("Local CSV file loaded and cleaned successfully!")
        else:
            # Use the mission ID
            fetcher = AstroFetcher(target_id)
            raw_data = fetcher.download_data(mission='Kepler')
            clean_data = SignalCleaner(raw_data).process_data(trim_edges_days=trim_edges_days_streamlit)
            st.success("Data fetched and cleaned successfully!")
    except ValueError as e:
        st.error(f"Error: {e}")
        st.stop()

    # Step 2: Detect transits
    st.subheader("Step 2/3: Detecting transits...")
    detector = AnomalyDetector(clean_data, window=window_streamlit)
    detector.FLARE_ASYMMETRY_THRESHOLD = flare_asymmetry_threshold_streamlit
    transits = detector.detect_baseline_departures(
        baseline_sigma=baseline_sigma_streamlit,
        min_duration_days=min_duration_days_streamlit,
    )
    df_flagged = detector.df
    st.success(f"Detected {len(transits)} anomalies.")

    # Step 3: Visualize results
    st.subheader("Step 3/3: Visualization")
    plotter = AstroPlotter(
        df_flagged=df_flagged,
        classified_results=transits,
        target_id=target_id,
        window=window_streamlit,
        baseline_sigma=detector.baseline_sigma,
        global_median=detector.global_median,
        global_std=detector.global_std,
    )
    plotter.set_streamlit()
    fig = plotter.show_results()
    st.pyplot(fig)
    plt.close(fig)  # Free memory after displaying the plot
    # Summary
    st.markdown("---")
    st.subheader("Summary")
    n_transits = sum(1 for ev in transits if ev['event_type'] == 'TRANSIT')
    st.write(f"**Analysis of {target_id} complete.**")
    st.write(f"**{n_transits} transit(s) detected via rolling-window baseline departure:**")
    for i, ev in enumerate(transits, start=1):
        st.write(
            f"{i}. {ev['event_type']} at t={ev['peak_time']:.2f} d "
            f"(depth: {ev['depth_sigma']}σ / {ev['depth_frac'] * 100:.3f}%, "
            f"duration: {ev['duration_days']} d, confidence: {ev['confidence']:.0%})"
        )

# Footer
st.markdown("---")
st.markdown("""
    **About:**
    This app is part of the MGA 802 project. It uses a rolling-window baseline departure method to detect transits in astronomical light curves.
    
    **Source Code:** [GitHub Repository](https://github.com/maxencesoixante/black_hole_MGA802hunters) 🚀
    
    It is inspired and designed to help the Black Hole Hunters community from Adler, University of Minnesota and University of Oxford
    
    **More info:** [Black Hole Hunters website](https://www.zooniverse.org/projects/cobalt-lensing/black-hole-hunters) 🌌
""")