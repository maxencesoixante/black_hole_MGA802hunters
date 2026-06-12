"""
main.py  —  Automated Astronomical Anomaly Detector
MGA 802 : Introduction to Programming with Python

This script is the "conductor" of the entire pipeline.  It ties together the
three modules written by each team member and runs them in sequence:

    1. Maxence  →  data_handler.py   :  fetch + clean the light curve
    2. Jules    →  anomaly_engine.py :  detect + classify anomalies
    3. Alexandre →  visualizer.py    :  display the annotated plot

How to run:
    python main.py
Then type the Kepler/TESS identifier when prompted (e.g.  KIC 11904151).
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# We import only what we need from each module.
# The  from X import Y  syntax makes Y available directly without the X. prefix.
from astro_module.data_handler  import AstroFetcher, SignalCleaner
from astro_module.anomaly_engine import AnomalyDetector, AnomalyClassifier
from astro_module.visualizer    import AstroPlotter


# ── Pipeline configuration ─────────────────────────────────────────────────────
# Defining thresholds here (not buried inside the classes) makes them easy to
# tweak for a different target without hunting through multiple files.
WINDOW          = 200   # rolling-window size in data points (~4 days of Kepler data)
SIGMA_THRESHOLD = 3.0   # detection sensitivity: flag points more than 3σ from the local mean


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Welcome banner
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("   Automated Astronomical Anomaly Detector")
print("   MGA 802  —  Session Project  (Maxence, Jules, Alexandre)")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  User interaction — with retry loop
# ─────────────────────────────────────────────────────────────────────────────
# We wrap the input + download inside a while True loop so that, if the target
# is not found, the user can immediately try a different one without restarting
# the whole programme.
#
# How the loop works:
#   • 'while True' runs forever until Python hits a 'break' statement.
#   • If the download succeeds, we 'break' out of the loop and continue.
#   • If it fails (ValueError), we print the error and loop back to ask again.

raw_data  = None
target_id = None

while True:
    # input() pauses and waits for the user to type something and press Enter.
    # .strip() removes accidental leading/trailing whitespace from the input.
    target_id = input(
        "\nEnter the star identifier to analyse\n"
        "(e.g.  KIC 11904151  —  or type 'quit' to exit): "
    ).strip()

    # Allow the user to exit the programme gracefully without a crash.
    if target_id.lower() in ('quit', 'exit', 'q'):
        print("Exiting. Goodbye!")
        raise SystemExit(0)

    print()
    print("[Step 1/3]  Fetching and cleaning data...")
    print("-" * 40)

    # try / except is Python's error-handling mechanism:
    #   • The code inside 'try' runs normally.
    #   • If a ValueError is raised (target not found), execution jumps to 'except'
    #     and the programme does NOT crash — it just prints the message and loops.
    try:
        # AstroFetcher contacts the NASA MAST archive.  If the target cannot be
        # found on Kepler it automatically tries TESS before giving up.
        fetcher  = AstroFetcher(target_id)
        raw_data = fetcher.download_data(mission='Kepler')

        # SignalCleaner removes NaNs, sigma-clips outliers (5σ), and normalises
        # the flux around 1.0 so values are comparable across different targets.
        clean_data = SignalCleaner(raw_data).process_data()

        # If we reach this line, the download and cleaning both succeeded.
        # 'break' exits the while loop and the pipeline continues below.
        break

    except ValueError as e:
        # str(e) converts the exception object to the human-readable message
        # we wrote in data_handler.py's raise ValueError(...).
        print(f"\n[Error] {e}")
        print("Please try a different identifier.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Jules — Anomaly detection and classification
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 2/3]  Detecting and classifying anomalies...")
print("-" * 40)

# AnomalyDetector flags every cadence that lies outside ±SIGMA_THRESHOLD × local std.
# It adds three columns to the DataFrame: rolling_mean, rolling_std, is_anomaly.
detector   = AnomalyDetector(clean_data, window=WINDOW, sigma_threshold=SIGMA_THRESHOLD)
df_flagged = detector.detect()

# get_anomaly_segments() groups nearby anomalous points into event windows,
# each padded with context on both sides so the classifier sees the baseline.
segments = detector.get_anomaly_segments()

# AnomalyClassifier analyses each segment's shape and assigns it to one of:
#   TRANSIT  /  FLARE  /  MICROLENSING  /  NOISE (NOISE is filtered out)
classified_results = AnomalyClassifier().classify_all(segments)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Alexandre — Visualisation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 3/3]  Generating visualisation...")
print("-" * 40)

# AstroPlotter takes the fully flagged DataFrame and the classified events,
# and produces an annotated professional figure.
plotter = AstroPlotter(
    df_flagged         = df_flagged,
    classified_results = classified_results,
    target_id          = target_id,
    sigma_threshold    = SIGMA_THRESHOLD,
    window             = WINDOW,
)
plotter.show_results()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"   Analysis of  {target_id}  complete.")
print(f"   {int(df_flagged['is_anomaly'].sum())} anomalous cadence(s) flagged.")
print(f"   {len(classified_results)} astrophysical event(s) classified:")
for i, event in enumerate(classified_results, start=1):
    # :.0%  formats a float as a percentage with 0 decimal places (e.g. 0.87 → "87%")
    print(f"     {i}. {event['event_type']}  (confidence: {event['confidence']:.0%})")
print("=" * 60)
