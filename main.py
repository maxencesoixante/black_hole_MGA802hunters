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
from astro_module.anomaly_engine import AnomalyDetector
from astro_module.visualizer    import AstroPlotter


# ── Pipeline configuration ─────────────────────────────────────────────────────
# Defining thresholds here (not buried inside the classes) makes them easy to
# tweak for a different target without hunting through multiple files.
#
# Detection is now driven SOLELY by the rolling-window baseline-departure method
# (detect_baseline_departures): a transit is a sustained drift of the rolling
# mean away from the global baseline.  The old point-by-point sigma-clip channel
# is no longer used.
WINDOW            = 200   # rolling-window size in data points
BASELINE_SIGMA    = 1.0   # how far (in global σ) the rolling mean must drift to flag a transit
MIN_DURATION_DAYS = 0.1   # minimum duration of a drift to count (~2.4 h — filters blips)
TRIM_EDGES_DAYS   = 0.5   # drop the first/last ~12 h of each quarter (thermal-ramp guard)


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

        # SignalCleaner removes NaNs, clips UPWARD outliers (asymmetric 5σ),
        # normalises the flux around 1.0, and trims the quarter edges to discard
        # the start-of-quarter thermal ramp.
        clean_data = SignalCleaner(raw_data).process_data(trim_edges_days=TRIM_EDGES_DAYS)

        # If we reach this line, the download and cleaning both succeeded.
        # 'break' exits the while loop and the pipeline continues below.
        break

    except ValueError as e:
        # str(e) converts the exception object to the human-readable message
        # we wrote in data_handler.py's raise ValueError(...).
        print(f"\n[Error] {e}")
        print("Please try a different identifier.\n")




# ─────────────────────────────────────────────────────────────────────────────
# 3.  Transit detection  (rolling-window baseline departure)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 2/3]  Detecting transits (rolling-window baseline departure)...")
print("-" * 40)

# Detection is driven entirely by the rolling window: detect_baseline_departures()
# computes the rolling-mean baseline and flags every sustained drift of that
# baseline away from the star's global level.  A sustained DIP is a transit.
# (The old point-by-point sigma-clip channel is intentionally no longer used.)
detector = AnomalyDetector(clean_data, window=WINDOW)
transits = detector.detect_baseline_departures(
    baseline_sigma    = BASELINE_SIGMA,
    min_duration_days = MIN_DURATION_DAYS,
)
df_flagged = detector.df   # carries the 'rolling_mean' column the plotter needs


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Visualisation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 3/3]  Generating visualisation...")
print("-" * 40)

# The plot is based purely on the rolling-window result: no point-anomaly markers,
# no local ±σ band — instead the global baseline envelope plus a diamond on each
# detected transit.
plotter = AstroPlotter(
    df_flagged         = df_flagged,
    classified_results = transits,
    target_id          = target_id,
    window             = WINDOW,
    baseline_sigma     = detector.baseline_sigma,
    global_median      = detector.global_median,
    global_std         = detector.global_std,
)
plotter.show_results()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Summary
# ─────────────────────────────────────────────────────────────────────────────
counts = {t: sum(1 for ev in transits if ev['event_type'] == t)
          for t in ('TRANSIT', 'FLARE', 'MICROLENSING')}
print("\n" + "=" * 60)
print(f"   Analysis of  {target_id}  complete.")
print(f"   {len(transits)} event(s) detected via rolling-window baseline departure")
print(f"   ({counts['TRANSIT']} transit, {counts['FLARE']} flare, "
      f"{counts['MICROLENSING']} microlensing):")
for i, ev in enumerate(transits, start=1):
    print(f"     {i}. {ev['event_type']}  at t={ev['peak_time']:.2f} d  "
          f"(depth: {ev['depth_sigma']}σ / {ev['depth_frac'] * 100:.3f}%, "
          f"duration: {ev['duration_days']} d, confidence: {ev['confidence']:.0%})")
print("=" * 60)
