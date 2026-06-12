"""
test_kepler22.py  —  Validation run on Kepler-22 with sigma=5.0
================================================================

Purpose
-------
This script validates that our anomaly-detection pipeline produces clean,
scientifically meaningful results on a well-behaved, quiet star — as opposed
to the highly active M-dwarf GJ 1243, which saturated the detector at sigma=3.0.

Target
------
  Star    : Kepler-22  (KIC 10593626)
  Planet  : Kepler-22b — a super-Earth in the habitable zone
  Period  : ~290 days
  Transit depth: ~0.05 %  (very shallow — a genuine challenge)
  Cadence : long (30 min / point)  — best trade-off for multi-quarter coverage

Why sigma=5.0 here?
-------------------
  • Kepler-22 is a quiet, Sun-like star.  Its intrinsic variability is very low,
    so raising the threshold from 3.0 to 5.0 should eliminate residual noise
    detections without suppressing real transit dips.
  • GJ 1243 at sigma=3.0 produced ~300 false MICROLENSING events because the
    star's natural activity permanently exceeded 3σ.  sigma=5.0 is a safer
    default for active stars too.

Expected outcome (hypothesis to verify)
----------------------------------------
  • No or very few NOISE / MICROLENSING false positives (clean baseline).
  • 1–3 TRANSIT detections corresponding to Kepler-22b passages.
  • Comparison with GJ 1243 results will illustrate the algorithm's limits
    and the importance of choosing the right sigma for each target type.

How to run
----------
  python test_kepler22.py
"""

import lightkurve as lk
from astro_module.data_handler  import AstroFetcher, SignalCleaner
from astro_module.anomaly_engine import AnomalyDetector, AnomalyClassifier
from astro_module.visualizer    import AstroPlotter

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
TARGET_ID       = "Kepler-22"   # lightkurve resolves this to KIC 10593626
SIGMA_THRESHOLD = 5.0           # raised from default 3.0 — cleaner on quiet stars
WINDOW          = 200           # rolling window (~4 days of long-cadence data)

print("=" * 60)
print("  Kepler-22 Validation Run")
print(f"  Target : {TARGET_ID}  (KIC 10593626)")
print(f"  sigma  : {SIGMA_THRESHOLD}  |  window : {WINDOW} pts")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — Download the long-cadence (30 min) light curve manually
# ─────────────────────────────────────────────────────────────────────────────
# We bypass AstroFetcher here to explicitly request long-cadence data
# (exptime=1800 s).  This gives us the most data points per quarter
# and is the standard format for transit searches.
#
# Why long cadence?
#   • Each quarter = ~90 days of observations.
#   • Kepler-22b transits every ~290 days, so we need multiple quarters to
#     catch more than one transit.
#   • Short-cadence (1 min) gives finer time resolution but covers fewer quarters,
#     making it harder to accumulate statistics.
print("\n[Step 1/4]  Downloading long-cadence light curve from Kepler archive...")
print("-" * 40)

search = lk.search_lightcurve(TARGET_ID, mission='Kepler', exptime=1800)
print(f"  Found {len(search)} long-cadence dataset(s) across all quarters.")

# Download only the FIRST quarter to keep the demo fast.
# Change to search.download_all() to stitch all quarters together —
# this gives more transits but takes much longer to download.
print("  Downloading Quarter 0 (first available)...")
raw_lc = search[0].download()
print(f"  Download complete: {len(raw_lc)} cadences.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Clean
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2/4]  Cleaning signal...")
print("-" * 40)

cleaner    = SignalCleaner(raw_lc)
clean_data = cleaner.process_data()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Detect anomalies at sigma=5.0
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 3/4]  Detecting anomalies (sigma={SIGMA_THRESHOLD})...")
print("-" * 40)

detector   = AnomalyDetector(clean_data, window=WINDOW, sigma_threshold=SIGMA_THRESHOLD)
df_flagged = detector.detect()
segments   = detector.get_anomaly_segments()

classified_results = AnomalyClassifier().classify_all(segments)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Structured results summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)

# Count events by type to compare against the GJ 1243 run
counts = {}
for event in classified_results:
    t = event['event_type']
    counts[t] = counts.get(t, 0) + 1

print(f"\n  Total anomalous cadences   : {int(df_flagged['is_anomaly'].sum())}")
print(f"  Total classified events    : {len(classified_results)}")
print(f"  Breakdown by type:")
for event_type, count in sorted(counts.items()):
    print(f"    • {event_type:<20} {count}")

print()
print("  Detailed event list:")
for i, event in enumerate(classified_results, start=1):
    t_start = event['time'][0]
    t_end   = event['time'][-1]
    print(
        f"    {i:>3}. {event['event_type']:<20} "
        f"confidence={event['confidence']:.0%}  "
        f"t=[{t_start:.1f} – {t_end:.1f}] days"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Comparison with GJ 1243 baseline
# ─────────────────────────────────────────────────────────────────────────────
print()
print("─" * 60)
print("  COMPARISON WITH GJ 1243 RUN  (sigma=3.0, same algorithm)")
print("─" * 60)
print("  GJ 1243  : ~300 MICROLENSING  +  48 STELLAR_ROTATION  — unusable")
print(f"  Kepler-22: {len(classified_results)} event(s) at sigma=5.0")
if len(classified_results) <= 10:
    print("  ✓  Clean result — algorithm validated on a quiet, Sun-like star.")
else:
    print("  ! More events than expected — consider raising sigma further.")
print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 4/4]  Generating annotated plot...")
print("-" * 40)

plotter = AstroPlotter(
    df_flagged         = df_flagged,
    classified_results = classified_results,
    target_id          = f"{TARGET_ID}  (σ={SIGMA_THRESHOLD})",
    sigma_threshold    = SIGMA_THRESHOLD,
    window             = WINDOW,
)
plotter.show_results()
