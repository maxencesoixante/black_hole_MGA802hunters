import pandas as pd
import numpy as np
# Pandas  → tabular data: DataFrames with named columns and integer row indices
# NumPy   → fast numerical operations on arrays (mathematical functions, statistics)


class AnomalyDetector:
    """
    Detects anomalies in a cleaned light curve using a rolling window approach.
    An anomaly is flagged when the flux deviates significantly from the local
    rolling mean, based on a dynamic threshold (N × local std).

    ── Core Concept: Rolling Window ─────────────────────────────────────────────
    Instead of computing a single average for the ENTIRE time series (global mean),
    a rolling window slides along the data point by point, computing a LOCAL
    mean and standard deviation using only the neighbouring points inside the window.

    Why local statistics?
    Because a star's baseline brightness can drift slowly over months of observation
    (instrumental focus changes, detector ageing, etc.).  A global mean would be
    blind to this drift.  A local mean tracks it and adapts.

    Visual example (window = 5, values = [1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0]):
        At index 3, the local mean ≈ 1.6 and local std ≈ 1.8.
        Point 5.0 is (5.0 − 1.6) / 1.8 ≈ 1.9σ above the mean.
        With a threshold of 3.0σ it would NOT be flagged;
        with a threshold of 1.5σ it WOULD be flagged.
    ─────────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, df: pd.DataFrame, window: int = 200, sigma_threshold: float = 3.0):
        """
        Initialise the detector with the cleaned light curve and hyperparameters.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned light curve — output of SignalCleaner.process_data().
            Must have 'time' and 'flux' columns.
        window : int
            Number of data points in each rolling window.
            200 points ≈ 4 days of Kepler data (one cadence every ~30 min).
            Larger window  → smoother baseline, less sensitive to long-duration events.
            Smaller window → noisier baseline, more false positives.
        sigma_threshold : float
            Number of standard deviations a point must exceed to be flagged.
            3.0σ means there is less than a 0.3% chance a genuine noise point
            is wrongly flagged (assuming Gaussian noise).

        ── OOP reminder: self ───────────────────────────────────────────────────
        Every line "self.X = value" stores that value ON the object so any
        method defined below can access it via self.X without needing it
        passed as an argument each time.
        ─────────────────────────────────────────────────────────────────────────
        """
        # Guard: raise a clear error early rather than crashing later with a
        # confusing KeyError when we try to access missing columns.
        if 'time' not in df.columns or 'flux' not in df.columns:
            raise ValueError("DataFrame must contain 'time' and 'flux' columns.")

        # .copy() prevents the original DataFrame (from SignalCleaner) from being
        # modified accidentally — good defensive programming practice.
        # reset_index(drop=True) ensures the row index is 0, 1, 2, … with no gaps,
        # which is critical when we later use index arithmetic for segmentation.
        self.df = df.copy().reset_index(drop=True)

        # Store the hyperparameters so every method can read them and they are
        # visible to the user for reproducibility.
        self.window           = window
        self.sigma_threshold  = sigma_threshold

    def detect(self) -> pd.DataFrame:
        """
        Run the rolling-window anomaly detection algorithm.

        For each data point the algorithm:
          1. Looks at the surrounding 'window' points (centred).
          2. Computes their LOCAL mean (baseline) and std (variability).
          3. Flags the point as an anomaly if it falls outside
             [rolling_mean  ±  sigma_threshold × rolling_std].

        Returns
        -------
        pd.DataFrame
            The input DataFrame enriched with three new columns:
            - 'rolling_mean' : local baseline estimate
            - 'rolling_std'  : local variability estimate
            - 'is_anomaly'   : True if the point is outside the envelope
        """
        print(f"Running anomaly detection (window={self.window}, sigma={self.sigma_threshold})...")

        # ── Rolling mean (local baseline) ────────────────────────────────────
        # .rolling() creates a "sliding window view" of the Series:
        #   window      = number of points to include in each window position
        #   center=True = the window is centred on the current point, so it looks
        #                 equally far to the LEFT and to the RIGHT.
        #                 Without this, the window only looks backward (causal),
        #                 which would create a lag and mis-place the baseline.
        #   min_periods = minimum number of non-NaN values required to compute
        #                 a result.  Setting it to 1 means we always get a value,
        #                 even at the edges where fewer neighbours exist.
        # .mean() then computes the arithmetic mean inside each window position.
        self.df['rolling_mean'] = (
            self.df['flux']
            .rolling(window=self.window, center=True, min_periods=1)
            .mean()
        )

        # ── Rolling standard deviation (local variability) ───────────────────
        # Same sliding window but computing the standard deviation instead.
        # A large rolling_std means the star is naturally variable in this region
        # (e.g. stellar pulsations), so the anomaly threshold is automatically
        # widened — the algorithm self-calibrates to the local noise level.
        self.df['rolling_std'] = (
            self.df['flux']
            .rolling(window=self.window, center=True, min_periods=1)
            .std()
        )

        # ── Dynamic detection envelope ────────────────────────────────────────
        # upper and lower are Pandas Series (one value per row of the DataFrame).
        # They define a "band" around the baseline within which normal points sit.
        # Points outside this band are anomalies.
        upper = self.df['rolling_mean'] + self.sigma_threshold * self.df['rolling_std']
        lower = self.df['rolling_mean'] - self.sigma_threshold * self.df['rolling_std']

        # ── Boolean anomaly flag ──────────────────────────────────────────────
        # Comparison operators (<, >) applied to a Pandas Series produce a
        # Series of True/False values (one per row).
        # The | operator is element-wise OR: True if EITHER condition is True.
        # Result: is_anomaly = True when flux is outside the detection envelope.
        self.df['is_anomaly'] = (self.df['flux'] > upper) | (self.df['flux'] < lower)

        # .sum() on a boolean Series counts the True values (True = 1, False = 0).
        n_anomalies = self.df['is_anomaly'].sum()
        print(f"Detection complete. {n_anomalies} anomalous point(s) found.")

        return self.df

    def get_anomaly_segments(self, min_gap_days: float = 0.5) -> list[dict]:
        """
        Group individual anomalous points into physically meaningful event segments.

        Instead of grouping by index distance, we group by TIME gap — this
        correctly merges all points belonging to the same transit (~7-15 hours)
        even when they are not strictly consecutive in index.

        Parameters
        ----------
        min_gap_days : float
            Maximum time gap (in days) between two anomalous points for them
            to be considered part of the SAME event.
            Default 0.5 days (12 h) comfortably covers a full Kepler transit
            window of ~6-10 hours.

        Returns
        -------
        list of dict, each containing:
            'start_idx'        : first row index of the padded window
            'end_idx'          : last row index of the padded window
            'time'             : NumPy array of timestamps for the window
            'flux'             : NumPy array of flux values for the window
            'anomaly_peak_idx' : row index of the most deviant point
        """
        # Guard: detect() must be called first because it creates 'is_anomaly'.
        if 'is_anomaly' not in self.df.columns:
            raise RuntimeError("Run detect() before calling get_anomaly_segments().")

        # self.df.index[mask] extracts the INTEGER row indices where mask is True.
        # .tolist() converts the resulting Index object to a plain Python list.
        anomaly_indices = self.df.index[self.df['is_anomaly']].tolist()

        if not anomaly_indices:
            print("No anomaly segments to extract.")
            return []

        # Retrieve the actual time values for all anomalous points.
        # We compare TIME gaps (in days), not index distances, because the
        # cadence is not always perfectly uniform (observation gaps exist).
        # .values returns a plain NumPy array — fast and easy to index.
        times = self.df.loc[anomaly_indices, 'time'].values

        segments        = []                      # completed event groups
        current_segment = [anomaly_indices[0]]    # start building the first group

        # Walk through each anomalous point (starting from the second one).
        for i in range(1, len(anomaly_indices)):
            idx      = anomaly_indices[i]
            time_gap = times[i] - times[i - 1]   # gap between this and previous point (days)

            if time_gap <= min_gap_days:
                # Gap is small enough → same physical event; extend current group.
                current_segment.append(idx)
            else:
                # Gap is too large → different event; save the finished group and start a new one.
                segments.append(current_segment)
                current_segment = [idx]

        # The loop ends before saving the last group — append it manually.
        segments.append(current_segment)

        # ── Pad each segment with context points ──────────────────────────────
        # We add "context" points on each side so the classifier can see the
        # quiet baseline before and after the event — crucial for measuring asymmetry
        # (how the flux rises vs. how it decays).
        context = self.window // 2   # // is integer division: 200 // 2 = 100 points ≈ 2 days

        result = []
        for seg in segments:
            # Clamp start/end to valid DataFrame boundaries with max() and min().
            # Without clamping, segments at the very beginning or end of the
            # time series could request out-of-range indices.
            start = max(0, seg[0] - context)                    # don't go below index 0
            end   = min(len(self.df) - 1, seg[-1] + context)   # don't exceed the last row

            # .loc[start:end] selects rows from 'start' to 'end' inclusive.
            segment_df = self.df.loc[start:end]

            # Find the strongest deviant point within the anomalous sub-group (seg).
            # np.abs() ensures dips (negative deviations) are treated equally to spikes.
            deviations = np.abs(
                self.df.loc[seg, 'flux'].values - self.df.loc[seg, 'rolling_mean'].values
            )
            # np.argmax() returns the POSITION of the maximum value in the array.
            # We convert it back to the original DataFrame row index via seg[position].
            peak_idx = seg[int(np.argmax(deviations))]

            result.append({
                'start_idx'        : start,
                'end_idx'          : end,
                'time'             : segment_df['time'].values,   # NumPy array
                'flux'             : segment_df['flux'].values,   # NumPy array
                'anomaly_peak_idx' : peak_idx
            })

        print(f"{len(result)} anomaly segment(s) extracted.")
        return result


class AnomalyClassifier:
    """
    Classifies an anomaly segment into one of four categories:

        TRANSIT      : U-shaped dip — an exoplanet crosses in front of the star
                       and blocks part of its light, causing a symmetric decrease.
        FLARE        : fast-rise slow-decay spike — a violent magnetic eruption
                       on the stellar surface releases a sudden burst of energy.
        MICROLENSING : symmetric bell-shaped brightening over several days —
                       a massive foreground object (e.g. a black hole or neutron
                       star) acts as a gravitational lens and temporarily focuses
                       the star's light toward us.
        NOISE        : signal too weak or too short to be a real astrophysical event.

    ── Classification Logic ─────────────────────────────────────────────────────
    Three hand-crafted features are extracted from each segment:

      Feature 1 — Direction   : is the peak a DIP (below baseline) or a brightening?
      Feature 2 — Asymmetry   : does the signal RISE faster than it DECAYS?
      Feature 3 — Amplitude + Duration : is the signal strong AND long enough?

    Decision rules applied in ORDER (first match wins):
      1. Peak is a dip below DIP_THRESHOLD                 → TRANSIT
      2. Rise slope >> Decay slope  (asymmetric profile)   → FLARE
      3. Too weak  OR  too short                           → NOISE
      4. Everything else  (symmetric brightening)          → MICROLENSING
    ─────────────────────────────────────────────────────────────────────────────
    """

    # ── Class-level constants ─────────────────────────────────────────────────
    # These are shared by ALL instances of AnomalyClassifier.
    # Defining them at class level (not inside __init__) makes them easy to
    # inspect and override without creating an object first.

    DIP_THRESHOLD = -0.0008
    # Normalised flux deviation below which we label the event a dip.
    # After median normalisation, a quiet star has flux ≈ 1.0.
    # A deviation of -0.0008 means the star is 0.08% dimmer — a conservative
    # threshold that catches even very shallow exoplanet transits.

    ASYMMETRY_THRESHOLD = 0.30
    # Slope asymmetry ratio above which we classify the event as a FLARE.
    # Flares have a characteristic "shark-fin" profile: nearly instantaneous rise
    # (triggered by magnetic reconnection) followed by a slow exponential decay.
    # 0.30 means the rise slope must be at least 30% larger than the decay slope.

    MIN_MICROLENSING_AMPLITUDE = 0.005
    # Minimum normalised flux amplitude for a MICROLENSING classification.
    # 0.005 means the star must brighten by at least 0.5% above its median.
    # Below this value the event is indistinguishable from residual detector noise.

    MIN_MICROLENSING_DURATION = 0.3
    # Minimum duration in days (~7 hours) for a MICROLENSING classification.
    # Real microlensing events from stellar-mass objects last days to weeks;
    # anything shorter is almost certainly an instrumental artefact.

    def classify_segment(self, segment: dict) -> dict:
        """
        Classify a single anomaly segment into one event type.

        Parameters
        ----------
        segment : dict
            One element from AnomalyDetector.get_anomaly_segments().
            Must contain 'flux' and 'time' keys (NumPy arrays).

        Returns
        -------
        dict with keys:
            'event_type'  : str   — 'TRANSIT', 'FLARE', 'MICROLENSING', or 'NOISE'
            'confidence'  : float — value in [0, 1] estimating classification certainty
            'description' : str   — human-readable explanation of the decision
        """
        flux = segment['flux']   # 1-D NumPy array of normalised flux values
        time = segment['time']   # 1-D NumPy array of timestamps in days

        # A segment with fewer than 5 points gives us no meaningful shape features.
        # Return NOISE immediately to avoid crashes in the subsequent calculations.
        if len(flux) < 5:
            return {
                'event_type' : 'NOISE',
                'confidence' : 0.0,
                'description': 'Segment too short to classify.'
            }

        # ── Local re-normalisation ────────────────────────────────────────────
        # Although the full light curve is already normalised around 1.0,
        # we re-normalise WITHIN the segment so the local baseline becomes 0.0.
        # After this step:
        #   norm_flux = +0.01  →  star is 1% brighter than segment median
        #   norm_flux = -0.01  →  star is 1% dimmer than segment median
        # This makes peak_value directly interpretable as a fractional deviation.
        baseline  = np.median(flux)
        norm_flux = (flux - baseline) / (np.abs(baseline) + 1e-10)
        # 1e-10 (= 0.0000000001) guards against division-by-zero if baseline ≈ 0.

        # ── Feature 1: Direction ──────────────────────────────────────────────
        # Find the index of the point with the LARGEST absolute deviation from 0.
        # np.abs() makes dips (negative) and spikes (positive) comparable.
        # np.argmax() returns the POSITION of the largest value in the array.
        peak_idx   = int(np.argmax(np.abs(norm_flux)))
        peak_value = norm_flux[peak_idx]   # signed: negative = dip, positive = brightening

        # is_dip is True when the peak is a downward deviation below DIP_THRESHOLD.
        # A DIP means the star got dimmer → could be a planet blocking part of the light.
        is_dip = peak_value < self.DIP_THRESHOLD

        # ── Feature 2: Slope-based asymmetry ─────────────────────────────────
        # Split the segment at the peak into two halves:
        #   left_flux  = the "approach" side (how flux changes leading up to the peak)
        #   right_flux = the "departure" side (how flux changes after the peak)
        left_flux  = norm_flux[:peak_idx + 1]   # from start of segment to peak (inclusive)
        right_flux = norm_flux[peak_idx:]        # from peak to end of segment (inclusive)

        # np.diff([a, b, c]) → [b-a, c-b]: differences between consecutive elements.
        # np.abs() converts all differences to positive magnitudes.
        # np.mean() gives the average step size = how steeply the flux changes per point.
        rise_slope  = np.mean(np.abs(np.diff(left_flux)))  if len(left_flux)  > 1 else 0.0
        decay_slope = np.mean(np.abs(np.diff(right_flux))) if len(right_flux) > 1 else 0.0

        # Normalised asymmetry score ∈ [-1, +1]:
        #   +1  → instantaneous rise, infinitely slow decay  (perfect flare shape)
        #    0  → perfectly symmetric (consistent with transit or microlensing)
        #   -1  → instantaneous decay, infinitely slow rise
        # slope_sum + 1e-10 prevents division by zero if both slopes are 0.
        slope_sum       = rise_slope + decay_slope + 1e-10
        slope_asymmetry = (rise_slope - decay_slope) / slope_sum

        # ── Feature 3: Sharpness ──────────────────────────────────────────────
        # Compare the peak height to the average flux at the EDGES of the segment.
        # norm_flux[[0, -1]] selects the first AND last element of the array.
        # A large sharpness ratio means the event is very concentrated at the centre
        # relative to the quiet wings — typical of transits and flares.
        wing_mean = np.mean(np.abs(norm_flux[[0, -1]])) + 1e-10
        sharpness = np.abs(peak_value) / wing_mean

        # ── Feature 4: Duration and amplitude ────────────────────────────────
        # time[-1] is the last timestamp; time[0] is the first — their difference
        # gives the total span of the segment window in days.
        duration_days = float(time[-1] - time[0])
        amplitude     = float(np.abs(peak_value))    # absolute normalised peak height

        # ── Decision tree (first matching rule wins) ──────────────────────────

        if is_dip:
            # Peak is a downward deviation → most likely an exoplanet blocking the star.
            event_type = 'TRANSIT'

            # Confidence scales with symmetry: a perfectly symmetric U-shape (sym = 1)
            # gets the highest confidence because transits are geometrically symmetric.
            # np.clip(value, min, max) clamps the result into the [0.4, 0.99] range
            # so we never output a confidence of 0 or 1 (never absolutely certain).
            sym        = 1.0 - abs(slope_asymmetry)   # 1.0 = perfectly symmetric
            confidence = round(float(np.clip(0.5 + sym * 0.49, 0.4, 0.99)), 2)
            description = (
                f"U-shaped flux dip detected "
                f"(slope asymmetry={slope_asymmetry:.2f}, sharpness={sharpness:.1f}, "
                f"duration={duration_days:.2f}d). "
                "Consistent with an exoplanet transit."
            )

        elif slope_asymmetry > self.ASYMMETRY_THRESHOLD:
            # Brightening AND rise much faster than decay → stellar magnetic flare.
            event_type = 'FLARE'

            # Higher asymmetry → stronger "shark-fin" profile → higher confidence.
            confidence = round(float(np.clip(0.5 + slope_asymmetry * 0.49, 0.4, 0.99)), 2)
            description = (
                f"Fast-rise slow-decay spike detected "
                f"(slope asymmetry={slope_asymmetry:.2f}, sharpness={sharpness:.1f}, "
                f"duration={duration_days:.2f}d). "
                "Consistent with a stellar flare."
            )

        elif amplitude < self.MIN_MICROLENSING_AMPLITUDE or duration_days < self.MIN_MICROLENSING_DURATION:
            # The event could be a microlensing event but is too faint or too brief
            # to distinguish from noise — discard it.
            event_type = 'NOISE'

            # The closer the amplitude is to the threshold, the less confident
            # we are that it is truly noise (it could be a very faint real event).
            confidence = round(float(np.clip(1.0 - amplitude * 1000, 0.5, 0.95)), 2)
            description = (
                f"Signal too weak or too short to be a real astrophysical event "
                f"(amplitude={amplitude:.5f}, duration={duration_days:.2f}d). "
                "Classified as residual noise."
            )

        else:
            # Symmetric brightening, sufficient amplitude and duration →
            # a massive foreground object bending the star's light toward us.
            event_type = 'MICROLENSING'

            sym        = 1.0 - abs(slope_asymmetry)
            confidence = round(float(np.clip(0.5 + sym * 0.49, 0.4, 0.99)), 2)
            description = (
                f"Symmetric brightness increase over {duration_days:.2f} day(s) detected "
                f"(slope asymmetry={slope_asymmetry:.2f}, amplitude={amplitude:.5f}). "
                "Consistent with gravitational microlensing."
            )

        return {
            'event_type' : event_type,
            'confidence' : confidence,
            'description': description
        }

    def classify_all(self, segments: list[dict]) -> list[dict]:
        """
        Classify every segment and return only the astrophysically significant ones.
        NOISE segments are counted but filtered out of the final output.

        Parameters
        ----------
        segments : list[dict]
            Output of AnomalyDetector.get_anomaly_segments().

        Returns
        -------
        list[dict]
            Each item is a segment dict enriched with 'event_type', 'confidence',
            and 'description'.  NOISE items are excluded.
        """
        if not segments:
            print("No segments to classify.")
            return []

        print(f"Classifying {len(segments)} segment(s)...")
        results     = []
        noise_count = 0

        # enumerate() gives us both the index (i) and the value (segment)
        # of each item as we loop — useful for the progress print below.
        for i, segment in enumerate(segments):
            classification = self.classify_segment(segment)

            if classification['event_type'] == 'NOISE':
                # Count discarded noise segments to report them at the end.
                noise_count += 1
                # 'continue' skips the rest of this loop iteration and jumps
                # directly to the next segment without adding to results.
                continue

            # ** (double-star) unpacking merges two dicts into one.
            # {**a, **b} creates a new dict containing ALL keys from both a and b.
            # If a key exists in both, b's value takes precedence.
            result = {**segment, **classification}
            results.append(result)

            print(
                f"  Segment {i + 1}: {classification['event_type']} "
                f"(confidence={classification['confidence']:.0%}) — {classification['description']}"
            )

        print(f"  ({noise_count} segment(s) discarded as noise)")
        return results
