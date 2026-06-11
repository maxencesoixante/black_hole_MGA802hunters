import pandas as pd
import numpy as np


class AnomalyDetector:
    """
    Detects anomalies in a cleaned light curve using a rolling window approach.
    An anomaly is flagged when the flux deviates significantly from the local
    rolling mean, based on a dynamic threshold (N × local std).
    """

    def __init__(self, df: pd.DataFrame, window: int = 200, sigma_threshold: float = 3.0):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned light curve with 'time' and 'flux' columns.
        window : int
            Size of the rolling window (number of data points).
        sigma_threshold : float
            Number of standard deviations above/below the rolling mean
            to flag a point as anomalous.
        """
        if 'time' not in df.columns or 'flux' not in df.columns:
            raise ValueError("DataFrame must contain 'time' and 'flux' columns.")

        self.df = df.copy().reset_index(drop=True)
        self.window = window
        self.sigma_threshold = sigma_threshold

    def detect(self) -> pd.DataFrame:
        """
        Runs the rolling-window detection.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with three new columns:
            - 'rolling_mean' : local baseline
            - 'rolling_std'  : local variability
            - 'is_anomaly'   : boolean flag
        """
        print(f"Running anomaly detection (window={self.window}, sigma={self.sigma_threshold})...")

        self.df['rolling_mean'] = (
            self.df['flux']
            .rolling(window=self.window, center=True, min_periods=1)
            .mean()
        )
        self.df['rolling_std'] = (
            self.df['flux']
            .rolling(window=self.window, center=True, min_periods=1)
            .std()
        )

        upper = self.df['rolling_mean'] + self.sigma_threshold * self.df['rolling_std']
        lower = self.df['rolling_mean'] - self.sigma_threshold * self.df['rolling_std']
        self.df['is_anomaly'] = (self.df['flux'] > upper) | (self.df['flux'] < lower)

        n_anomalies = self.df['is_anomaly'].sum()
        print(f"Detection complete. {n_anomalies} anomalous point(s) found.")

        return self.df

    def get_anomaly_segments(self, min_gap_days: float = 0.5) -> list[dict]:
        """
        Groups anomalous points into segments based on temporal proximity.

        Instead of grouping by index distance, we group by time gap — this
        correctly merges all points belonging to the same transit (~7-15 hours)
        even when they are not strictly consecutive in index.

        Parameters
        ----------
        min_gap_days : float
            Maximum time gap (in days) between two anomalous points to
            consider them part of the same event. Default 0.5 days (12h),
            which covers a full Kepler transit (~6-10h).

        Returns
        -------
        list of dict, each with keys:
            - 'start_idx'        : first index of the segment (with context)
            - 'end_idx'          : last index of the segment (with context)
            - 'time'             : array of time values (with context)
            - 'flux'             : array of flux values (with context)
            - 'anomaly_peak_idx' : index of the strongest anomalous point
        """
        if 'is_anomaly' not in self.df.columns:
            raise RuntimeError("Run detect() before calling get_anomaly_segments().")

        anomaly_indices = self.df.index[self.df['is_anomaly']].tolist()

        if not anomaly_indices:
            print("No anomaly segments to extract.")
            return []

        # Group by temporal proximity
        times = self.df.loc[anomaly_indices, 'time'].values
        segments = []
        current_segment = [anomaly_indices[0]]

        for i in range(1, len(anomaly_indices)):
            idx = anomaly_indices[i]
            time_gap = times[i] - times[i - 1]
            if time_gap <= min_gap_days:
                current_segment.append(idx)
            else:
                segments.append(current_segment)
                current_segment = [idx]
        segments.append(current_segment)

        # Build segment windows with context around each anomaly group
        context = self.window // 2
        result = []
        for seg in segments:
            start = max(0, seg[0] - context)
            end = min(len(self.df) - 1, seg[-1] + context)
            segment_df = self.df.loc[start:end]

            deviations = np.abs(
                self.df.loc[seg, 'flux'].values - self.df.loc[seg, 'rolling_mean'].values
            )
            peak_idx = seg[int(np.argmax(deviations))]

            result.append({
                'start_idx': start,
                'end_idx': end,
                'time': segment_df['time'].values,
                'flux': segment_df['flux'].values,
                'anomaly_peak_idx': peak_idx
            })

        print(f"{len(result)} anomaly segment(s) extracted.")
        return result


class AnomalyClassifier:
    """
    Classifies an anomaly segment into one of four categories:

        - TRANSIT      : U-shaped dip (exoplanet passing in front of the star)
        - FLARE        : fast-rise slow-decay spike (stellar magnetic eruption)
        - MICROLENSING : symmetric bell-shaped brightening over several days
                         (massive object acting as gravitational lens)
        - NOISE        : signal too weak or too short to be a real event

    Classification logic
    --------------------
    1. Direction  → dip below baseline   : TRANSIT
    2. Asymmetry  → rise faster than decay : FLARE
    3. Amplitude + duration checks:
         - too weak OR too short          : NOISE
         - otherwise                      : MICROLENSING
    """

    DIP_THRESHOLD = -0.0008             # normalized flux below this → dip (TRANSIT)
    ASYMMETRY_THRESHOLD = 0.30          # slope asymmetry above this → FLARE
    MIN_MICROLENSING_AMPLITUDE = 0.005  # était 0.0005 → x10 plus strict
    MIN_MICROLENSING_DURATION = 0.3     # min duration in days (~7h) for MICROLENSING

    def classify_segment(self, segment: dict) -> dict:
        """
        Classifies a single segment.

        Parameters
        ----------
        segment : dict
            One item from AnomalyDetector.get_anomaly_segments().

        Returns
        -------
        dict with keys:
            - 'event_type'  : str   ('TRANSIT', 'FLARE', 'MICROLENSING', or 'NOISE')
            - 'confidence'  : float between 0 and 1
            - 'description' : str
        """
        flux = segment['flux']
        time = segment['time']

        if len(flux) < 5:
            return {
                'event_type': 'NOISE',
                'confidence': 0.0,
                'description': 'Segment too short to classify.'
            }

        # Normalize flux around local median (robust baseline)
        baseline = np.median(flux)
        norm_flux = (flux - baseline) / (np.abs(baseline) + 1e-10)

        # Find peak (maximum absolute deviation)
        peak_idx = int(np.argmax(np.abs(norm_flux)))
        peak_value = norm_flux[peak_idx]

        # Feature 1: Direction (dip or brightening)
        is_dip = peak_value < self.DIP_THRESHOLD

        # Feature 2: Slope-based asymmetry
        left_flux = norm_flux[:peak_idx + 1]
        right_flux = norm_flux[peak_idx:]

        rise_slope = np.mean(np.abs(np.diff(left_flux))) if len(left_flux) > 1 else 0.0
        decay_slope = np.mean(np.abs(np.diff(right_flux))) if len(right_flux) > 1 else 0.0

        slope_sum = rise_slope + decay_slope + 1e-10
        slope_asymmetry = (rise_slope - decay_slope) / slope_sum

        # Feature 3: Sharpness
        wing_mean = np.mean(np.abs(norm_flux[[0, -1]])) + 1e-10
        sharpness = np.abs(peak_value) / wing_mean

        # Feature 4: Duration and amplitude (for NOISE filtering)
        duration_days = float(time[-1] - time[0])
        amplitude = float(np.abs(peak_value))

        # ── Classification ──────────────────────────────────────────────────

        if is_dip:
            event_type = 'TRANSIT'
            sym = 1.0 - abs(slope_asymmetry)
            confidence = round(float(np.clip(0.5 + sym * 0.49, 0.4, 0.99)), 2)
            description = (
                f"U-shaped flux dip detected "
                f"(slope asymmetry={slope_asymmetry:.2f}, sharpness={sharpness:.1f}, "
                f"duration={duration_days:.2f}d). "
                "Consistent with an exoplanet transit."
            )

        elif slope_asymmetry > self.ASYMMETRY_THRESHOLD:
            event_type = 'FLARE'
            confidence = round(float(np.clip(0.5 + slope_asymmetry * 0.49, 0.4, 0.99)), 2)
            description = (
                f"Fast-rise slow-decay spike detected "
                f"(slope asymmetry={slope_asymmetry:.2f}, sharpness={sharpness:.1f}, "
                f"duration={duration_days:.2f}d). "
                "Consistent with a stellar flare."
            )

        elif amplitude < self.MIN_MICROLENSING_AMPLITUDE or duration_days < self.MIN_MICROLENSING_DURATION:
            # Symmetric brightening but too weak or too short → likely noise
            event_type = 'NOISE'
            confidence = round(float(np.clip(1.0 - amplitude * 1000, 0.5, 0.95)), 2)
            description = (
                f"Signal too weak or too short to be a real astrophysical event "
                f"(amplitude={amplitude:.5f}, duration={duration_days:.2f}d). "
                "Classified as residual noise."
            )

        else:
            event_type = 'MICROLENSING'
            sym = 1.0 - abs(slope_asymmetry)
            confidence = round(float(np.clip(0.5 + sym * 0.49, 0.4, 0.99)), 2)
            description = (
                f"Symmetric brightness increase over {duration_days:.2f} day(s) detected "
                f"(slope asymmetry={slope_asymmetry:.2f}, amplitude={amplitude:.5f}). "
                "Consistent with gravitational microlensing."
            )

        return {
            'event_type': event_type,
            'confidence': confidence,
            'description': description
        }

    def classify_all(self, segments: list[dict]) -> list[dict]:
        """
        Classifies all segments and returns enriched results.
        NOISE events are filtered out of the final output.
        """
        if not segments:
            print("No segments to classify.")
            return []

        print(f"Classifying {len(segments)} segment(s)...")
        results = []
        noise_count = 0

        for i, segment in enumerate(segments):
            classification = self.classify_segment(segment)

            if classification['event_type'] == 'NOISE':
                noise_count += 1
                continue  # Skip noise — don't add to results

            result = {**segment, **classification}
            results.append(result)
            print(
                f"  Segment {i + 1}: {classification['event_type']} "
                f"(confidence={classification['confidence']:.0%}) — {classification['description']}"
            )

        print(f"  ({noise_count} segment(s) discarded as noise)")
        return results
