import pandas as pd
import numpy as np


class AnomalyDetector:
    """
    Detects anomalies in a cleaned light curve using a rolling window approach.
    An anomaly is flagged when the flux deviates significantly from the local
    rolling mean, based on a dynamic threshold (N × local std).
    """

    def __init__(self, df: pd.DataFrame, window: int = 50, sigma_threshold: float = 3.0):
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

    def get_anomaly_segments(self, min_gap: int = 20) -> list[dict]:
        """
        Groups consecutive anomalous points into segments (events).

        Parameters
        ----------
        min_gap : int
            Minimum number of normal points between two anomalies to consider
            them as separate events. Increased to 20 to properly merge
            multi-point events like exoplanet transits.

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

        # Group indices into segments separated by at least min_gap normal points
        segments = []
        current_segment = [anomaly_indices[0]]

        for idx in anomaly_indices[1:]:
            if idx - current_segment[-1] <= min_gap:
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

            # Peak = point with largest absolute deviation from rolling mean
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
    Classifies an anomaly segment into one of three astrophysical event types
    based on light curve shape analysis:

        - FLARE       : sudden spike upward, fast rise / slow decay (asymmetric slopes)
        - TRANSIT     : U-shaped dip (symmetric flux decrease below baseline)
        - MICROLENSING: symmetric bell-shaped brightening (symmetric flux increase)
    """

    DIP_THRESHOLD = -0.002      # normalized flux minimum below this → dip event
    ASYMMETRY_THRESHOLD = 0.30  # slope asymmetry ratio above this → Flare

    def classify_segment(self, segment: dict) -> dict:
        """
        Classifies a single segment using slope-based asymmetry.

        Instead of comparing left/right half *lengths* (which are always equal
        for single-point anomalies), we compare the mean absolute *slope*
        on each side of the peak — a flare rises steeply and decays slowly.

        Parameters
        ----------
        segment : dict
            One item from AnomalyDetector.get_anomaly_segments().

        Returns
        -------
        dict with keys:
            - 'event_type'  : str   ('FLARE', 'TRANSIT', or 'MICROLENSING')
            - 'confidence'  : float between 0 and 1
            - 'description' : str   (human-readable explanation)
        """
        flux = segment['flux']

        if len(flux) < 5:
            return {
                'event_type': 'UNKNOWN',
                'confidence': 0.0,
                'description': 'Segment too short to classify.'
            }

        # --- Normalize flux around local median (robust baseline) ---
        baseline = np.median(flux)
        norm_flux = (flux - baseline) / (np.abs(baseline) + 1e-10)

        # --- Find peak (maximum absolute deviation) ---
        peak_idx = int(np.argmax(np.abs(norm_flux)))
        peak_value = norm_flux[peak_idx]

        # ---- Feature 1: Direction (dip or brightening) ----
        is_dip = peak_value < self.DIP_THRESHOLD

        # ---- Feature 2: Slope-based asymmetry ----
        # Rise slope = mean |Δflux| per step on the left side of peak
        # Decay slope = mean |Δflux| per step on the right side of peak
        left_flux = norm_flux[:peak_idx + 1]
        right_flux = norm_flux[peak_idx:]

        rise_slope = np.mean(np.abs(np.diff(left_flux))) if len(left_flux) > 1 else 0.0
        decay_slope = np.mean(np.abs(np.diff(right_flux))) if len(right_flux) > 1 else 0.0

        slope_sum = rise_slope + decay_slope + 1e-10
        # Positive → rise faster than decay (Flare pattern)
        # Near 0   → symmetric (Microlensing or Transit)
        slope_asymmetry = (rise_slope - decay_slope) / slope_sum

        # ---- Feature 3: Sharpness (peak vs wing ratio) ----
        wing_mean = np.mean(np.abs(norm_flux[[0, -1]])) + 1e-10
        sharpness = np.abs(peak_value) / wing_mean

        # ---- Classification logic ----
        if is_dip:
            event_type = 'TRANSIT'
            sym = 1.0 - abs(slope_asymmetry)
            confidence = round(float(np.clip(0.5 + sym * 0.49, 0.4, 0.99)), 2)
            description = (
                f"U-shaped flux dip detected "
                f"(slope asymmetry={slope_asymmetry:.2f}, sharpness={sharpness:.1f}). "
                "Consistent with an exoplanet transit."
            )

        elif slope_asymmetry > self.ASYMMETRY_THRESHOLD:
            event_type = 'FLARE'
            confidence = round(float(np.clip(0.5 + slope_asymmetry * 0.49, 0.4, 0.99)), 2)
            description = (
                f"Fast-rise slow-decay spike detected "
                f"(slope asymmetry={slope_asymmetry:.2f}, sharpness={sharpness:.1f}). "
                "Consistent with a stellar flare."
            )

        else:
            event_type = 'MICROLENSING'
            sym = 1.0 - abs(slope_asymmetry)
            confidence = round(float(np.clip(0.5 + sym * 0.49, 0.4, 0.99)), 2)
            description = (
                f"Symmetric brightness increase detected "
                f"(slope asymmetry={slope_asymmetry:.2f}, sharpness={sharpness:.1f}). "
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

        Parameters
        ----------
        segments : list of dict
            Output of AnomalyDetector.get_anomaly_segments().

        Returns
        -------
        list of dict, each combining the original segment data with
        the classification result.
        """
        if not segments:
            print("No segments to classify.")
            return []

        print(f"Classifying {len(segments)} segment(s)...")
        results = []

        for i, segment in enumerate(segments):
            classification = self.classify_segment(segment)
            result = {**segment, **classification}
            results.append(result)
            print(
                f"  Segment {i + 1}: {classification['event_type']} "
                f"(confidence={classification['confidence']:.0%}) — {classification['description']}"
            )

        return results
