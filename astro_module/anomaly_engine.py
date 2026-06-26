import pandas as pd
import numpy as np
# Pandas  → tabular data: DataFrames with named columns and integer row indices
# NumPy   → fast numerical operations on arrays (mathematical functions, statistics)

class AnomalyDetector:
    """
    Detects astrophysical events in a cleaned light curve by measuring how far
    the ROLLING-MEAN baseline drifts from the star's GLOBAL baseline.

    ── Core Concept: baseline departure ─────────────────────────────────────────
    A rolling window slides along the light curve, averaging `window` neighbouring
    points to produce a smooth local baseline (the rolling mean). When that local
    baseline drifts far enough from the star's global median, a sustained event is
    flagged:

        departure(σ_global) = (rolling_mean − global_median) / global_std

    A sustained DIP is a transit; a sustained BUMP is a flare (if asymmetric) or a
    microlensing event (if symmetric).  See detect_baseline_departures() for the full
    method, its parameters, and its documented sensitivity limits.
    ─────────────────────────────────────────────────────────────────────────────

    Attributes
        ----------
        df : pd.DataFrame
        Cleaned light curve data with 'time' and 'flux' columns.
        window : int
            Number of data points in each rolling window.
        """

    def __init__(self, df: pd.DataFrame, window: int = 200):
        """
        Initialise the detector with the cleaned light curve and its hyperparameter.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned light curve — output of SignalCleaner.process_data().
            Must have 'time' and 'flux' columns.
        window : int
            Number of data points in each rolling window.
            Larger window  → smoother baseline, better for broad events.
            Smaller window → follows shorter events but noisier.
            Default is 200.

        Raises
        ------
        ValueError
            If the DataFrame does not contain 'time' and 'flux' columns.
        """
        if 'time' not in df.columns or 'flux' not in df.columns:
            raise ValueError("DataFrame must contain 'time' and 'flux' columns.")

        # .copy() prevents the original DataFrame (from SignalCleaner) from being
        # modified accidentally — good defensive programming practice.
        # reset_index(drop=True) ensures the row index is 0, 1, 2, … with no gaps,
        # which is critical for the index arithmetic used to group event runs.
        self.df = df.copy().reset_index(drop=True)

        # Store the hyperparameter so every method can read it and it is
        # visible to the user for reproducibility.
        self.window = window

    def detect_baseline_departures(
        self,
        baseline_sigma: float = 1.0,
        min_duration_days: float = 0.1,
        min_points: int = 50,
        flag: bool = True,
    ) -> list[dict]:
        """
        Detect SUSTAINED departures of the rolling baseline from the global one.

        ── How it works ─────────────────────────────────────────────────────────
        A naive "point-by-point" detector compares each cadence to its own LOCAL
        rolling mean. That is BLIND to BROAD events: when a dip (or bump) lasts
        LONGER than the rolling window, the rolling mean simply slides into the
        event and no single point ever "sticks out". The event hides itself
        inside the baseline — the same self-erasing effect that SignalCleaner
        avoids by refusing flatten(). On KIC 11904151 this is why the ~0.04%
        depression near day 229 is visible in the rolling-mean curve yet would be
        flagged zero times by a point-wise test.

        This method asks a different question — not "does this point leave its
        LOCAL band?" but "does the LOCAL baseline leave the GLOBAL baseline?":

            departure(σ_global) = (rolling_mean − global_median) / global_std

        A run of points whose departure stays beyond ±baseline_sigma for at least
        min_duration_days (and min_points cadences) is reported as a candidate
        event. The point requirement also rejects the flat rolling-mean plateaus
        that appear across observation gaps (few points spanning a long time).

        ── Known sensitivity limit (documented, not a bug) ──────────────────────
        This method only sees BROAD departures of the baseline, so it has a blind
        spot for two event families:

          • FLARES — short, sharp brightenings (minutes–~1 h). The rolling window
            dilutes them; on an active star they drown in the rotational
            modulation (tested on GJ 1243).
          • FAINT, SHORT MICROLENSING — e.g. KOI-3278's self-lensing pulse
            (0.1 % over 5 h) is only ~0.11σ_global on its spotted host: below the
            threshold and too brief. Only the host's broad variability is seen.

        The shape CLASSIFIER below can still label such events (FLARE / MICROLENSING)
        once detected — the gap is detection, not labelling. Catching them would
        need a complementary SHORT-timescale channel (brief-spike detection for
        flares) or PERIOD FOLDING to stack a weak periodic signal.

        Parameters
        ----------
        baseline_sigma : float
            How far the rolling mean must drift from the global baseline,
            measured in global standard deviations, to count as a departure.
            Default is 1.0.
        min_duration_days : float
            Minimum time span of a departure run to be reported (filters blips).
            Default is 0.1.
        min_points : int
            Minimum number of cadences in a run (filters gap-straddling artefacts).
            Default is 50.
        flag : bool
            If True, add a boolean 'is_baseline_anomaly' column to self.df marking
            the peak cadence of each reported run, so a visualizer can draw it.
            Default is True.

        Returns
        -------
        list of dict, each containing:
            'direction'     : 'DIP' (dimming) or 'BUMP' (brightening)
            'candidate'     : human-readable guess for the event type
            'peak_time'     : timestamp (days) of the strongest departure
            'depth_sigma'   : strongest |departure| in the run, in σ_global
            'depth_frac'    : same depth expressed as a fraction of the baseline
            'duration_days' : time span of the run
            'n_points'      : number of cadences in the run
            'peak_idx'      : DataFrame row index of the strongest departure
        """
        # We need the rolling baseline.  Compute it on demand if it is not
        # already present, so this method is fully self-contained.
        if 'rolling_mean' not in self.df.columns:
            self.df['rolling_mean'] = (
                self.df['flux']
                .rolling(window=self.window, center=True, min_periods=1)
                .mean()
            )

        print(
            f"Running baseline-departure detection "
            f"(baseline_sigma={baseline_sigma}, min_duration={min_duration_days}d)..."
        )

        # ── Global reference (robust) ─────────────────────────────────────────
        # The median is unaffected by the very events we are hunting; the std sets
        # the natural noise scale of the star against which a drift is "large".
        global_median = float(self.df['flux'].median())
        global_std    = float(self.df['flux'].std())
        if global_std == 0:
            global_std = 1e-10   # degenerate guard: perfectly flat light curve

        # Expose the detection reference so a visualiser can draw the exact
        # envelope this method used (global baseline ± baseline_sigma × global_std).
        self.global_median = global_median
        self.global_std    = global_std
        self.baseline_sigma = baseline_sigma

        time = self.df['time'].values
        rmean = self.df['rolling_mean'].values

        # Signed departure of the local baseline from the global baseline,
        # expressed in units of the global noise.
        departure = (rmean - global_median) / global_std

        # Boolean mask: True where the baseline has drifted beyond the threshold.
        # np.abs() treats dips and bumps symmetrically.
        mask = np.abs(departure) >= baseline_sigma

        # ── Group consecutive True values into runs ───────────────────────────
        # np.diff on the integer mask is +1 where a run starts and -1 where it ends.
        # np.flatnonzero returns the positions of those transitions.
        m = mask.astype(int)
        starts = np.flatnonzero(np.diff(np.concatenate(([0], m))) == 1)
        ends   = np.flatnonzero(np.diff(np.concatenate((m, [0]))) == -1)

        events = []
        flag_indices = []
        for s, e in zip(starts, ends):
            n_pts = e - s + 1
            span  = float(time[e] - time[s])

            # Reject runs that are too brief or too sparse (gap artefacts).
            if span < min_duration_days or n_pts < min_points:
                continue

            # Strongest departure inside the run (where |departure| is largest).
            run_dep   = departure[s:e + 1]
            local_pk  = int(np.argmax(np.abs(run_dep)))
            peak_idx  = s + local_pk
            depth_sig = float(np.abs(run_dep[local_pk]))
            signed    = float(run_dep[local_pk])

            direction = 'DIP' if signed < 0 else 'BUMP'
            # Classify the event by its SHAPE:
            #   DIP                       → TRANSIT      (planet crosses the disc)
            #   BUMP, asymmetric profile  → FLARE        (fast rise, slow decay)
            #   BUMP, symmetric profile   → MICROLENSING (a compact foreground
            #                                             mass — white dwarf, neutron
            #                                             star or black hole — lenses
            #                                             the star; symmetric in time)
            event_type = self._classify_departure(s, e, direction)
            candidate = {
                'TRANSIT'     : 'transit / eclipse candidate',
                'FLARE'       : 'stellar flare candidate',
                'MICROLENSING': 'microlensing (compact object) candidate',
            }[event_type]

            # Confidence grows with how far the baseline drifted beyond the
            # threshold, clamped to [0.4, 0.99] (never absolutely certain).
            confidence = round(
                float(np.clip(0.5 + (depth_sig - baseline_sigma) * 0.3, 0.4, 0.99)),
                2,
            )

            events.append({
                'event_type'      : event_type,    # 'TRANSIT' or 'BRIGHTENING'
                'direction'       : direction,
                'candidate'       : candidate,
                'confidence'      : confidence,
                'peak_time'       : float(time[peak_idx]),
                'depth_sigma'     : round(depth_sig, 2),
                'depth_frac'      : round(depth_sig * global_std, 6),
                'duration_days'   : round(span, 3),
                'n_points'        : int(n_pts),
                'peak_idx'        : int(peak_idx),
                # Alias so the dict is drop-in compatible with the plotter's
                # event-annotation loop (which reads 'anomaly_peak_idx').
                'anomaly_peak_idx': int(peak_idx),
            })
            flag_indices.append(peak_idx)

        # Optionally mark the peak cadence of each event for plotting.
        if flag:
            self.df['is_baseline_anomaly'] = False
            if flag_indices:
                self.df.loc[flag_indices, 'is_baseline_anomaly'] = True

        n_by_type = {t: sum(1 for ev in events if ev['event_type'] == t)
                     for t in ('TRANSIT', 'FLARE', 'MICROLENSING')}
        print(f"{len(events)} baseline-departure event(s) found "
              f"({n_by_type['TRANSIT']} transit, {n_by_type['FLARE']} flare, "
              f"{n_by_type['MICROLENSING']} microlensing).")
        return events

    # ── Asymmetry threshold separating flares from microlensing ───────────────
    # A flare rises far faster than it decays (a "shark-fin"); microlensing is
    # symmetric in time. 0.30 = the rise side must be ≥30% steeper than the decay
    # side to be called a flare.
    FLARE_ASYMMETRY_THRESHOLD = 0.30

    def _classify_departure(self, s, e, direction):
        """
        Label a baseline-departure run by the SHAPE of its flux profile.

        Parameters
        ----------
        s, e : int
            First and last row index of the run (where the rolling mean is
            beyond the threshold).
        direction : str
            'DIP' (dimming) or 'BUMP' (brightening), from the sign of the
            baseline departure.

        Returns
        -------
        str
            'TRANSIT', 'FLARE', or 'MICROLENSING'.
            - DIP                      → TRANSIT
            - BUMP, asymmetric profile → FLARE        (fast rise / slow decay)
            - BUMP, symmetric profile  → MICROLENSING (compact-object lensing)
        """
        if direction == 'DIP':
            return 'TRANSIT'

        # Pad the run with one run-length of context on each side so the rising
        # and decaying wings of the event are included in the slope estimate.
        n   = len(self.df)
        pad = max(5, e - s)
        a   = max(0, s - pad)
        b   = min(n - 1, e + pad)

        # Analyse the ROLLING MEAN, not the raw flux: on an active star the raw
        # scatter is so large it drowns the asymmetry signal.  The rolling mean is
        # smooth, so a flare's characteristic "fast rise / slow decay" survives.
        profile = self.df['rolling_mean'].values[a:b + 1]
        if len(profile) < 5:
            return 'MICROLENSING'   # too few points to judge shape → default

        # Split the profile at its brightest point and measure the TOTAL rise vs
        # the TOTAL decay (peak height above each wing) and the TIME taken on each
        # side.  Slope = height / time.  A flare rises far faster than it decays.
        peak       = int(np.argmax(profile))
        peak_val   = profile[peak]
        rise_h     = peak_val - profile[0]
        decay_h    = peak_val - profile[-1]
        rise_t     = max(peak, 1)
        decay_t    = max(len(profile) - 1 - peak, 1)
        rise_slope  = rise_h  / rise_t
        decay_slope = decay_h / decay_t
        asymmetry = (rise_slope - decay_slope) / (abs(rise_slope) + abs(decay_slope) + 1e-12)

        return 'FLARE' if asymmetry > self.FLARE_ASYMMETRY_THRESHOLD else 'MICROLENSING'
