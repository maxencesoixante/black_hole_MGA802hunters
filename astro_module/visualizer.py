import matplotlib.pyplot as plt
import matplotlib.patches as mpatches   # used to create custom legend entries (coloured rectangles)
import pandas as pd
import numpy as np

# Seaborn is optional: it only adds nicer visual styling on top of Matplotlib.
# If it is not installed ("pip install seaborn"), the plot still works —
# we fall back to Matplotlib's built-in "seaborn-v0_8-darkgrid" style instead.
try:
    import seaborn as sns
    _SEABORN_AVAILABLE = True
except ImportError:
    _SEABORN_AVAILABLE = False


class AstroPlotter:
    """
    Generates a professional, publication-quality annotated light-curve plot.

    ── What this class produces ─────────────────────────────────────────────────
    A single Matplotlib figure containing:
      • All cadences in the light curve plotted as small scatter dots.
          - Normal cadences → light blue/grey (discrete, stay in the background)
          - Anomalous cadences → vivid red    (immediately catch the eye)
      • The rolling-mean baseline as a dashed blue line, so the reader can
        see the quiet trend the detector used as its reference.
      • A shaded ±σ confidence band around the baseline, showing the envelope
        outside which a point is flagged as an anomaly.
      • Text annotations for every classified event (TRANSIT / FLARE / MICROLENSING)
        placed directly above the strongest point of the event with a pointer arrow.
      • A full legend, descriptive axis labels, and a title.

    ── OOP reminder ─────────────────────────────────────────────────────────────
    Like the other classes in this project, AstroPlotter bundles together
    all the data it needs (stored in __init__) and the actions it can perform
    (show_results).  This keeps the plotting code self-contained and reusable:
    you can create multiple AstroPlotter objects for different targets without
    any global state.
    ─────────────────────────────────────────────────────────────────────────────
    """

    # ── Colour palette — class-level constants ────────────────────────────────
    # Defined here (outside __init__) so they are shared by every AstroPlotter
    # object and can be tweaked in one place without touching any method logic.
    COLOUR_NORMAL   = '#B0C8E8'   # soft steel blue  — quiet, normal cadences
    COLOUR_ANOMALY  = '#E84040'   # vivid red         — anomalous cadences
    COLOUR_BASELINE = '#2255AA'   # medium blue       — rolling-mean trend line
    COLOUR_BAND     = '#90B8D8'   # pale blue         — ±σ confidence shading

    # Per-event-type annotation colours make it easy to distinguish events at a glance.
    COLOUR_LABELS = {
        'TRANSIT'     : '#9B59B6',   # purple — exoplanet transit dip
        'FLARE'       : '#E67E22',   # orange — stellar magnetic flare
        'MICROLENSING': '#27AE60',   # green  — gravitational microlensing
    }

    # Channel-B (baseline-departure) events get their own gold diamond so they
    # are never confused with the round point-by-point anomaly markers.
    COLOUR_BASELINE_EVENT = '#F1C40F'   # gold      — diamond marker fill
    COLOUR_BASELINE_EDGE  = '#7D6608'   # dark gold — diamond edge + label text

    def __init__(
        self,
        df_flagged          : pd.DataFrame,
        classified_results  : list,
        target_id           : str   = "Unknown Target",
        sigma_threshold     : float = 3.0,
        window              : int   = 200,
        baseline_events     : list  = None,
        show_point_anomalies: bool  = True,
        show_detection_band : bool  = True,
        baseline_sigma      : float = None,
        global_median       : float = None,
        global_std          : float = None,
    ):
        """
        Store all the data needed to produce the plot.

        Parameters
        ----------
        df_flagged : pd.DataFrame
            Output of AnomalyDetector.detect() (and/or detect_baseline_departures).
            Always needs 'time', 'flux', 'rolling_mean'.  'rolling_std' and
            'is_anomaly' are only needed when the matching layer is shown.
        classified_results : list[dict]
            The events to annotate.  Each dict must have 'event_type',
            'confidence' and 'anomaly_peak_idx'.  This is now fed directly by
            AnomalyDetector.detect_baseline_departures() (transits), but the old
            AnomalyClassifier.classify_all() output works unchanged too.
        target_id : str
            The star identifier displayed in the figure title (e.g. 'KIC 11904151').
        sigma_threshold : float
            The local σ used by the point-by-point detector — only used to draw
            the ±σ band when show_detection_band is True.
        window : int
            The rolling window size used during detection — shown in the legend.
        baseline_events : list[dict] or None
            Legacy channel-B input drawn as separate gold diamonds.  Leave None
            when the transits are already passed via classified_results.
        show_point_anomalies : bool
            If True (default, legacy), split the scatter into normal/anomalous
            using 'is_anomaly' and draw the red markers.  Set False to base the
            figure purely on the rolling-window (baseline-departure) results.
        show_detection_band : bool
            If True (default, legacy), shade the local ±σ point-detection band.
            Set False when not using the point-by-point detector.
        baseline_sigma : float or None
            If given, draw the rolling-window detection envelope:
            global baseline ± baseline_sigma × global_std.  This is the exact
            threshold the baseline-departure detector used.
        global_median, global_std : float or None
            The global reference used by the baseline-departure detector.  If not
            supplied they are recomputed from the flux column.
        """
        # self.X = value stores each argument ON the object so that show_results()
        # can access all of them via self.X without needing them passed again.
        self.df                   = df_flagged
        self.classified_results   = classified_results
        self.target_id            = target_id
        self.sigma_threshold      = sigma_threshold
        self.window               = window
        self.show_point_anomalies = show_point_anomalies
        self.show_detection_band  = show_detection_band
        self.baseline_sigma       = baseline_sigma
        # Fall back to recomputing the global reference if the caller did not pass it.
        self.global_median = global_median if global_median is not None else float(df_flagged['flux'].median())
        self.global_std    = global_std    if global_std    is not None else float(df_flagged['flux'].std())
        # Default to an empty list so "if self.baseline_events:" is always safe.
        self.baseline_events    = baseline_events or []

    def show_results(self, save_path: str = None):
        """
        Render and display the complete annotated light-curve figure.

        Parameters
        ----------
        save_path : str or None
            Optional file path to also save the figure to disk
            (e.g. 'output/KIC11904151_results.png').
            If None (default), the figure is only shown on screen.
        """

        # ── 1. Global visual styling ──────────────────────────────────────────
        # We apply a dark-grid style to make the plot look clean and professional.
        # Seaborn's set_theme() is the preferred approach when seaborn is installed;
        # otherwise we fall back to Matplotlib's own built-in style sheet.
        if _SEABORN_AVAILABLE:
            sns.set_theme(style='darkgrid', font_scale=1.1)
        else:
            # plt.style.use() applies one of Matplotlib's built-in style sheets.
            # 'seaborn-v0_8-darkgrid' mimics the seaborn dark-grid look natively.
            plt.style.use('seaborn-v0_8-darkgrid')

        # plt.subplots() creates a Figure (the whole canvas) and one Axes (the plot area).
        # figsize=(width, height) in inches — 16×6 gives a wide panoramic aspect ratio
        # well suited to time-series that span many more X-values than Y-values.
        # dpi=120 → crisp rendering on modern high-resolution screens.
        fig, ax = plt.subplots(figsize=(16, 6), dpi=120)

        # ── 2-4. Data scatter (point-anomaly layer is now optional) ───────────
        # When show_point_anomalies is False, the figure no longer relies on the
        # point-by-point detector at all: every cadence is drawn in one neutral
        # colour, and detection is conveyed solely by the rolling-window layer.
        if self.show_point_anomalies and 'is_anomaly' in self.df.columns:
            # Legacy behaviour: split normal vs anomalous and highlight in red.
            df_normal  = self.df[~self.df['is_anomaly']]
            df_anomaly = self.df[ self.df['is_anomaly']]
            ax.scatter(df_normal['time'], df_normal['flux'],
                       color=self.COLOUR_NORMAL, s=2, alpha=0.5, zorder=1,
                       label='Normal cadence')
            ax.scatter(df_anomaly['time'], df_anomaly['flux'],
                       color=self.COLOUR_ANOMALY, s=18, alpha=0.95, zorder=3,
                       label='Detected anomaly')
        else:
            # Rolling-window-only mode: a single neutral cloud of cadences.
            ax.scatter(self.df['time'], self.df['flux'],
                       color=self.COLOUR_NORMAL, s=2, alpha=0.5, zorder=1,
                       label='Cadence')

        # ── 5. Rolling-mean baseline ──────────────────────────────────────────
        # ax.plot() draws a continuous line — appropriate for the smooth baseline trend.
        # linestyle='--' (dashed) visually distinguishes it from the data scatter.
        ax.plot(
            self.df['time'],
            self.df['rolling_mean'],
            color     = self.COLOUR_BASELINE,
            linewidth = 1.5,
            linestyle = '--',
            alpha     = 0.85,
            zorder    = 2,
            label     = f'Rolling mean (window={self.window} pts)',
        )

        # ── 6a. Local ±σ point-detection band (legacy, optional) ──────────────
        # This is the envelope of the POINT-BY-POINT detector.  Only shown when
        # that detector is actually in use.
        if self.show_detection_band and 'rolling_std' in self.df.columns:
            upper_band = self.df['rolling_mean'] + self.sigma_threshold * self.df['rolling_std']
            lower_band = self.df['rolling_mean'] - self.sigma_threshold * self.df['rolling_std']
            ax.fill_between(
                self.df['time'], lower_band, upper_band,
                color=self.COLOUR_BAND, alpha=0.15, zorder=0,
                label=f'±{self.sigma_threshold}σ detection band',
            )

        # ── 6b. Rolling-window detection envelope (baseline departure) ────────
        # This is the threshold the baseline-departure detector actually uses:
        # a transit is flagged when the rolling mean leaves the horizontal band
        # global_baseline ± baseline_sigma × global_std.  A flat reference line
        # plus its band makes the criterion legible at a glance.
        if self.baseline_sigma is not None:
            hi = self.global_median + self.baseline_sigma * self.global_std
            lo = self.global_median - self.baseline_sigma * self.global_std
            ax.axhline(self.global_median, color='#555555', linewidth=0.8,
                       linestyle=':', alpha=0.7, zorder=0,
                       label='Global baseline')
            ax.fill_between(
                self.df['time'], lo, hi,
                color=self.COLOUR_BAND, alpha=0.18, zorder=0,
                label=f'±{self.baseline_sigma}σ baseline envelope',
            )

        # ── 7. Annotate each detected event ───────────────────────────────────
        # Events carrying a 'direction' key come from the baseline-departure
        # detector: the event lives on the ROLLING-MEAN curve, so we mark it there
        # with a diamond and offset the label in screen pixels (scale-safe).
        # Legacy classifier events (no 'direction') keep the original behaviour.
        marked_baseline = False
        for event in self.classified_results:
            peak_row   = self.df.loc[event['anomaly_peak_idx']]
            peak_time  = peak_row['time']
            event_type = event['event_type']
            confidence = event['confidence']
            colour     = self.COLOUR_LABELS.get(event_type, self.COLOUR_ANOMALY)

            is_baseline = 'direction' in event
            if is_baseline:
                # The departure is in the rolling mean → anchor the marker there.
                peak_y = peak_row['rolling_mean']
                ax.scatter([peak_time], [peak_y], color=colour, marker='D',
                           s=70, edgecolors='white', linewidths=1.0, zorder=4,
                           label=('Transit (baseline departure)'
                                  if not marked_baseline else None))
                marked_baseline = True

                # Screen-pixel offset: label below a dip, above a bump.
                dy = -38 if event['direction'] == 'DIP' else 38
                va = 'top' if event['direction'] == 'DIP' else 'bottom'
                ax.annotate(
                    f"{event_type}\n({confidence:.0%})",
                    xy=(peak_time, peak_y), xytext=(0, dy),
                    textcoords='offset points',
                    fontsize=8.5, color=colour, fontweight='bold',
                    ha='center', va=va,
                    arrowprops=dict(arrowstyle='->', color=colour, lw=1.5),
                    zorder=5,
                )
            else:
                # Legacy point-anomaly event: anchor on the flux point.
                peak_flux = peak_row['flux']
                ax.annotate(
                    f"{event_type}\n({confidence:.0%})",
                    xy=(peak_time, peak_flux),
                    xytext=(peak_time, peak_flux + 0.03),
                    fontsize=8.5, color=colour, fontweight='bold',
                    ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->', color=colour, lw=1.5),
                    zorder=5,
                )

        # ── 7b. Sustained baseline-departure events (channel B) ───────────────
        # These broad, shallow events are found by AnomalyDetector
        # .detect_baseline_departures(): the rolling mean ITSELF drifts far from
        # the global baseline.  The point-by-point detector (section 4) is blind
        # to them, so we mark them with a gold diamond placed ON the rolling-mean
        # curve (that is where the departure lives), with its own annotation.
        if self.baseline_events:
            # One scatter call for all diamonds → a single, clean legend entry.
            be_times = [self.df.loc[ev['peak_idx'], 'time']         for ev in self.baseline_events]
            be_vals  = [self.df.loc[ev['peak_idx'], 'rolling_mean'] for ev in self.baseline_events]
            ax.scatter(
                be_times,
                be_vals,
                color      = self.COLOUR_BASELINE_EVENT,
                marker     = 'D',          # diamond — visually distinct from round dots
                s          = 70,
                edgecolors = self.COLOUR_BASELINE_EDGE,
                linewidths = 1.0,
                zorder     = 4,
                label      = 'Baseline-departure event',
            )

            for ev in self.baseline_events:
                t = self.df.loc[ev['peak_idx'], 'time']
                v = self.df.loc[ev['peak_idx'], 'rolling_mean']

                # The flux axis spans only ~0.3% here, so a data-unit offset would
                # throw the label far off-screen.  We offset in SCREEN POINTS
                # instead (scale-independent): the label sits a fixed ~38 px from
                # the diamond — BELOW for a dip, ABOVE for a bump — and never
                # distorts the axis limits.
                dy = -38 if ev['direction'] == 'DIP' else 38
                va = 'top' if ev['direction'] == 'DIP' else 'bottom'

                ax.annotate(
                    f"BASELINE {ev['direction']}\n"
                    f"{ev['depth_sigma']}σ / {ev['depth_frac'] * 100:.3f}%",
                    xy         = (t, v),
                    xytext     = (0, dy),
                    textcoords = 'offset points',   # offset measured in screen pixels
                    fontsize   = 8.5,
                    color      = self.COLOUR_BASELINE_EDGE,
                    fontweight = 'bold',
                    ha         = 'center',
                    va         = va,
                    arrowprops = dict(
                        arrowstyle = '->',
                        color      = self.COLOUR_BASELINE_EVENT,
                        lw         = 1.5,
                    ),
                    zorder = 5,
                )

        # ── 8. Axis labels and title ──────────────────────────────────────────
        # labelpad adds space between the axis line and the label text.
        ax.set_xlabel('Time (days  —  BKJD)', fontsize=12, labelpad=8)
        ax.set_ylabel('Normalised Flux  (1.0 = typical brightness)', fontsize=12, labelpad=8)

        # Multi-line title: first line identifies the target; second line
        # summarises the detection.  In rolling-window-only mode we report the
        # number of transits; in legacy mode we also report point anomalies.
        n_transits = sum(1 for ev in self.classified_results
                         if ev.get('event_type') == 'TRANSIT')
        n_events   = len(self.classified_results)
        title = f'Automated Transit Detection  —  {self.target_id}\n'
        if self.show_point_anomalies and 'is_anomaly' in self.df.columns:
            n_anomalous = int(self.df['is_anomaly'].sum())
            title += (f'{n_anomalous} anomalous cadence(s) detected   •   '
                      f'{n_events} event(s) classified')
        else:
            title += f'{n_transits} transit(s) detected (rolling-window baseline departure)'
        # Only mention legacy channel-B diamonds when actually supplied.
        if self.baseline_events:
            title += f'   •   {len(self.baseline_events)} baseline-departure event(s)'
        ax.set_title(
            title,
            fontsize   = 13,
            fontweight = 'bold',
            pad        = 14,
        )

        # ── 9. Legend ─────────────────────────────────────────────────────────
        # ax.get_legend_handles_labels() retrieves the automatic legend entries
        # created by the 'label=' arguments in scatter/plot/fill_between above.
        base_handles, base_labels = ax.get_legend_handles_labels()

        # Legacy classifier events (no 'direction') are drawn with annotations
        # only, so they need a colour patch in the legend.  Baseline transits
        # already have their own diamond legend entry, so we skip those here and
        # only add patches for the legacy event types that are actually present.
        legacy_types  = [ev['event_type'] for ev in self.classified_results
                         if 'direction' not in ev]
        present_types = [t for t in self.COLOUR_LABELS if t in legacy_types]
        extra_handles = [
            mpatches.Patch(facecolor=self.COLOUR_LABELS[t], edgecolor='none', label=t)
            for t in present_types
        ]
        extra_labels = present_types

        ax.legend(
            handles    = base_handles + extra_handles,
            labels     = base_labels  + extra_labels,
            loc        = 'upper right',
            fontsize   = 9,
            framealpha = 0.85,    # slightly transparent legend box
            markerscale= 2.0,     # enlarge scatter markers in the legend for readability
        )

        # ── 10. Final adjustments and display ─────────────────────────────────
        # tight_layout() automatically adjusts padding so that labels, title,
        # and tick marks are never cut off at the figure edges.
        plt.tight_layout()

        if save_path:
            # bbox_inches='tight' ensures the saved image includes all labels.
            # dpi=150 produces a high-resolution file suitable for reports.
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        # plt.show() opens the interactive Matplotlib window.
        # The programme pauses here until the user closes the window.
        # plt.show()  # Commenté pour Streamlit
        # plt.close(fig)   # free memory after the window is closed

        # Return the figure object for Streamlit
        return fig
