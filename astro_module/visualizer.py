import matplotlib.pyplot as plt
import pandas as pd

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
    Generates a professional, annotated light-curve plot for the rolling-window
    (baseline-departure) detector.

    ── What this class produces ─────────────────────────────────────────────────
    A single Matplotlib figure containing:
      • Every cadence of the light curve as small scatter dots (one neutral cloud).
      • The rolling-mean baseline as a dashed blue line.
      • The detection envelope: the global baseline ± baseline_sigma × global_std —
        the exact threshold the detector uses (an event is flagged when the rolling
        mean leaves this band).
      • A coloured diamond + label on each detected event, placed on the rolling-mean
        curve where the departure lives (TRANSIT = purple, FLARE = orange,
        MICROLENSING = green).
      • A legend, axis labels, and a title summarising the detections.

    ── OOP reminder ─────────────────────────────────────────────────────────────
    AstroPlotter bundles together all the data it needs (stored in __init__) and
    the action it performs (show_results), so it is self-contained and reusable.
    ─────────────────────────────────────────────────────────────────────────────
    """

    # ── Colour palette — class-level constants ────────────────────────────────
    # Defined here (outside __init__) so they are shared by every AstroPlotter
    # object and can be tweaked in one place without touching any method logic.
    COLOUR_NORMAL   = '#B0C8E8'   # soft steel blue  — the cadence cloud
    COLOUR_BASELINE = '#2255AA'   # medium blue       — rolling-mean trend line
    COLOUR_BAND     = '#90B8D8'   # pale blue         — detection-envelope shading

    # Per-event-type colours make it easy to distinguish events at a glance.
    COLOUR_LABELS = {
        'TRANSIT'     : '#9B59B6',   # purple — planet transit dip
        'FLARE'       : '#E67E22',   # orange — stellar flare
        'MICROLENSING': '#27AE60',   # green  — compact-object microlensing
    }

    # Boolean to know if the instance is for streamlit use
    for_streamlit: bool = False,

    def __init__(
        self,
        df_flagged        : pd.DataFrame,
        classified_results: list,
        target_id         : str   = "Unknown Target",
        window            : int   = 200,
        baseline_sigma    : float = None,
        global_median     : float = None,
        global_std        : float = None,
    ):
        """
        Store all the data needed to produce the plot.

        Parameters
        ----------
        df_flagged : pd.DataFrame
            Output of AnomalyDetector.detect_baseline_departures() — needs the
            'time', 'flux' and 'rolling_mean' columns.
        classified_results : list[dict]
            The events to annotate (from detect_baseline_departures()).  Each dict
            must have 'event_type', 'confidence', 'anomaly_peak_idx' and 'direction'.
        target_id : str
            The star identifier displayed in the figure title (e.g. 'KIC 11904151').
        window : int
            The rolling window size used during detection — shown in the legend.
        baseline_sigma : float or None
            If given, draw the detection envelope: global_median ± baseline_sigma ×
            global_std.  This is the exact threshold the detector used.
        global_median, global_std : float or None
            The global reference used by the detector.  If not supplied they are
            recomputed from the flux column.
        """
        # self.X = value stores each argument ON the object so that show_results()
        # can access all of them via self.X without needing them passed again.
        self.df                 = df_flagged
        self.classified_results = classified_results
        self.target_id          = target_id
        self.window             = window
        self.baseline_sigma     = baseline_sigma
        # Fall back to recomputing the global reference if the caller did not pass it.
        self.global_median = global_median if global_median is not None else float(df_flagged['flux'].median())
        self.global_std    = global_std    if global_std    is not None else float(df_flagged['flux'].std())

    def set_streamlit(self):
        self.for_streamlit = True

    def show_results(self, save_path: str = None):
        """
        Render and display the complete annotated light-curve figure.

        Parameters
        ----------
        save_path : str or None
            Optional file path to also save the figure to disk.
            If None (default), the figure is only shown on screen.
        """

        # ── 1. Global visual styling ──────────────────────────────────────────
        if _SEABORN_AVAILABLE:
            sns.set_theme(style='darkgrid', font_scale=1.1)
        else:
            plt.style.use('seaborn-v0_8-darkgrid')

        # A wide panoramic aspect ratio suits a time-series; dpi=120 keeps it crisp.
        fig, ax = plt.subplots(figsize=(16, 6), dpi=120)

        # ── 2. Data scatter (one neutral cloud) ───────────────────────────────
        # Detection is conveyed by the rolling-window layer below, so every cadence
        # is drawn in a single discreet colour.
        ax.scatter(
            self.df['time'],
            self.df['flux'],
            color  = self.COLOUR_NORMAL,
            s      = 2,
            alpha  = 0.5,
            zorder = 1,
            label  = 'Cadence',
        )

        # ── 3. Rolling-mean baseline ──────────────────────────────────────────
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

        # ── 4. Detection envelope (baseline departure) ────────────────────────
        # An event is flagged when the rolling mean leaves the horizontal band
        # global_median ± baseline_sigma × global_std.  Drawing that band makes the
        # criterion legible at a glance.
        if self.baseline_sigma is not None:
            hi = self.global_median + self.baseline_sigma * self.global_std
            lo = self.global_median - self.baseline_sigma * self.global_std
            ax.axhline(self.global_median, color='#555555', linewidth=0.8,
                       linestyle=':', alpha=0.7, zorder=0, label='Global baseline')
            ax.fill_between(
                self.df['time'], lo, hi,
                color=self.COLOUR_BAND, alpha=0.18, zorder=0,
                label=f'±{self.baseline_sigma}σ baseline envelope',
            )

        # ── 5. Annotate each detected event ───────────────────────────────────
        # Each event lives on the ROLLING-MEAN curve, so we mark it there with a
        # coloured diamond.  The label is offset in SCREEN PIXELS (scale-safe: the
        # flux axis spans only ~0.x %, so a data-unit offset would fly off-screen).
        # Only the first diamond of each type is labelled → one legend entry per
        # event type, in the correct colour.
        seen_types = set()
        for event in self.classified_results:
            peak_row   = self.df.loc[event['anomaly_peak_idx']]
            peak_time  = peak_row['time']
            peak_y     = peak_row['rolling_mean']
            event_type = event['event_type']
            confidence = event['confidence']
            colour     = self.COLOUR_LABELS.get(event_type, self.COLOUR_BASELINE)

            ax.scatter([peak_time], [peak_y], color=colour, marker='D',
                       s=70, edgecolors='white', linewidths=1.0, zorder=4,
                       label=(event_type if event_type not in seen_types else None))
            seen_types.add(event_type)

            # Label below a dip, above a bump.
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

        # ── 6. Axis labels and title ──────────────────────────────────────────
        ax.set_xlabel('Time (days  —  BKJD)', fontsize=12, labelpad=8)
        ax.set_ylabel('Normalised Flux  (1.0 = typical brightness)', fontsize=12, labelpad=8)

        counts = {t: sum(1 for ev in self.classified_results
                         if ev.get('event_type') == t)
                  for t in ('TRANSIT', 'FLARE', 'MICROLENSING')}
        parts = [f'{n} {t.lower()}' for t, n in counts.items() if n]
        summary = '   •   '.join(parts) if parts else 'no events'
        ax.set_title(
            f'Automated Event Detection  —  {self.target_id}\n'
            f'{summary}  (rolling-window baseline departure)',
            fontsize=13, fontweight='bold', pad=14,
        )

        # ── 7. Legend ─────────────────────────────────────────────────────────
        # Every drawn artist carries a 'label=', so the automatic legend is complete.
        ax.legend(
            loc        = 'upper right',
            fontsize   = 9,
            framealpha = 0.85,
            markerscale= 2.0,
        )

        # ── 8. Final adjustments and display ──────────────────────────────────
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        if self.for_streamlit :
            # Return the figure object for Streamlit
            self.for_streamlit = False
            plt.savefig('astronomical_detector.png')
            plt.close(fig)   # free memory after the window is closed
        else:
            plt.show()
            plt.close(fig)   # free memory after the window is closed