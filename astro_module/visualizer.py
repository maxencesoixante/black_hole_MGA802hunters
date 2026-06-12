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

    def __init__(
        self,
        df_flagged        : pd.DataFrame,
        classified_results: list,
        target_id         : str   = "Unknown Target",
        sigma_threshold   : float = 3.0,
        window            : int   = 200,
    ):
        """
        Store all the data needed to produce the plot.

        Parameters
        ----------
        df_flagged : pd.DataFrame
            Output of AnomalyDetector.detect().
            Required columns: 'time', 'flux', 'rolling_mean', 'rolling_std', 'is_anomaly'.
        classified_results : list[dict]
            Output of AnomalyClassifier.classify_all() — list of event dicts.
            Each dict must have: 'event_type', 'confidence', 'anomaly_peak_idx'.
        target_id : str
            The star identifier displayed in the figure title (e.g. 'KIC 11904151').
        sigma_threshold : float
            The σ value used during detection — needed to draw the correct band.
            Must match the value passed to AnomalyDetector.
        window : int
            The rolling window size used during detection — shown in the legend.
        """
        # self.X = value stores each argument ON the object so that show_results()
        # can access all of them via self.X without needing them passed again.
        self.df                 = df_flagged
        self.classified_results = classified_results
        self.target_id          = target_id
        self.sigma_threshold    = sigma_threshold
        self.window             = window

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

        # ── 2. Split the data into normal and anomalous cadences ──────────────
        # Boolean indexing: df[mask] keeps only the rows where mask is True.
        # ~ is the NOT operator for boolean arrays: ~True = False, ~False = True.
        df_normal  = self.df[~self.df['is_anomaly']]   # rows where is_anomaly = False
        df_anomaly = self.df[ self.df['is_anomaly']]   # rows where is_anomaly = True

        # ── 3. Plot normal cadences ───────────────────────────────────────────
        # ax.scatter() draws individual dots — ideal here because:
        #   • the telescope samples are not perfectly evenly spaced (observation gaps)
        #   • connecting thousands of points with lines would create visual noise
        # Key parameters:
        #   s     = marker size in points² (small: there are thousands of points)
        #   alpha = opacity from 0 (invisible) to 1 (fully opaque); 0.5 lets
        #           overlapping points show through without becoming a solid blob
        #   zorder = drawing order; lower values are drawn FIRST (behind other elements)
        ax.scatter(
            df_normal['time'],    # X axis: observation time in days
            df_normal['flux'],    # Y axis: normalised flux  (1.0 = typical brightness)
            color  = self.COLOUR_NORMAL,
            s      = 2,
            alpha  = 0.5,
            zorder = 1,
            label  = 'Normal cadence',
        )

        # ── 4. Plot anomalous cadences ────────────────────────────────────────
        # Same scatter call but with vivid red, larger markers, higher opacity,
        # and a higher zorder so they appear ON TOP of normal points.
        ax.scatter(
            df_anomaly['time'],
            df_anomaly['flux'],
            color  = self.COLOUR_ANOMALY,
            s      = 18,          # larger so they are immediately visible in a dense plot
            alpha  = 0.95,
            zorder = 3,
            label  = 'Detected anomaly',
        )

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

        # ── 6. Confidence band  (±σ detection envelope) ───────────────────────
        # ax.fill_between() shades the area between two Y curves at every X value.
        # Any point OUTSIDE this shaded band was flagged as an anomaly by the detector.
        upper_band = self.df['rolling_mean'] + self.sigma_threshold * self.df['rolling_std']
        lower_band = self.df['rolling_mean'] - self.sigma_threshold * self.df['rolling_std']

        ax.fill_between(
            self.df['time'],
            lower_band,
            upper_band,
            color  = self.COLOUR_BAND,
            alpha  = 0.15,        # very transparent — just a hint, does not hide data
            zorder = 0,           # drawn first, behind everything else
            label  = f'±{self.sigma_threshold}σ detection band',
        )

        # ── 7. Annotate each classified event ─────────────────────────────────
        # For each real (non-NOISE) event, add a text label with a pointer arrow
        # pointing to the most deviant point in the event window.
        for event in self.classified_results:
            # .loc[row_index] selects ONE row by its integer index and returns
            # a Pandas Series (like a dictionary of column → value).
            peak_row   = self.df.loc[event['anomaly_peak_idx']]
            peak_time  = peak_row['time']
            peak_flux  = peak_row['flux']
            event_type = event['event_type']
            confidence = event['confidence']

            # Pick the annotation colour for this event type, default to red.
            colour = self.COLOUR_LABELS.get(event_type, self.COLOUR_ANOMALY)

            # ax.annotate() draws a text label and an optional arrow.
            # xy       = (x, y) coordinates of the ARROWHEAD (the data point)
            # xytext   = (x, y) coordinates of the TEXT BOX (offset above the point)
            # The vertical offset of 0.03 in flux units ≈ 3% brightness — enough
            # to clear the marker without going off-screen on most real datasets.
            ax.annotate(
                f"{event_type}\n({confidence:.0%})",       # label: two lines of text
                xy         = (peak_time, peak_flux),       # arrowhead at the data point
                xytext     = (peak_time, peak_flux + 0.03),# text box slightly above
                fontsize   = 8.5,
                color      = colour,
                fontweight = 'bold',
                ha         = 'center',   # horizontal alignment: centre the text on xytext
                va         = 'bottom',   # vertical alignment: text sits above xytext
                arrowprops = dict(
                    arrowstyle = '->',   # simple arrow with a pointed tip
                    color      = colour,
                    lw         = 1.5,    # line width of the arrow shaft
                ),
                zorder = 5,   # drawn last, on top of all other elements
            )

        # ── 8. Axis labels and title ──────────────────────────────────────────
        # labelpad adds space between the axis line and the label text.
        ax.set_xlabel('Time (days  —  BKJD)', fontsize=12, labelpad=8)
        ax.set_ylabel('Normalised Flux  (1.0 = typical brightness)', fontsize=12, labelpad=8)

        # Multi-line title: first line identifies the target; second line gives
        # a quick summary of the detection results.
        n_anomalous = int(self.df['is_anomaly'].sum())
        n_events    = len(self.classified_results)
        ax.set_title(
            f'Automated Anomaly Detection  —  {self.target_id}\n'
            f'{n_anomalous} anomalous cadence(s) detected   •   '
            f'{n_events} astrophysical event(s) classified',
            fontsize   = 13,
            fontweight = 'bold',
            pad        = 14,
        )

        # ── 9. Legend ─────────────────────────────────────────────────────────
        # ax.get_legend_handles_labels() retrieves the automatic legend entries
        # created by the 'label=' arguments in scatter/plot/fill_between above.
        base_handles, base_labels = ax.get_legend_handles_labels()

        # ax.annotate() does NOT create automatic legend entries, so we build
        # custom Patch objects (coloured rectangles) for each event type.
        # mpatches.Patch() creates a simple filled rectangle — enough to show
        # "this colour = this event type" in the legend.
        extra_handles = [
            mpatches.Patch(facecolor=colour, edgecolor='none', label=etype)
            for etype, colour in self.COLOUR_LABELS.items()
        ]
        extra_labels = list(self.COLOUR_LABELS.keys())

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
        plt.show()
        plt.close(fig)   # free memory after the window is closed
