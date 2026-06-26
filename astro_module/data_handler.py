import warnings
# Suppress non-critical library warnings (deprecation notices, etc.) so the
# programme output stays clean and readable for the user.
warnings.filterwarnings('ignore')

import lightkurve as lk  # NASA's official Python library for reading telescope light curves
import pandas as pd       # Pandas: the standard library for tabular data (rows + named columns = DataFrame)
import numpy as np        # NumPy: the foundation of scientific computing in Python (fast array maths)
import os                 # Standard library for file-system operations (checking paths, creating folders)


class AstroFetcher:
    """
    Class responsible for fetching light curves from NASA archives
    or loading local datasets (e.g., from Kaggle).

    ── OOP Concept: What is a Class? ───────────────────────────────────────────
    A class is a blueprint for creating objects.  It bundles together:
      • attributes  — data the object stores (e.g. which target star it works on)
      • methods     — actions the object can perform (e.g. download data)

    Think of it like a cookie cutter: the class is the cutter, each object
    you create from it is an individual cookie that shares the same shape
    but can hold its own data.
    ─────────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, target_id):
        """
        ── OOP Concept: The Constructor (__init__) ──────────────────────────────
        __init__ is a special method (called a "dunder" = double-underscore method)
        that Python calls automatically the moment a new object is created:

            fetcher = AstroFetcher("KIC 11904151")
                                        ↑
                          Python passes this string to __init__ as 'target_id'

        ── OOP Concept: self ────────────────────────────────────────────────────
        'self' is a reference to the object being created — think of it as
        "this specific cookie, not the cutter."
        Writing  self.target_id = target_id  "glues" the value onto THIS object
        so that every other method can access it later via self.target_id
        without needing it passed as an argument again.
        ─────────────────────────────────────────────────────────────────────────

        Parameters
        ----------
        target_id : str
            Unique astronomical identifier of the star to observe.
            Examples:  "KIC 11904151"   (Kepler Input Catalog)
                       "TIC 261136679"  (TESS Input Catalog)
        """
        # Attach the star identifier permanently to this object instance.
        # From now on, self.target_id is accessible in every method of this class.
        self.target_id = target_id

    def download_data(self, mission='Kepler'):
        """
        Query the NASA MAST online archive and download the first available
        light curve for this target.

        Parameters
        ----------
        mission : str
            The space telescope mission to query.
            Supported values: 'Kepler' or 'TESS'.

        Returns
        -------
        lightkurve.LightCurve
            A Lightkurve object containing time and flux arrays.
        """
        # missions_to_try is a list of missions we will query in order.
        # We always start with the requested mission; if it returns nothing we
        # automatically fall back to the other mission so the user still gets data.
        # 'Kepler' and 'TESS' both produce light curves compatible with the rest
        # of our pipeline (SignalCleaner handles both transparently).
        missions_to_try = [mission, 'TESS'] if mission.lower() == 'kepler' else [mission, 'Kepler']

        for current_mission in missions_to_try:
            print(f"Searching for data for target '{self.target_id}' (Mission: {current_mission})...")

            # Wrap the search in try-except because lightkurve can raise internal
            # exceptions (e.g. network timeout, unresolvable name).  We catch them
            # and move on to the next mission rather than crashing the whole programme.
            try:
                search_result = lk.search_lightcurve(self.target_id, mission=current_mission)
            except Exception as e:
                # Print the internal lightkurve error as a warning and continue.
                print(f"  Warning: search raised an error for {current_mission}: {e}")
                search_result = []   # treat as empty so we try the next mission

            # len() on a SearchResult (or empty list) returns the number of results.
            if len(search_result) > 0:
                print(f"  Found {len(search_result)} dataset(s) on {current_mission}. Downloading the first one...")
                # search_result[0] selects the first (most recent / highest quality)
                # observation.  .download() fetches the actual FITS file from NASA.
                return search_result[0].download()

            print(f"  No data found on {current_mission}.")

        # If we reach this point, every mission returned zero results.
        # Raise a clear, actionable error so the user knows what to fix.
        raise ValueError(
            f"\nNo light-curve data found for '{self.target_id}' in any mission.\n"
            f"Possible reasons:\n"
            f"  • The identifier may be misspelled (check capitalisation and spaces).\n"
            f"  • This star may not have been observed by Kepler or TESS.\n"
            f"Known working targets you can try:\n"
            f"  • KIC 11904151  (Kepler-10  — confirmed exoplanet system)\n"
            f"  • KIC 757076   (active flare star)\n"
            f"  • KIC 3558849  (microlensing candidate)\n"
        )

    def load_local_csv(self, filepath):
        """
        Load a light curve from a local CSV file instead of downloading from NASA.
        Useful for working offline or with Kaggle datasets.

        The CSV must have at least two columns named exactly 'time' and 'flux'.

        Parameters
        ----------
        filepath : str
            Full or relative path to the CSV file on disk.

        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame with at least the 'time' and 'flux' columns.
        """
        print(f"Loading local dataset from {filepath}...")

        # os.path.exists() checks whether a file or folder actually exists on disk.
        # Returning False here means the path is wrong or the file was deleted.
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")

        # pd.read_csv() reads a comma-separated values file into a DataFrame.
        # A DataFrame is Pandas' main data structure — think of it as an in-memory
        # spreadsheet where each column has a name and each row has an index number.
        df = pd.read_csv(filepath)

        # Validate that the required columns exist before proceeding.
        # Without 'time' we can't plot the signal on the X axis;
        # without 'flux' we have no brightness measurement to analyse.
        if 'time' not in df.columns or 'flux' not in df.columns:
            raise ValueError("The CSV must contain 'time' and 'flux' columns.")

        # len(df) returns the number of rows — a quick sanity check.
        print(f"Local data loaded successfully ({len(df)} rows).")
        return df


class SignalCleaner:
    """
    Class responsible for preprocessing light curve data.
    Adapts automatically to Lightkurve objects or Pandas DataFrames.

    IMPORTANT — why we do NOT use flatten():
    -----------------------------------------
    Lightkurve's flatten() fits and divides out a smooth trend over a long
    window (typically 401 points). This removes instrumental drift, which is
    useful — but it also suppresses exoplanet transit dips, because a ~15-point
    dip looks like a short-term trend to the smoother.

    Instead, we normalise the flux by its median so all values sit near 1.0.
    This keeps the relative depth of transits (~0.1–1% dips) intact while
    still making the signal scale-invariant and comparable across targets.

    IMPORTANT — why our 5σ clipping is ASYMMETRIC:
    ----------------------------------------------
    The same reasoning applies to outlier removal.  Cosmic rays and saturation
    push the flux UP, but transits push it DOWN — and a deep transit can be a
    −10σ point.  A symmetric ±5σ clip would therefore delete the transit bottoms
    along with the cosmic rays.  So we clip ONLY the upper side (mean + 5σ /
    sigma_upper=5) and leave every downward excursion untouched.
    """

    def __init__(self, raw_data):
        """
        Store the raw input data on the object for later processing.

        Parameters
        ----------
        raw_data : lightkurve.LightCurve or pd.DataFrame
            The unprocessed light curve coming from either
            AstroFetcher.download_data() or AstroFetcher.load_local_csv().
        """
        # self.data stores the raw input exactly as received.
        # We do NOT transform it here — transformation happens in process_data()
        # so each step has a clear, separate responsibility.
        self.data = raw_data

    def _trim_edges(self, df, trim_edges_days):
        """
        Drop every point within `trim_edges_days` of each end of the light curve.

        Why: the start (and end) of a Kepler quarter carries a thermal/focus
        "ramp".  After the spacecraft repoints (quarter roll, monthly downlink,
        safe-mode recovery) the optics temperature changes, the telescope focus
        drifts, and the fraction of starlight falling in the photometric aperture
        drifts with it — for roughly a day, until the spacecraft thermally
        settles.  That smooth drift is an INSTRUMENT artefact, not astrophysics,
        and the baseline-departure detector would otherwise flag it as a false
        event.  Trimming the unreliable edges removes it at the source.
        """
        if not trim_edges_days or trim_edges_days <= 0 or len(df) == 0:
            return df
        t = df['time']
        t0, t1 = t.min(), t.max()
        before = len(df)
        df = df[(t >= t0 + trim_edges_days) & (t <= t1 - trim_edges_days)]
        removed = before - len(df)
        if removed:
            print(f"Trimmed {removed} edge point(s) "
                  f"({trim_edges_days} d at each end — thermal-ramp guard).")
        return df

    def process_data(self, trim_edges_days: float = 0.0):
        """
        Clean and normalise the light curve regardless of its input format.

        Steps applied:
          1. Remove rows with missing values (NaN).
          2. Remove UPWARD outliers using the asymmetric 5-sigma rule.
          3. Normalise flux around 1.0 by dividing every value by the median.
          4. Optionally trim the quarter edges (thermal-ramp guard).

        Parameters
        ----------
        trim_edges_days : float
            If > 0, drop every cadence within this many days of the start and of
            the end of the light curve, to discard the start-of-quarter thermal
            ramp.  Default 0.0 keeps the previous behaviour (no trimming).

        Returns
        -------
        pd.DataFrame
            A cleaned DataFrame with 'time' and 'flux' columns, where
            flux ≈ 1.0 represents the star's typical observed brightness.
        """
        print("Cleaning data...")

        # isinstance(object, Type) is a Python built-in that returns True if
        # 'object' is an instance of 'Type'.  We use it to detect the input
        # format and apply the appropriate cleaning strategy for each case.
        if isinstance(self.data, pd.DataFrame):

            # ── Branch A: Pandas DataFrame (local CSV / Kaggle) ─────────────
            print("Data format: Pandas DataFrame. Applying standard statistical cleaning...")

            # Step 1 — Remove missing values
            # dropna() removes every row that has a missing value (NaN) in at
            # least one of the specified columns.
            # NaN ("Not a Number") appears when the telescope failed to record
            # a measurement — e.g. during a spacecraft safe-mode event or a
            # data-downlink interruption.
            # subset= limits the check to only 'time' and 'flux'; other columns
            # are ignored even if they contain NaNs.
            df = self.data.dropna(subset=['time', 'flux'])

            # Step 2 — 5-sigma outlier removal
            #
            # Mathematics background:
            #   mean  (μ) = sum of all values ÷ number of values  =  "average" brightness
            #   std   (σ) = standard deviation = a measure of how spread out the
            #               values are around the mean.
            #               Small σ → values cluster tightly around the mean.
            #               Large σ → values are widely spread.
            #
            # The 5-sigma rule (empirical):
            #   In a normally distributed dataset, 99.9999% of real data points
            #   fall within ±5σ of the mean.  Any point outside that range is
            #   so statistically extreme that it is almost certainly an artefact
            #   (cosmic-ray hit on the CCD, saturation, data corruption) —
            #   not a real astrophysical signal.
            mean_flux = df['flux'].mean()   # arithmetic average of all flux values
            std_flux  = df['flux'].std()    # standard deviation of all flux values

            # ASYMMETRIC 5σ rule — clip UPWARD outliers only.
            #
            # Cosmic-ray hits, hot pixels and saturation all push the flux UP, so
            # we remove anything above mean + 5σ.  But a transit pushes the flux
            # DOWN, and a deep transit can sit far below mean − 5σ (e.g. a 1% dip
            # in a star with 0.1% noise is a −10σ point).  A symmetric ±5σ filter
            # would delete exactly those transit bottoms — the signal we are
            # hunting — so we deliberately keep the lower side untouched.
            df = df[df['flux'] <= mean_flux + 5 * std_flux].copy()
            # .copy() creates a fully independent DataFrame.
            # Without it, Pandas may raise a "SettingWithCopyWarning" because
            # modifying a slice of the original can have unpredictable side effects.

            # Step 3 — Median normalisation
            #
            # The median is the middle value when all measurements are sorted.
            # It is more robust than the mean when a few extreme values remain,
            # because extreme outliers cannot "pull" it far from the centre.
            #
            # Dividing every flux value by the median rescales the whole signal:
            #   After normalisation, flux ≈ 1.0 means "typical brightness."
            #   flux > 1.0  →  star is brighter than usual  (e.g. a stellar flare)
            #   flux < 1.0  →  star is dimmer than usual    (e.g. a planet transit)
            #
            # The tiny constant 1e-10 (= 0.0000000001) is added to the denominator
            # as a safety guard against division-by-zero in the (highly unlikely)
            # event that the median is exactly 0.
            median_flux  = df['flux'].median()
            df['flux']   = df['flux'] / (median_flux + 1e-10)

            # Step 4 — Trim the unreliable quarter edges (thermal-ramp guard).
            df = self._trim_edges(df, trim_edges_days)

            print(f"Cleaning complete. ({len(df)} points retained)")

            # reset_index(drop=True) renumbers the rows 0, 1, 2, … continuously.
            # After dropping rows, "holes" appear in the index (e.g. 0, 3, 7, …).
            # Resetting gives a clean, gapless index so downstream code that
            # relies on integer positions works correctly.
            return df.reset_index(drop=True)

        else:
            # ── Branch B: Lightkurve object (NASA API) ───────────────────────
            print("Data format: Lightkurve Object. Applying astrophysical cleaning...")

            # Lightkurve provides convenient built-in methods we can chain:
            #   .remove_nans()          — drops cadences with missing flux values
            #   .remove_outliers(...)   — applies the sigma-clipping rule
            #
            # ASYMMETRIC clipping (sigma_upper=5, sigma_lower=inf): we remove only
            # UPWARD outliers (cosmic rays / hot pixels spike up) and keep every
            # downward excursion.  A symmetric sigma=5 clip would delete the bottom
            # of deep transits (a 1% dip can be ~10σ) — the exact signal we want.
            # NO .flatten() either — see the class docstring for the full reason.
            clean_lc = self.data.remove_nans().remove_outliers(
                sigma_upper=5, sigma_lower=np.inf
            )

            # Lightkurve stores flux as an AstroPy Quantity object (number + unit).
            # .value strips away the unit and returns a plain NumPy array,
            # which is what Pandas and NumPy functions expect as input.
            flux_values = clean_lc.flux.value

            # np.median() computes the median of a NumPy array.
            # float() converts the single-element NumPy result to a regular Python
            # float — cleaner for the arithmetic operation that follows.
            median_flux = float(np.median(flux_values))

            # Build a clean Pandas DataFrame from the Lightkurve arrays.
            # clean_lc.time.value gives timestamps in Barycentric Kepler Julian
            # Date (BKJD): days elapsed since 2009-01-01 noon UTC, corrected
            # for Earth's orbital motion around the Sun.
            df = pd.DataFrame({
                'time': clean_lc.time.value,
                'flux': flux_values / (median_flux + 1e-10)  # normalise around 1.0
            })

            # Trim the unreliable quarter edges (thermal-ramp guard).
            df = self._trim_edges(df, trim_edges_days)

            print(f"Cleaning complete. ({len(df)} points retained)")
            return df.reset_index(drop=True)

    def save_to_csv(self, df, filename="clean_data.csv"):
        """
        Persist the cleaned DataFrame to disk as a CSV file.

        The file is saved at  data/processed/<filename>  relative to the
        current working directory.  The folder is created automatically
        if it does not already exist.

        Parameters
        ----------
        df       : pd.DataFrame  — the cleaned light curve to save
        filename : str           — desired output file name
        """
        # os.path.join() builds a file-system path from separate components.
        # On Mac/Linux it produces "data/processed"; on Windows "data\\processed".
        # Always prefer os.path.join() over manual string concatenation with "/"
        # to keep the code portable across operating systems.
        save_folder = os.path.join("data", "processed")

        # os.makedirs() creates the directory and any missing intermediate parents.
        # exist_ok=True suppresses the error that would otherwise be raised if
        # the directory already exists — safe to call repeatedly.
        os.makedirs(save_folder, exist_ok=True)

        # Combine the folder path and the filename into one complete path string.
        full_path = os.path.join(save_folder, filename)

        # df.to_csv() serialises the DataFrame to disk in CSV format.
        # index=False tells Pandas NOT to write the integer row index (0, 1, 2, …)
        # as an extra first column — those numbers carry no scientific meaning
        # and would just clutter the output file.
        df.to_csv(full_path, index=False)
        print(f"File successfully saved at: {full_path}")