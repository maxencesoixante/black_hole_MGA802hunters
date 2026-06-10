import warnings
warnings.filterwarnings('ignore')

import lightkurve as lk
import pandas as pd
import numpy as np
import os


class AstroFetcher:
    """
    Class responsible for fetching light curves from NASA archives
    or loading local datasets (e.g., from Kaggle).
    """

    def __init__(self, target_id):
        self.target_id = target_id

    def download_data(self, mission='Kepler'):
        """Fetches data from the NASA API."""
        print(f"Searching for online data for target {self.target_id} (Mission: {mission})...")
        search_result = lk.search_lightcurve(self.target_id, mission=mission)

        if len(search_result) == 0:
            raise ValueError(f"No online data found for {self.target_id}.")

        print(f"Online data found. Downloading...")
        return search_result[0].download()

    def load_local_csv(self, filepath):
        """
        Loads light curve data from a local CSV file.
        Expects 'time' and 'flux' columns.
        """
        print(f"Loading local dataset from {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")

        df = pd.read_csv(filepath)

        if 'time' not in df.columns or 'flux' not in df.columns:
            raise ValueError("The CSV must contain 'time' and 'flux' columns.")

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
    """

    def __init__(self, raw_data):
        self.data = raw_data

    def process_data(self):
        print("Cleaning data...")

        # 1. Pandas DataFrame (local CSV / Kaggle)
        if isinstance(self.data, pd.DataFrame):
            print("Data format: Pandas DataFrame. Applying standard statistical cleaning...")

            df = self.data.dropna(subset=['time', 'flux'])

            # Remove extreme outliers (5 sigma)
            mean_flux = df['flux'].mean()
            std_flux = df['flux'].std()
            df = df[
                (df['flux'] >= mean_flux - 5 * std_flux) &
                (df['flux'] <= mean_flux + 5 * std_flux)
            ].copy()

            # Normalize around median
            median_flux = df['flux'].median()
            df['flux'] = df['flux'] / (median_flux + 1e-10)

            print(f"Cleaning complete. ({len(df)} points retained)")
            return df.reset_index(drop=True)

        # 2. Lightkurve object (NASA API)
        else:
            print("Data format: Lightkurve Object. Applying astrophysical cleaning...")

            # Remove NaNs and sigma-clip outliers — NO flatten() to preserve transit dips
            clean_lc = self.data.remove_nans().remove_outliers(sigma=5)

            # Normalize flux around median so values sit near 1.0
            flux_values = clean_lc.flux.value
            median_flux = float(np.median(flux_values))

            df = pd.DataFrame({
                'time': clean_lc.time.value,
                'flux': flux_values / (median_flux + 1e-10)
            })

            print(f"Cleaning complete. ({len(df)} points retained)")
            return df.reset_index(drop=True)

    def save_to_csv(self, df, filename="clean_data.csv"):
        """Saves the cleaned DataFrame to a local CSV file."""
        save_folder = os.path.join("data", "processed")
        os.makedirs(save_folder, exist_ok=True)
        full_path = os.path.join(save_folder, filename)
        df.to_csv(full_path, index=False)
        print(f"File successfully saved at: {full_path}")