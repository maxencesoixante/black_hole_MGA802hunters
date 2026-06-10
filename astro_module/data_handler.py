import warnings
warnings.filterwarnings('ignore') # Masque les avertissements inoffensifs
import lightkurve as lk
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')


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
    Class responsible for preprocessing data.
    It adapts automatically to Lightkurve objects or Pandas DataFrames.
    """

    def __init__(self, raw_data):
        self.data = raw_data

    def process_data(self):
        print("Cleaning data...")

        # 1. If the data is a Pandas DataFrame (Local CSV / Kaggle)
        if isinstance(self.data, pd.DataFrame):
            print("Data format: Pandas DataFrame. Applying standard statistical cleaning...")
            # Remove missing values
            df = self.data.dropna(subset=['time', 'flux'])

            # Remove extreme mathematical outliers (5 standard deviations)
            mean_flux = df['flux'].mean()
            std_flux = df['flux'].std()
            df = df[(df['flux'] >= mean_flux - 5 * std_flux) & (df['flux'] <= mean_flux + 5 * std_flux)]

            print("Cleaning complete.")
            return df

        # 2. If the data is a Lightkurve object (NASA API)
        else:
            print("Data format: Lightkurve Object. Applying astrophysical cleaning...")
            clean_lc = self.data.remove_nans()
            flat_lc = clean_lc.flatten(window_length=401).remove_outliers(sigma=5)

            df = pd.DataFrame({
                'time': flat_lc.time.value,
                'flux': flat_lc.flux.value
            })

            print("Cleaning complete.")
            return df

    def save_to_csv(self, df, filename="clean_data.csv"):
        """Saves the cleaned DataFrame to a local CSV file."""
        save_folder = os.path.join("data", "processed")
        os.makedirs(save_folder, exist_ok=True)
        full_path = os.path.join(save_folder, filename)
        df.to_csv(full_path, index=False)
        print(f"File successfully saved at: {full_path}")