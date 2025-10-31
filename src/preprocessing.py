import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

class Preprocessing:
    """
    Preprocessing pipeline for SPAS-Dataset-BD.
    Handles:
        - Column formatting
        - Month string cleaning and vectorization
        - Numeric conversions and AP ratio calculation
        - Label encoding for categorical features
        - Saving processed dataset to disk
    """

    def __init__(self, df: pd.DataFrame, save_path: str = "./data/processed/processed_data.csv", log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Logger setup
        self.logger = logging.getLogger("Preprocessing")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "preprocessing.log"))
            ch.setLevel(logging.DEBUG)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.df = df.copy()
        self.save_path = save_path
        self.logger.info("Preprocessing initialized.")

        # Month mapping
        self.months = ['january', 'february', 'march', 'april', 'may', 'june',
                       'july', 'august', 'september', 'october', 'november', 'december']
        self.month_to_idx = {m: i for i, m in enumerate(self.months)}
        self.month_abbr_map = {
            'jan': 'january', 'feb': 'february', 'mar': 'march', 'apr': 'april', 'may': 'may',
            'jun': 'june', 'jul': 'july', 'aug': 'august', 'sep': 'september',
            'oct': 'october', 'nov': 'november', 'dec': 'december'
        }

    def clean_columns(self):
        self.logger.info("Cleaning column names and categorical string columns.")
        self.df.columns = self.df.columns.str.strip().str.replace(' ', '_').str.lower()
        self.df['season'] = self.df['season'].str.strip().str.lower()
        self.df['crop_name'] = self.df['crop_name'].str.strip().str.lower()

    def clean_month_string(self, s):
        s = str(s).strip().lower()
        for abbr, full in self.month_abbr_map.items():
            s = re.sub(fr'\b{abbr}\b', full, s)
        return s

    def month_range_to_vector(self, s):
        s = self.clean_month_string(s)
        split = re.split(r'\s+to\s+', s)
        vec = [0] * 12
        if len(split) == 2:
            start, end = split
            if start in self.month_to_idx and end in self.month_to_idx:
                i1, i2 = self.month_to_idx[start], self.month_to_idx[end]
                if i1 <= i2:
                    for i in range(i1, i2 + 1): vec[i] = 1
                else:
                    for i in range(i1, 12): vec[i] = 1
                    for i in range(0, i2 + 1): vec[i] = 1
        return vec

    def process_growth_harvest(self):
        self.logger.info("Processing growth and harvest months into vectors.")
        growth_vectors = self.df['growth'].apply(self.month_range_to_vector).to_list()
        harvest_vectors = self.df['harvest'].apply(self.month_range_to_vector).to_list()

        growth_df = pd.DataFrame(growth_vectors, columns=[f'growth_{m[:3]}' for m in self.months])
        harvest_df = pd.DataFrame(harvest_vectors, columns=[f'harvest_{m[:3]}' for m in self.months])

        self.df = pd.concat([self.df, growth_df, harvest_df], axis=1)
        self.df.drop(columns=['growth', 'harvest'], inplace=True)

    def process_transplant(self):
        self.logger.info("Cleaning and encoding transplant month.")
        self.df['transplant'] = self.df['transplant'].apply(self.clean_month_string)
        self.df['transplant_month'] = self.df['transplant'].apply(
            lambda x: self.month_to_idx[x] if x in self.month_to_idx else np.nan
        )
        self.df.drop(columns=['transplant'], inplace=True)

    def convert_numeric(self):
        self.logger.info("Converting area, production, and weather columns to numeric.")
        self.df['area'] = pd.to_numeric(self.df['area'], errors='coerce').astype('Int64')
        self.df['production'] = pd.to_numeric(self.df['production'], errors='coerce').astype('Int64')
        self.df['ap_ratio'] = (self.df['area'] / self.df['production']).astype(float)

        weather_cols = ['avg_temp', 'min_temp', 'max_temp', 'avg_humidity',
                        'min_relative_humidity', 'max_relative_humidity']
        self.df[weather_cols] = self.df[weather_cols].apply(pd.to_numeric, errors='coerce')

    def encode_categories(self):
        self.logger.info("Encoding categorical columns with LabelEncoder.")
        # Remove invalid rows
        self.df = self.df[self.df['crop_name'] != '#ref!']
        self.df = self.df[self.df['season'].notna()].copy()

        # Label encoding
        self.df['crop_name_enc'] = LabelEncoder().fit_transform(self.df['crop_name'])
        self.df['season_enc'] = LabelEncoder().fit_transform(self.df['season'])
        self.df['district_enc'] = LabelEncoder().fit_transform(self.df['district'])

        self.df.drop(columns=['crop_name', 'season', 'district'], inplace=True)

    def save_processed(self):
        try:
            self.df.to_csv(self.save_path, index=False)
            self.logger.info(f"Processed dataset saved to: {self.save_path}")
        except Exception as e:
            self.logger.exception(f"Error saving processed data: {str(e)}")
            raise

    def run(self):
        self.clean_columns()
        self.process_growth_harvest()
        self.process_transplant()
        self.convert_numeric()
        self.encode_categories()
        self.save_processed()
        self.logger.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    df_raw = pd.read_csv("./data/raw/SPAS-Dataset-BD.csv")
    preprocessor = Preprocessing(df=df_raw, save_path="./data/processed/processed_data.csv", log_dir="./logs")
    preprocessor.run()
