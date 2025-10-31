import os
import pandas as pd
import logging

class DataIngestion:
    def __init__(self, data_path: str, log_dir: str = "logs"):
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Logging setup
        self.logger = logging.getLogger("Data-Handler")
        self.logger.setLevel(logging.DEBUG)

        # Handlers
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        fh = logging.FileHandler(os.path.join(log_dir, "data_history.log"))
        fh.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Attach handlers only once (to prevent duplicates on re-import)
        if not self.logger.handlers:
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        # Store paths
        self.data_path = data_path
        self.logger.info("DataIngestion initialized.")

    def load_data(self) -> pd.DataFrame:
        """Load dataset into memory."""
        try:
            self.logger.info(f"Attempting to load dataset from: {self.data_path}")
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df

        except FileNotFoundError:
            self.logger.error(f"File not found at: {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("The CSV file is empty.")
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error while loading data: {str(e)}")
            raise

    def preview_data(self, df: pd.DataFrame, n: int = 5):
        """Preview top records of the dataset."""
        self.logger.debug(f"Previewing first {n} records:")
        self.logger.debug(f"\n{df.head(n).to_string(index=False)}")

    def validate_columns(self, df: pd.DataFrame, expected_cols: list):
        """Validate if required columns exist."""
        self.logger.info("Validating dataset columns...")
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            self.logger.warning(f"Missing columns: {missing}")
        else:
            self.logger.info("All expected columns found.")
        return missing == []

if __name__ == "__main__":
    data_path = "./experiments/SPAS-Dataset-BD.csv"
    ingestion = DataIngestion(data_path=data_path, log_dir='./LOGS')

    df = ingestion.load_data()
    ingestion.preview_data(df)
    ingestion.validate_columns(df, expected_cols=["district", "crop", "season"])
