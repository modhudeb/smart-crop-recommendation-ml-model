import os
import pandas as pd
import logging

class DataIngestion:
    def __init__(self, data_path: str, save_path:str, log_dir: str = "logs"):
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
        self.save_path = save_path
        self.logger.info("DataIngestion initialized.")

    def load_data(self) -> pd.DataFrame:
        """Load dataset into memory."""
        try:
            self.logger.info(f"Attempting to load dataset from: {self.data_path}")
            df = pd.read_csv(self.data_path)
            dir_name = os.path.dirname(self.save_path)
            os.makedirs(dir_name, exist_ok=True)
            df.to_csv(self.save_path, index=False)
            self.logger.info(f"Successfully saved {os.path.basename(self.save_path)} to: {dir_name}")
            # return df

        except FileNotFoundError:
            self.logger.error(f"File not found at: {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("The CSV file is empty.")
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error while loading data: {str(e)}")
            raise

    def preview_data(self, n: int = 5):
        """Preview top records of the dataset."""
        df = pd.read_csv(self.save_path)
        self.logger.debug(f"Previewing first {n} records:")
        self.logger.debug(f"\n{df.head(n).to_string(index=False)}")


if __name__ == "__main__":
    data_path = "./experiments/SPAS-Dataset-BD.csv"
    save_path = "./data/raw/SPAS-Dataset-BD.csv"
    
    ingestion = DataIngestion(data_path=data_path, save_path=save_path, log_dir='./logs')
    ingestion.load_data()
    ingestion.preview_data(n=2)
