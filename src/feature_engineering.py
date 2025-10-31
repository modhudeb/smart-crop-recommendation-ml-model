import os
import logging
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_score
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
import shap

# --- Logger Setup ---
logger = logging.getLogger('Feature-Engineering')
logger.setLevel(logging.DEBUG)

log_loc = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_loc, exist_ok=True)
fh = logging.FileHandler(os.path.join(log_loc, 'feature_engineering.log'))
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
logger.addHandler(fh)


class FeatureEngineering:
    def __init__(self, df_valid, cat_cols):
        self.df_valid = df_valid.copy()
        self.cat_cols = cat_cols
        self.mi_scores = None
        self.gain_ratios = None
        self.shap_values = None
        self.important_features = None

    def compute_information_gain(self):
        logger.info("Calculating Mutual Information (Information Gain) scores.")
        X = self.df_valid.drop(columns=['area', 'ap_ratio', 'production', 'crop_name_enc', 'production_log'], errors='ignore')
        y = self.df_valid['crop_name_enc']
        discrete_mask = X.columns.isin(self.cat_cols)

        mi_scores = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=42)
        self.mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        logger.debug(f"Top 5 MI Scores:\n{self.mi_scores.head()}")
        return self.mi_scores

    def compute_gain_ratio(self):
        logger.info("Calculating Gain Ratio for discrete features.")
        X = self.df_valid.drop(columns=['area', 'ap_ratio', 'production', 'crop_name_enc', 'production_log'], errors='ignore')
        y = self.df_valid['crop_name_enc']
        discrete_mask = X.columns.isin(self.cat_cols)

        ratios = {}
        for col in X.columns[discrete_mask]:
            ig = mutual_info_score(X[col], y)
            h = entropy(pd.Series(X[col]).value_counts(normalize=True), base=2)
            ratios[col] = ig / h if h != 0 else 0

        self.gain_ratios = pd.Series(ratios).sort_values(ascending=False)
        logger.debug(f"Top 5 Gain Ratios:\n{self.gain_ratios.head()}")
        return self.gain_ratios

    def compute_shap_importance(self):
        logger.info("Computing SHAP feature importance with RandomForestClassifier.")
        X = self.df_valid.drop(columns=['area', 'ap_ratio', 'production', 'crop_name_enc', 'production_log'], errors='ignore')
        y = self.df_valid['crop_name_enc']

        rf_fast = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_fast.fit(X, y)

        explainer = shap.TreeExplainer(rf_fast)
        X_sample = shap.sample(X, 1000)
        shap_values = explainer.shap_values(X_sample)
        self.shap_values = shap_values

        # Compute mean absolute SHAP value for each feature
        feature_importance = np.abs(shap_values).mean(axis=(0, 1))
        self.important_features = pd.Series(feature_importance, index=X.columns).sort_values(ascending=False)
        logger.debug(f"Top 5 SHAP Features:\n{self.important_features.head()}")
        return self.important_features

    def run(self):
        logger.info("Starting feature engineering pipeline.")
        self.compute_information_gain()
        self.compute_gain_ratio()
        self.compute_shap_importance()
        logger.info("Feature engineering pipeline completed successfully.")
        return {
            'mi_scores': self.mi_scores,
            'gain_ratios': self.gain_ratios,
            'shap_importance': self.important_features
        }


if __name__ == "__main__":
    logger.info("Feature engineering module executed as script. Please integrate with data pipeline.")
