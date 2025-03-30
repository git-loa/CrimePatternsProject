"""
This module contains the DataHandler class
"""

from .feature_engineering import FeatureEngineering


class DataHandler:
    """
    Class to handle data
    """

    def __init__(self, data, config):
        """
        Handles dataset loading, filtering, and preprocessing.
        Parameters:
            data (pd.DataFrame): The dataset to be managed.
            config (dict): Configuration settings.
        """
        self.data = data
        self.config = config

        self.feature_engineering = FeatureEngineering(config)

        # Register features
        self.feature_engineering.register_feature(
            "compute_clearance_rate", FeatureEngineering.compute_clearance_rate
        )
        self.feature_engineering.register_feature(
            "compute_population_density", FeatureEngineering.compute_population_density
        )
        self.feature_engineering.register_feature(
            "add_adjusted_expenditures", FeatureEngineering.add_adjusted_expenditures
        )
        self.feature_engineering.register_feature(
            "compute_adjusted_income", FeatureEngineering.compute_adjusted_income
        )
        self.feature_engineering.register_feature(
            "compute_house_affordability",
            FeatureEngineering.compute_house_affordability,
        )
        self.feature_engineering.register_feature(
            "compute_home_ownership_rate",
            FeatureEngineering.compute_home_ownership_rate,
        )
        self.feature_engineering.register_feature(
            "compute_mobile_home_ratio", FeatureEngineering.compute_mobile_home_ratio
        )
        self.feature_engineering.register_feature(
            "compute_vacancy_rate", FeatureEngineering.compute_vacancy_rate
        )
        self.feature_engineering.register_feature(
            "compute_persons_and_household_metrics",
            FeatureEngineering.compute_persons_and_household_metrics,
        )

    def preprocess_data(self, category):
        """
        Preprocesses the data: filters by category, computes the target variable (crime rate),
        and applies feature engineering.
        """
        # Filter the data by category
        filtered_data = self.filter_data(category)

        # Compute the crime rate (target variable)
        filtered_data = self.compute_crime_rate(filtered_data)

        # Apply feature engineering to generate additional features
        filtered_data = self.feature_engineering.apply_features(filtered_data)
        return filtered_data

    def compute_crime_rate(self, df):
        """
        Computes the crime rate as Total Crimes per 1,000 people and adds it to the dataset.
        Parameters:
            data (pd.DataFrame): The input dataset.
        Returns:
            pd.DataFrame: The dataset with the crime rate as the target column.
        """
        data = df.copy()
        if "Violent_sum" in data.columns and "Population" in data.columns:
            data["crime_rate"] = data["Violent_sum"] / data["Population"]
            return data

        raise ValueError(
            "Columns 'Violent_sum' and 'Population' are required to compute the crime rate."
        )

    def filter_data(self, category):
        """
        Filters the dataset by category (e.g., Urban, Rural).
        """
        if category == "all":
            return self.data
        if category in ["Urban", "Suburban", "Rural"]:
            return self.data[self.data["Category"] == category]
        if category in self.data["County"].values:
            return self.data[self.data["County"] == category]
        raise ValueError("Invalid category specified.")
