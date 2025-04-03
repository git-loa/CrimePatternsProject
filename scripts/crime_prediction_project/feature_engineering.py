"""
This module contains the class FeatureEngineering.
"""


class FeatureEngineering:
    """
    Feature enginnering class to handle feature creation
    """

    def __init__(self, config):
        """
        Initializes the FeatureEngineering class with configuration.
        Parameters:
            config (dict): Configuration dictionary for feature handling.
        """
        self.config = config
        self.feature_registry = {}

    def register_feature(self, name, func):
        """
        Registers a feature function in the registry.
        Parameters:
            name (str): The name of the feature.
            func (callable): The function to compute the feature.
        """
        self.feature_registry[name] = func

    def apply_features(
        self,
        df,
    ):
        """
        Applies the selected feature functions to the dataset.
        Parameters:
            data (pd.DataFrame): The input dataset.
        Returns:
            pd.DataFrame: The dataset with applied features.
        """

        data = df.copy()

        for feature_name in self.config["features_to_apply"]:
            if feature_name in self.feature_registry:
                feature_func = self.feature_registry[feature_name]
                data = feature_func(data)
            else:
                raise ValueError(f"Feature '{feature_name}' is not registered.")
        return data

    # Feature Function: Compute Clearance Rate
    @staticmethod
    def compute_clearance_rate(df):
        """
        Computes the clearance rate as ViolentClr_sum / Violent_sum.
        """
        data = df.copy()
        if "ViolentClr_sum" in data.columns and "Violent_sum" in data.columns:
            data["clearance_rate"] = data["ViolentClr_sum"] / data["Violent_sum"]
            # print("Computed .... clearance_rate")
        else:
            raise ValueError(
                "Columns 'ViolentClr_sum' and 'Violent_sum' are required for clearance rate."
            )
        return data

    # Feature Function: Compute Population Density
    @staticmethod
    def compute_population_density(df):
        """
        Computes the population density as Population / Area_sq_mi.
        """
        data = df.copy()
        if "Population" in data.columns and "Area_sq_mi" in data.columns:
            data["population_density"] = data["Population"] / data["Area_sq_mi"]
        else:
            raise ValueError(
                "Columns 'Population' and 'Area_sq_mi' are required for population density."
            )
        return data

    # Dynamically Add Adjusted Expenditure Columns

    @staticmethod
    def add_adjusted_expenditures(df):
        """
        Adjusts expenditure columns by CPI_Population.
        Parameters:
            data (pd.DataFrame): The input dataset.
        Returns:
            pd.DataFrame: The dataset with adjusted expenditure columns.
        """
        data = df.copy()
        if "CPI_Population" not in data.columns or (data["CPI_Population"] <= 0).any():
            raise ValueError(
                "CPI_Population column is missing or invalid (e.g., contains zero or NaN values)."
            )
        if all(
            col in data.columns
            for col in [
                "police_budget",
                "education_budget",
                "welfare_budget",
                "mental_health_budget",
                "rehab_budget",
                "health_budget",
                "judiciary_budget",
                "prison_budget",
            ]
        ):
            data["adj_police_budget"] = data["police_budget"] / data["CPI_Population"]
            data["adj_education_budget"] = (
                data["education_budget"] / data["CPI_Population"]
            )
            data["adj_welfare_budget"] = data["welfare_budget"] / data["CPI_Population"]
            data["adj_mental_health_budget"] = (
                data["mental_health_budget"] / data["CPI_Population"]
            )
            data["adj_rehab_budget"] = data["rehab_budget"] / data["CPI_Population"]
            data["adj_health_budget"] = data["health_budget"] / data["CPI_Population"]
            data["adj_judiciary_budget"] = (
                data["judiciary_budget"] / data["CPI_Population"]
            )
            data["adj_prison_budget"] = data["prison_budget"] / data["CPI_Population"]

            data["social_vs_security"] = (
                data["adj_education_budget"]
                + data["adj_welfare_budget"]
                + data["adj_health_budget"]
            ) / (
                data["adj_police_budget"]
                + data["adj_judiciary_budget"]
                + data["adj_prison_budget"]
            )
            data["security_vs_social"] = (
                data["adj_police_budget"]
                + data["adj_judiciary_budget"]
                + data["adj_prison_budget"]
            ) / (
                data["adj_education_budget"]
                + data["adj_welfare_budget"]
                + data["adj_health_budget"]
            )
        else:
            raise ValueError("Required columns for adjusted_expenditures are missing.")

        return data

    # Feature Function: Adjusted Income
    @staticmethod
    def compute_adjusted_income(data):
        """
        Computes adjusted income as median household income divided by CPI.
        """
        if "median_household_income" in data.columns and "CPI" in data.columns:
            data["adjusted_income"] = data["median_household_income"] / data["CPI"]
        else:
            raise ValueError(
                "Columns 'median_household_income' and 'CPI' are required for adjusted income."
            )
        return data

    # Feature Function: House Affordability
    @staticmethod
    def compute_house_affordability(data):
        """
        Computes house affordability as median house value divided by median household income.
        """
        if (
            "median_house_value" in data.columns
            and "median_household_income" in data.columns
        ):
            data["house_affordability"] = (
                data["median_house_value"] / data["median_household_income"]
            )
        else:
            raise ValueError(
                "Columns 'median_house_value' and 'median_household_income' are required for house affordability."
            )
        return data

    # Feature Function: Total Persons and Household Metrics
    @staticmethod
    def compute_persons_and_household_metrics(df):
        """
        Computes total persons, total persons for owners/renters, and persons per household.
        """
        data = df.copy()
        if all(
            col in data.columns
            for col in [
                "Vacant_Housing_Units",
                "Total_Housing_Units",
                "Owner_Occupied",
                "Avg_Hsehld_Size_Owner_Occupied",
                "Renter_Occupied",
                "Avg_HseHld_Size_Renter_Occupied",
                "Occupied_Housing_Units",
                "Mobile_Home",
            ]
        ):
            data["home_ownership_rate"] = (
                data["Owner_Occupied"] / data["Occupied_Housing_Units"]
            )
            data["vacancy_rate"] = (
                data["Vacant_Housing_Units"] / data["Total_Housing_Units"]
            )
            data["Total_Persons_Owner"] = (
                data["Owner_Occupied"] * data["Avg_Hsehld_Size_Owner_Occupied"]
            )
            data["Total_Persons_Renter"] = (
                data["Renter_Occupied"] * data["Avg_HseHld_Size_Renter_Occupied"]
            )
            data["Total_Persons"] = (
                data["Total_Persons_Owner"] + data["Total_Persons_Renter"]
            )
            data["Number_of_Persons_per_HseHld"] = (
                data["Total_Persons"] / data["Occupied_Housing_Units"]
            )
            data["renter_ratio"] = data["Total_Persons_Renter"] / data["Total_Persons"]
            data["mobile_home_ratio"] = (
                data["Mobile_Home"] / data["Total_Housing_Units"]
            )
        else:
            raise ValueError(
                "Required columns for total persons or household metrics are missing."
            )
        return data
