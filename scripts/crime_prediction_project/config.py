"""
Configuration file
"""

CONFIG = {
    # General settings
    "random_seed": 42,
    # Default model parameters for each model type
    "default_model_params": {
        "LinearRegression": {},
        "Ridge": {"alpha": 1.0},
        "Lasso": {"alpha": 0.1},
        "RandomForest": {"n_estimators": 100, "max_depth": 5},
        "XGBoost": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    },
    # Cross-validation settings
    "kfolds": 5,
    # Dataset and feature engineering
    "feature_columns": [
        "Population",
        "crime_rate",
        "clearance_rate",
        "population_density",
        "unemployment_rate",
        "adjusted_income",
        "poverty_rate",
        "rent_burden",
        "home_ownership_rate",
        "mobile_home_ratio",
        "vacancy_rate",
        "Total_Persons_Owner",
        "Total_Persons_Renter",
        "Total_Persons",
        "Number_of_Persons_per_HseHld",
        "Median_Age",
        "police_budget",
        "education_budget",
        "welfare_budget",
        "mental_health_budget",
        "rehab_budget",
        "health_budget",
        "judiciary_budget",
        "prison_budget",
        "adj_police_budget",
        "adj_education_budget",
        "adj_welfare_budget",
        "adj_mental_health_budget",
        "adj_rehab_budget",
        "adj_health_budget",
        "adj_judiciary_budget",
        "adj_prison_budget",
        "median_house_value",
        "house_affordability",
        "Category",
        "Category_encoded",
        "Category_Rural",
        "Category_Suburban",
        "Category_Urban",
        "uninsured_rate",
        "high_school_rate",
        "renter_ratio",
        "social_vs_security",
        "security_vs_social",
    ],
    "target_column": "crime_rate",
    "features_to_apply": [
        "compute_clearance_rate",
        "compute_population_density",
        "add_adjusted_expenditures",
        "compute_adjusted_income",
        "compute_house_affordability",
        "compute_persons_and_household_metrics",
    ],
    "non_categorical_features": [
        "Population",
        "clearance_rate",
        "population_density",
        "mobile_home_ratio",
        "poverty_rate",
        "adjusted_income",
        "unemployment_rate",
        "high_school_rate",
        "uninsured_rate",
        "house_affordability",
        "adj_police_budget",
        "adj_education_budget",
        "adj_welfare_budget",
        "adj_mental_health_budget",
        "adj_rehab_budget",
        "adj_health_budget",
        "adj_judiciary_budget",
        "adj_prison_budget",
        "home_ownership_rate",
        "rent_burden",
        "renter_ratio",
        "social_vs_security",
        "security_vs_social",
    ],
    # Grid parameters for hyperparameter tuning
    "param_grids": {
        "Ridge": {
            "model__alpha": [0.01, 0.1, 1, 10, 100],
            "model__random_state": [42],  # Ensure reproducibility
        },
        "Lasso": {
            "model__alpha": [0.01, 0.1, 1, 10, 100],
            "model__max_iter": [1000, 5000, 10000],
            "model__tol": [0.0001, 0.001, 0.01],
            "model__fit_intercept": [True, False],
            "model__selection": ["cyclic", "random"],
        },
        "LinearRegression": {
            "model__fit_intercept": [True, False],
            "model__n_jobs": [-1],  # Use all CPU cores
            "model__positive": [True, False],
            "pca__n_components": [3, 5, 10, 11],
        },
        "RandomForest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__random_state": [42],  # Ensure reproducibility
        },
        "XGBoost": {
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
            "model__n_estimators": [50, 100, 200],
            "model__colsample_bytree": [0.3, 0.5, 0.7, 1.0],
            "model__random_state": [42],  # Ensure reproducibility
        },
    },
}
