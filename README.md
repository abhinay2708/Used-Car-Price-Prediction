
# Used Car Price Prediction

## Project Overview

This project focuses on predicting used car prices using various machine learning techniques. The goal is to build a robust model that can estimate the price of a used car based on its features such as brand, model year, mileage, fuel type, engine specifications, accident history, and title status.

## Dataset

The project utilizes a dataset named `used_cars.csv`. This dataset contains comprehensive information about used cars, including:

- `brand`: Brand of the car
- `model_year`: Manufacturing year of the car
- `milage`: Total mileage of the car
- `fuel_type`: Type of fuel used by the car
- `engine`: Engine specifications (e.g., HP, liters, V-engine)
- `transmission`: Transmission type
- `ext_col`: Exterior color
- `int_col`: Interior color
- `accident`: Accident history
- `clean_title`: Status of the car title
- `price`: Selling price of the car (target variable)

## Dependencies

The following Python libraries are required to run this notebook:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `re`
- `sklearn` (for `ColumnTransformer`, `OneHotEncoder`, `Pipeline`, `RandomForestRegressor`, `train_test_split`, `mean_absolute_error`)
- `xgboost` (for `XGBRegressor`)

## Data Preprocessing and Feature Engineering

The notebook performs several data preprocessing and feature engineering steps:

- **Mileage Cleaning**: Removes non-numeric characters from the 'milage' column and converts it to integer type.
- **Fuel Type Standardization**: Converts various fuel type entries into standardized categories (e.g., 'Plug-In Hybrid' to 'Hybrid', and handles missing/unsupported values).
- **Engine Feature Extraction**: Extracts 'engine_hp' (horsepower) and 'liters' from the 'engine' column using regular expressions. Fills missing 'engine_hp' values based on median HP for similar 'liters' values, and then with the overall median if any still remain. Missing 'liters' values are filled with the median.
- **V-Engine Indicator**: Creates a binary feature 'v_engine' to indicate if the car has a V-engine.
- **Transmission Type Cleaning**: Standardizes various transmission entries into categories like 'Manual', 'CVT', 'Single-Speed', and 'Automatic'.
- **Accident History**: Converts 'accident' column to a binary (0/1) representation, where 1 indicates an accident reported and 0 for none reported or missing.
- **Clean Title**: Converts 'clean_title' column to a binary (0/1) representation.
- **Price Cleaning**: Removes '$' and ',' from the 'price' column and converts it to integer type.
- **Feature Dropping**: Removes original 'model', 'engine', 'transmission', 'ext_col', 'int_col' columns after extracting relevant features.
- **Log Transformation**: Applies log transformation to the 'price' column (`log_price`) to handle skewness.
- **Car Age Calculation**: Creates a 'car_age' feature (2024 - `model_year`).
- **Mileage Per Year**: Calculates 'milage_per_year' by dividing 'milage' by 'car_age'.
- **Outlier Treatment**: Handles outliers in 'engine_hp', 'price', 'milage', and 'liters' using the IQR method.
- **Categorical Encoding**: One-hot encodes categorical features ('brand', 'fuel_type', 'transmission_type') using `OneHotEncoder` within a `ColumnTransformer`.

## Exploratory Data Analysis (EDA)

The notebook includes various visualizations to understand the data:

- Bar plots for `brand`, `model_year`, `fuel_type`, `accident`, `clean_title`, `car_age`, `v_engine`, `transmission_type`.
- Box plots for `brand`, `model_year`, `milage`, `price`, `log_price`, `engine_hp`, `liters`.
- Distribution plots (`distplot`) for `milage`, `price`, `log_price`, `engine_hp`, `liters`, `milage_per_year`.
- Scatter plots for `brand` vs `model_year`, `brand` vs `fuel_type`, `milage_per_year` vs `price`, `milage_per_year` vs `log_price`.
- Heatmap for numerical feature correlation.

## Model Training

Two regression models are trained and evaluated:

### 1. Random Forest Regressor

- **Pipeline**: A `Pipeline` is used to combine preprocessing steps and the `RandomForestRegressor` model.
- **Hyperparameters**: `n_estimators=300`, `max_depth=15`, `min_samples_split=5`.
- **Evaluation**: Mean Absolute Error (MAE) is calculated on the exponentiated predictions (to revert log transformation).

### 2. XGBoost Regressor

- **Pipeline**: A `Pipeline` is used to combine preprocessing steps and the `XGBRegressor` model.
- **Hyperparameters**: `n_estimators=800`, `learning_rate=0.03`, `max_depth=7`, `min_child_weight=3`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Evaluation**: Mean Absolute Error (MAE) is calculated on the exponentiated predictions.

## Results

The MAE for both models are printed, and a scatter plot comparing actual vs predicted prices (after exponentiation) is generated to visualize model performance.

## Usage

To run this project:

1. Ensure you have all the dependencies installed.
2. Place the `used_cars.csv` dataset in the same directory as the `used_car_price.ipynb` notebook, or update the data loading path.
3. Open and run the `used_car_price.ipynb` notebook in a Jupyter environment.
