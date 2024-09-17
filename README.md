# Calories Burnt Prediction Project

## Overview

This project aims to predict the number of calories burned during exercise based on various user metrics such as age, weight, exercise duration, and heart rate. We’ll start by building a model using the **XGBRegressor** from the XGBoost library. Over time, we'll experiment with other machine learning models to find the most accurate one.

## Datasets

We’re working with two datasets:

1. **exercise.csv**: Contains details about each user's physical attributes and exercise data:
   - `User_ID`: A unique identifier for each user.
   - `Gender`: Male or Female.
   - `Age`: Age of the user (years).
   - `Height`: Height in centimeters.
   - `Weight`: Weight in kilograms.
   - `Duration`: Duration of the exercise in minutes.
   - `Heart_Rate`: Average heart rate during the exercise.
   - `Body_Temp`: Body temperature during the workout (in Celsius).

2. **calories.csv**: Contains the actual number of calories burned by each user:
   - `User_ID`: Corresponding identifier for the user.
   - `Calories`: The number of calories burned.

### Data Preparation

The two datasets are merged using the `User_ID` column, combining both the exercise data and calorie information into one unified dataset. This will make it easier to train machine learning models.

## Workflow

### 1. Data Loading and Cleaning
   - Load both CSV files into pandas DataFrames.
   - Merge them on the `User_ID` column.
   - Handle missing data and clean the dataset for modeling.

### 2. Exploratory Data Analysis (EDA)
   - Visualize relationships between the features (like heart rate, exercise duration) and the target (calories burned).
   - Analyze data distributions and check for any correlations.

### 3. Feature Engineering
   - Convert categorical data like `Gender` into numerical values (e.g., Male = 0, Female = 1).
   - Create or modify features if necessary to improve model performance.

### 4. Initial Modeling: XGBoost
   - Split the dataset into training and testing sets.
   - Train an initial model using **XGBRegressor**.
   - Evaluate the model using metrics like:
     - R-squared (R²)
     - Mean Absolute Error (MAE)
     - Root Mean Squared Error (RMSE)
   
### 5. Model Evaluation
   - Analyze the performance of the XGBRegressor model on the test set.
   - If the model performs well, move to the next phase. Otherwise, try tweaking features or parameters.

### 6. Testing Other Models
   Once we have a baseline with XGBoost, we’ll test other regression models to see if they can improve accuracy:
   - Random Forest Regressor
   - Linear Regression
   - Gradient Boosting
   - Support Vector Regressor (SVR)
   - Neural Networks (if needed)

### 7. Model Tuning
   - Perform hyperparameter tuning using tools like `GridSearchCV` or `RandomizedSearchCV`.
   - Aim for the best possible accuracy on unseen data.

### 8. Final Model Selection
   - Compare all the models based on their performance.
   - Select the model that provides the best balance of accuracy and generalization.

## Installation

### Requirements

You’ll need the following Python libraries to run the project:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Running the Project

1. Clone the repository to your local machine:

   ```bash
   git clone <repository-url>
   ```

2. Go to the project directory:

   ```bash
   cd calories-burnt-prediction
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Open and run the Jupyter notebook to train the model:

   ```bash
   jupyter notebook Calories_Prediction.ipynb
   ```

## Project Structure

```
|-- data/
|   |-- exercise.csv
|   |-- calories.csv
|
|-- code/
|   |-- Calories_Prediction.ipynb
|
|-- models/
|   |-- xgb_model.pkl
|
|-- README.md
|-- requirements.txt
```

## Results

Initially, we’ll report the performance of the XGBRegressor model. As we test other models and optimize them, we’ll update this section with the final results and the selected best-performing model.

## Future Enhancements

Some possible improvements and next steps:
- **Hyperparameter Tuning**: Fine-tune models to improve performance.
- **Feature Importance**: Analyze which features have the biggest impact on calorie prediction.
- **Deployment**: Consider deploying the model using web frameworks like Flask or Streamlit (optional).

## Contributing

We welcome contributions! If you'd like to contribute, here's how you can do it:

1. Fork the project repository.
2. Create a new branch for your changes:
   ```bash
   git checkout -b feature-branch-name
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push the branch:
   ```bash
   git push origin feature-branch-name
   ```
5. Open a pull request for review.

## References
The datasets are publicly available and can be downloaded from Kaggle: [ Calories Burnt Dataset by Fernando Fernandez.](kaggle.com/datasets/fmendes/fmendesdat263xdemos/data)

