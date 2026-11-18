# Titanic Survival Prediction

Machine learning project that predicts passenger survival on the Titanic using classification models.

## Dataset
- Source: `Titanic-Dataset.csv`
- Original shape: 891 rows × 12 columns
- Target variable: `Survived` (0 = Died, 1 = Survived)

## Data Cleaning Steps
1. Dropped the `Cabin` column due to a very high number of missing values.
2. Dropped the `Name` column (no useful predictive information for the models).
3. Removed all rows with missing values in `Age` and `Embarked`.
4. Final clean dataset: **712 rows**.

**Note:** No imputation was used. Rows with missing values were simply dropped.

## Models Used
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost (imported and evaluated)

Hyperparameter tuning was performed with **GridSearchCV** + 5-fold cross-validation (`KFold`, `shuffle=True`, `random_state=42`).

### Hyperparameter Grids
- **Logistic Regression**: `C`, `solver`
- **Random Forest**: `n_estimators`, `max_depth`, `max_features`
- **Gradient Boosting**: `n_estimators`, `max_depth`, `learning_rate`

## Evaluation
Tuned models were evaluated using 5-fold cross-validation. Metrics reported:
- Accuracy
- Precision
- Recall
- Confusion Matrix (for the last fold)

## Key Insights
- People seated closer to emergency doors had a higher chance of survival.
- Passengers in **1st and 2nd class** had significantly higher survival rates.
- Passengers in **3rd class** had lower survival rates.
- Random Forest achieved higher **Precision** than XGBoost.
- XGBoost achieved higher **Recall** than Random Forest.
- The current EDA and feature engineering are not sufficient to clearly decide which model is superior overall.

## Notes
- No missing value imputation was applied.
- Feature engineering and exploratory analysis can be improved for better model performance and clearer conclusions.
