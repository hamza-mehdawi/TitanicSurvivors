# 🚢 Predicting Titanic Survivors

A machine learning project that predicts passenger survival on the Titanic using classification models. The analysis covers the full data science pipeline: data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Key Insights](#key-insights)
- [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## 📌 Project Overview

The goal of this project is to build a classification model that predicts which passengers survived the Titanic disaster. This is based on the famous [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic), one of the most popular beginner machine learning competitions.

---

## 📂 Dataset

**File:** `Titanic-Dataset.csv`

The dataset contains **891 passenger records** with the following columns:

| Column | Description |
|---|---|
| `PassengerId` | Unique passenger identifier |
| `Survived` | Target variable — 0 = Did not survive, 1 = Survived |
| `Pclass` | Passenger class (1st, 2nd, or 3rd) |
| `Name` | Passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 🗂️ Project Structure

```
Project Predicting Titanic Survivors/
│
├── main.ipynb           # Main Jupyter notebook with full analysis
├── Titanic-Dataset.csv  # Raw dataset
└── README.md            # Project documentation
```

---

## 🧹 Data Cleaning

The raw dataset had several missing value issues that were addressed as follows:

### Columns Dropped
- **`Cabin`** — Dropped due to a very high number of missing values (only 204 out of 891 non-null). Imputing with mean/median was not meaningful for a categorical column like cabin number.
- **`Name`** — Dropped because passenger names have no meaningful information for predicting survival.

### Missing Values Handled
- **`Age`** and **`Embarked`** — Rows with missing values in these columns were dropped entirely, as filling them with mean or median would introduce noise for these particular features.

### Result After Cleaning
After cleaning, the dataset was reduced from **891** to **712** clean records with **10 columns** and **zero missing values**.

---

## 📊 Exploratory Data Analysis

The EDA covers survival patterns across multiple features:

### Survival Rate
- Approximately **38–40%** of passengers survived, while ~**60%** did not.

### Passenger Class Distribution
- The majority of passengers (~55%) were in **3rd class**.
- ~24% were in **1st class** and ~21% in **2nd class**.

### Survival by Gender
- **Women had a significantly higher survival rate than men.**
- This reflects the historical "women and children first" evacuation protocol used during the disaster.
- Men were largely sacrificed in the accident — survival priority was given to women and children.

### Survival by Age
- Younger passengers (especially children) had a higher chance of survival, consistent with the evacuation priority.

---

## 💡 Key Insights

> **Gender** was one of the strongest predictors of survival. Women were much more likely to survive than men due to the rescue priority protocol ("women and children first").

> **Passenger Class (Pclass)** played a significant role — 1st class passengers had a much higher survival rate than 3rd class passengers, likely due to cabin location (closer to lifeboats) and social factors.

> **Age** mattered — children had higher survival rates, while middle-aged and older male passengers had the lowest survival rates.

> **Family size (SibSp + Parch)** showed nuanced effects — passengers with small families tended to survive more than solo travelers or those with very large families.

> **Fare** was positively correlated with survival — higher fares often indicated higher passenger class and better access to lifeboats.

---

## ⚙️ Feature Engineering & Preprocessing

The following preprocessing steps were applied:

- **Label Encoding** — Categorical columns (`Sex`, `Embarked`) were encoded into numeric values using `LabelEncoder`.
- **Feature Scaling** — Numeric features were standardized using `StandardScaler` or normalized using `MinMaxScaler` where appropriate.
- **Train/Test Split** — The cleaned dataset was split into training and test sets using `train_test_split`.
- **Cross-Validation** — `StratifiedKFold` and `cross_val_score` were used to ensure robust evaluation.

---

## 🤖 Models Used

The following classification models were trained and compared:

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline linear model |
| **Random Forest Classifier** | Ensemble of decision trees |
| **Gradient Boosting Classifier** | Boosted ensemble method |
| **XGBoost** | Extreme Gradient Boosting |

Hyperparameter tuning was performed using **GridSearchCV** to find optimal parameters for the best-performing model.

---

## 📈 Evaluation Metrics

Models were evaluated using:

- **Accuracy Score**
- **Precision, Recall, F1-Score**
- **Confusion Matrix** (with visual display using `ConfusionMatrixDisplay`)
- **ROC-AUC Score** and **ROC Curve**

---

## 📦 Requirements

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

Or using conda:

```bash
conda install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Key Libraries Used

| Library | Version |
|---|---|
| `numpy` | Latest |
| `pandas` | Latest |
| `matplotlib` | Latest |
| `seaborn` | Latest |
| `scikit-learn` | Latest |
| `xgboost` | Latest |

---

## ▶️ How to Run

1. **Clone the repository** (or download the files):
   ```bash
   git clone <your-repo-url>
   cd "Project Predicting Titanic Survivors"
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```

4. **Run all cells** in order from top to bottom.

---

## 🏆 Summary

This project demonstrates a complete machine learning workflow for binary classification:

1. ✅ Data loading and inspection
2. ✅ Data cleaning (handling missing values, dropping irrelevant features)
3. ✅ Exploratory data analysis with visualizations
4. ✅ Feature engineering and preprocessing
5. ✅ Training and comparing multiple ML models
6. ✅ Hyperparameter tuning
7. ✅ Model evaluation with comprehensive metrics

---

*Dataset source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)*
