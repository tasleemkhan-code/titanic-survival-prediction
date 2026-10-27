# Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster, using **Logistic Regression**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tasleemkhan-code/titanic-survival-prediction/blob/main/titanic_logistic_regression.ipynb)

## Project Overview

The goal is to build a binary classifier that predicts the `survived` column (0 = did not survive, 1 = survived) from passenger details such as class, sex, age, fare and family size.

## Dataset

- Source: the Titanic dataset from Seaborn (`sns.load_dataset("titanic")`)
- Original size: 891 passengers, 15 columns
- After cleaning: 889 records, 8 input features + 1 target

| Feature | Description |
|---|---|
| `pclass` | Ticket class (1, 2, 3) |
| `sex` | Gender of passenger |
| `age` | Age in years |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Ticket fare |
| `embarked` | Port of embarkation |
| `alone` | Whether the passenger travelled alone |
| `survived` | **Target** (0 = No, 1 = Yes) |

## Workflow

1. **Load data** from Seaborn
2. **Drop redundant columns:** `alive`, `embark_town`, `adult_male`, `who`, `class`, `deck` (duplicates of other columns or mostly missing values)
3. **Handle missing values:** `age` filled with the median
4. **Encode categorical features** (`sex`, `embarked`, `alone`) using `LabelEncoder`
5. **Train/test split:** 80% train, 20% test (`random_state=42`)
6. **Train** a Logistic Regression model
7. **Evaluate** using accuracy, confusion matrix and classification report

## Results✅

**Accuracy: 80.34%** on the test set (178 passengers)

**Confusion Matrix**

| | Predicted: Not Survived | Predicted: Survived |
|---|---|---|
| **Actual: Not Survived** | 90 | 19 |
| **Actual: Survived** | 16 | 53 |

**Classification Report**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (Not Survived) | 0.85 | 0.83 | 0.84 | 109 |
| 1 (Survived) | 0.74 | 0.77 | 0.75 | 69 |

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## How to Run

**Option 1: Google Colab (easiest)**
Click the "Open in Colab" badge above and run all cells.

**Option 2: Locally**
```bash
git clone https://github.com/tasleemkhan-code/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
jupyter notebook titanic_logistic_regression.ipynb
```

## Author

**Tasleem Khan**
BCA (Hons.) Student, Cloud & AI/ML
GitHub: [@tasleemkhan-code](https://github.com/tasleemkhan-code)
