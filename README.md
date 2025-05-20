# Food Rating Prediction using Machine Learning

This project explores machine learning techniques to predict food ratings using structured data. The dataset comes from a Kaggle competition and includes various features related to food items.

## 📁 Dataset

The main dataset used is:

- `train.csv`: Contains features and corresponding ratings of food items.

The dataset is expected to be in the directory: `/kaggle/input/recipe-for-rating-predict-food-ratings-using-ml/`.

## 🚀 Getting Started

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/food-rating-ml.git
cd food-rating-ml
```

2. **Open the Notebook**

Use Jupyter Notebook or any compatible environment (e.g., Kaggle Notebooks) to open:

```
21f3001205-notebook-t12024-ipynb (3).ipynb
```

## 📦 Dependencies

Make sure you have the following Python libraries installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 🧠 ML Workflow

The notebook follows these steps:

1. **Data Loading** – Load `train.csv` using `pandas`.
2. **Exploratory Data Analysis (EDA)** – Visualize and summarize key features.
3. **Data Preprocessing** – Handle missing values, encode categorical features, etc.
4. **Model Training** – Train ML models (e.g., Linear Regression, Random Forest).
5. **Model Evaluation** – Evaluate using appropriate metrics like RMSE or R².
6. **Conclusion** – Insights and results summary.

## 📊 Sample Code Snippet

```python
import pandas as pd

train_data = pd.read_csv("/kaggle/input/recipe-for-rating-predict-food-ratings-using-ml/train.csv")
print(train_data.shape)
train_data.head()
```

## 📌 Notes

- This notebook is intended for educational and experimentation purposes.
- If running outside Kaggle, update the dataset path accordingly.

