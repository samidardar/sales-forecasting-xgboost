# 📊 Sales Forecasting with XGBoost

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Advanced sales forecasting model using XGBoost with Optuna hyperparameter tuning, time-series features, and comprehensive exploratory data analysis. Predicts future sales for e-commerce stores and items with high accuracy.

## 🎯 Project Overview

This project develops a robust sales forecasting model using machine learning to predict future sales for multiple stores and items. The model achieves strong performance through advanced feature engineering, automated hyperparameter optimization, and careful validation on time-series data.

### Key Features

- **Advanced Time-Series Feature Engineering**: Lag features and rolling averages
- **Automated Hyperparameter Tuning**: Optuna-based optimization for XGBoost
- **Comprehensive EDA**: In-depth visualization of sales patterns and seasonality
- **Time-Based Validation**: Proper train/validation split for time-series data
- **Strong Performance**: RMSE of 6.98 and MAE of 5.39 on validation set

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Validation RMSE** | 6.9780 |
| **Validation MAE** | 5.3860 |
| **Training Samples** | 730,500 |
| **Validation Samples** | 182,500 |

### Feature Importance

| Feature | Importance |
|---------|------------|
| `rolling_mean_7_sales` | 0.738 |
| `dayofweek` | 0.136 |
| `lag_1_sales` | 0.111 |
| `month` | 0.007 |
| `day` | 0.004 |

## 🏗️ Project Structure

```
sales-forecasting-xgboost/
├── notebooks/
│   └── sales_forecasting_model.ipynb   # Main Jupyter notebook
├── data/
│   ├── train.csv                       # Training dataset
│   ├── test.csv                        # Test dataset
│   └── sample_submission.csv           # Submission template
├── requirements.txt                     # Python dependencies
├── README.md                           # Project documentation
└── .gitignore                          # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or Google Colab
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/samidardar/sales-forecasting-xgboost.git
   cd sales-forecasting-xgboost
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

#### Option 1: Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Upload the data files (`train.csv`, `test.csv`, `sample_submission.csv`)
3. Run all cells sequentially

#### Option 2: Local Jupyter Notebook
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/sales_forecasting_model.ipynb`
3. Run all cells

## 📊 Methodology

### 1. Data Loading and Exploration
- Load training, test, and submission data
- Inspect data types and basic statistics
- Check for missing values

### 2. Feature Engineering

**Time-Based Features:**
- `year`, `month`, `day`: Extracted from date column
- `dayofweek`: Day of week (0-6)
- `weekofyear`: Week number of the year

**Advanced Time-Series Features:**
- `lag_1_sales`: Previous day's sales (grouped by store and item)
- `rolling_mean_7_sales`: 7-day rolling average of sales

### 3. Exploratory Data Analysis

Comprehensive visualizations reveal:
- **Overall Sales Trend**: General upward trajectory over time
- **Store Performance**: Significant variation across stores
- **Item Performance**: Different items have distinct sales patterns
- **Monthly Seasonality**: Clear seasonal trends throughout the year
- **Weekly Patterns**: Specific days show higher/lower sales

### 4. Data Preparation

- **Time-Based Split**: Training data before 2017-01-01, validation after
- **Feature Selection**: 9 features including engineered time-series features
- **Handling Missing Values**: Fill NaN values in lag/rolling features with 0

### 5. Hyperparameter Optimization

Using Optuna framework to optimize XGBoost hyperparameters:
- `n_estimators`: Number of boosting rounds
- `learning_rate`: Step size shrinkage
- `max_depth`: Maximum tree depth
- `subsample`: Subsample ratio of training instances
- `colsample_bytree`: Subsample ratio of columns
- `gamma`: Minimum loss reduction
- `min_child_weight`: Minimum sum of instance weight

**Optimization Results:**
- 50 trials completed
- Best RMSE: ~7.095

### 6. Model Training

Final XGBoost model trained on combined training + validation data to leverage all historical information for production predictions.

### 7. Evaluation and Visualization

- **Performance Metrics**: RMSE and MAE
- **Actual vs Predicted Plot**: Visual assessment of model fit
- **Feature Importance Plot**: Understanding key predictive features

## 📦 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
optuna>=3.0.0
scikit-learn>=1.0.0
kagglehub>=0.3.0
```

## 🔍 Key Insights

1. **Recent Historical Sales are Critical**: The rolling 7-day average is the most important feature (73.8% importance), indicating that recent trends are strong predictors.

2. **Day of Week Matters**: Weekly seasonality plays a significant role (13.6% importance), suggesting consistent weekly patterns.

3. **Immediate Past Predicts Future**: The lag-1 feature (previous day's sales) contributes meaningfully (11.1% importance).

4. **Store and Item Variability**: Different stores and items show distinct sales patterns, requiring individualized predictions.

## 🎯 Future Improvements

- [ ] Experiment with additional lag periods (7, 14, 30 days)
- [ ] Incorporate external features (holidays, promotions, weather)
- [ ] Test ensemble methods (combining multiple models)
- [ ] Implement deep learning models (LSTM, GRU, Transformer)
- [ ] Add confidence intervals for predictions
- [ ] Deploy model as REST API for real-time predictions

## 📝 Dataset

The project uses a demand forecasting dataset with:
- **913,000 training samples** (2013-2017)
- **10 stores**
- **50 items**
- **Daily sales data**

Dataset source: Kaggle Demand Forecasting Competition

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Sami Dardar**
- GitHub: [@samidardar](https://github.com/samidardar)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/samidardar)

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- XGBoost and Optuna communities for excellent documentation
- Google Colab for free computational resources

---

⭐ If you find this project useful, please consider giving it a star!
