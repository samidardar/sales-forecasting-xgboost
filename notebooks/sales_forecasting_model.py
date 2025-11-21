"""
Sales Forecasting Model using XGBoost
This notebook implements an end-to-end sales forecasting pipeline
with feature engineering, model training, and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class SalesForecastingModel:
    """
    A comprehensive sales forecasting model using XGBoost
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        
    def load_data(self, filepath):
        """Load sales data from CSV file"""
        self.data = pd.read_csv(filepath)
        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def create_time_features(self, df, date_column='date'):
        """
        Create time-based features from date column
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Extract time features
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        df['weekofyear'] = df[date_column].dt.isocalendar().week
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
        
        return df
    
    def create_lag_features(self, df, target_col='sales', lags=[1, 7, 14, 30]):
        """
        Create lag features for time series
        """
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        return df
    
    def encode_categorical_features(self, df, categorical_cols):
        """Encode categorical variables"""
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def preprocess_data(self, df, date_col='date', target_col='sales', 
                       categorical_cols=None, create_lags=True):
        """
        Complete preprocessing pipeline
        """
        print("Starting preprocessing...")
        
        # Create time features
        df = self.create_time_features(df, date_col)
        
        # Create lag features if specified
        if create_lags:
            df = self.create_lag_features(df, target_col)
        
        # Encode categorical features
        if categorical_cols:
            df = self.encode_categorical_features(df, categorical_cols)
        
        # Drop rows with NaN values created by lag features
        df = df.dropna()
        
        print(f"Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def prepare_train_test(self, df, target_col='sales', test_size=0.2):
        """
        Split data into train and test sets
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [target_col, 'date']]
        X = df[feature_cols]
        y = df[target_col]
        
        # Time series split (important for temporal data)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, **xgb_params):
        """
        Train XGBoost model
        """
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        default_params.update(xgb_params)
        
        print("Training XGBoost model...")
        self.model = xgb.XGBRegressor(**default_params)
        self.model.fit(X_train, y_train, verbose=False)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Training complete!")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        print("\nModel Performance Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, y_pred
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        """
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_test, y_pred):
        """
        Plot actual vs predicted values
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Time series plot
        axes[0].plot(y_test.values, label='Actual', linewidth=2, alpha=0.7)
        axes[0].plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
        axes[0].set_title('Sales Forecast: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Sales', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_test, y_pred, alpha=0.5)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_title('Actual vs Predicted Scatter Plot', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Actual Sales', fontsize=12)
        axes[1].set_ylabel('Predicted Sales', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def cross_validate(self, X, y, n_splits=5):
        """
        Perform time series cross-validation
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(self.model, X, y, cv=tscv, 
                                scoring='neg_mean_squared_error', n_jobs=-1)
        rmse_scores = np.sqrt(-scores)
        
        print(f"\nCross-Validation RMSE scores: {rmse_scores}")
        print(f"Mean RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
        
        return rmse_scores
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
        return self.model.predict(X)


# Example usage
if __name__ == "__main__":
    # Initialize model
    forecaster = SalesForecastingModel(random_state=42)
    
    # Load your data (replace with actual file path)
    # df = forecaster.load_data('sales_data.csv')
    
    # Preprocess data
    # df_processed = forecaster.preprocess_data(
    #     df, 
    #     date_col='date',
    #     target_col='sales',
    #     categorical_cols=['store_id', 'product_category'],
    #     create_lags=True
    # )
    
    # Prepare train/test split
    # X_train, X_test, y_train, y_test = forecaster.prepare_train_test(
    #     df_processed, target_col='sales', test_size=0.2
    # )
    
    # Train model
    # forecaster.train_model(X_train, y_train, n_estimators=200, learning_rate=0.05)
    
    # Evaluate model
    # metrics, y_pred = forecaster.evaluate_model(X_test, y_test)
    
    # Visualize results
    # forecaster.plot_feature_importance(top_n=20)
    # forecaster.plot_predictions(y_test, y_pred)
    
    print("Sales Forecasting Model is ready to use!")
