import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')  # Built-in matplotlib style
plt.rcParams['figure.figsize'] = (15, 10)

# Configuration
MODEL_PATH = "models/stocklens_lstm_gru_best.h5"
SCALER_PATH = "models/scaler.pkl"
FEATURE_LIST_PATH = "models/feature_list.json"
X_PATH = "data/interim/X.npy"
Y_PATH = "data/interim/y.npy"

def evaluate_model():
    """
    Evaluate the trained model on test data and generate comprehensive metrics
    """
    print("="*60)
    print("STOCKLENS AI - MODEL EVALUATION")
    print("="*60)
    
    # Load model and data
    print("\n[1/6] Loading model and data...")
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    
    with open(FEATURE_LIST_PATH, 'r') as f:
        feature_list = json.load(f)
    
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    print(f"   ✓ Model loaded from {MODEL_PATH}")
    print(f"   ✓ Data shape: X={X.shape}, y={y.shape}")
    print(f"   ✓ Features: {feature_list}")
    
    # Split data (use last 20% as test set - temporal split)
    print("\n[2/6] Splitting data...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   ✓ Train samples: {len(X_train)}")
    print(f"   ✓ Test samples: {len(X_test)}")
    
    # Make predictions
    print("\n[3/6] Generating predictions...")
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    print("   ✓ Predictions complete")
    
    # Denormalize predictions and actuals
    print("\n[4/6] Denormalizing values...")
    close_idx = feature_list.index("Close")
    close_min = scaler.data_min_[close_idx]
    close_max = scaler.data_max_[close_idx]
    
    def denorm(arr):
        return arr * (close_max - close_min) + close_min
    
    y_train_real = denorm(y_train)
    y_test_real = denorm(y_test)
    y_pred_train_real = denorm(y_pred_train)
    y_pred_test_real = denorm(y_pred_test)
    
    print("   ✓ Values converted to actual prices (USD)")
    
    # Calculate metrics
    print("\n[5/6] Calculating performance metrics...")
    print("\n" + "-"*60)
    print("TRAINING SET PERFORMANCE")
    print("-"*60)
    
    train_mse = mean_squared_error(y_train_real, y_pred_train_real)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train_real, y_pred_train_real)
    train_r2 = r2_score(y_train_real, y_pred_train_real)
    train_mape = np.mean(np.abs((y_train_real - y_pred_train_real) / y_train_real)) * 100
    
    print(f"  MSE  (Mean Squared Error)     : ${train_mse:.2f}")
    print(f"  RMSE (Root Mean Squared Error): ${train_rmse:.2f}")
    print(f"  MAE  (Mean Absolute Error)    : ${train_mae:.2f}")
    print(f"  R²   (R-squared Score)        : {train_r2:.4f}")
    print(f"  MAPE (Mean Absolute % Error)  : {train_mape:.2f}%")
    
    print("\n" + "-"*60)
    print("TEST SET PERFORMANCE")
    print("-"*60)
    
    test_mse = mean_squared_error(y_test_real, y_pred_test_real)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_real, y_pred_test_real)
    test_r2 = r2_score(y_test_real, y_pred_test_real)
    test_mape = np.mean(np.abs((y_test_real - y_pred_test_real) / y_test_real)) * 100
    
    print(f"  MSE  (Mean Squared Error)     : ${test_mse:.2f}")
    print(f"  RMSE (Root Mean Squared Error): ${test_rmse:.2f}")
    print(f"  MAE  (Mean Absolute Error)    : ${test_mae:.2f}")
    print(f"  R²   (R-squared Score)        : {test_r2:.4f}")
    print(f"  MAPE (Mean Absolute % Error)  : {test_mape:.2f}%")
    
    # Directional accuracy (did we predict up/down correctly?)
    print("\n" + "-"*60)
    print("DIRECTIONAL ACCURACY (Next-Day Trend)")
    print("-"*60)
    
    # For test set, calculate if we predicted the direction correctly
    test_actual_direction = np.diff(y_test_real)  # Positive = up, Negative = down
    test_pred_direction = y_pred_test_real[1:] - y_test_real[:-1]
    
    directional_accuracy = np.mean((test_actual_direction > 0) == (test_pred_direction > 0)) * 100
    print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
    print(f"  (How often we correctly predicted UP vs DOWN)")
    
    # Generate visualizations
    print("\n[6/6] Generating visualizations...")
    os.makedirs("reports/figures", exist_ok=True)
    
    # Plot 1: Actual vs Predicted (Test Set)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: Test predictions over time
    axes[0, 0].plot(y_test_real, label='Actual', alpha=0.8, linewidth=2, color='#2E86AB')
    axes[0, 0].plot(y_pred_test_real, label='Predicted', alpha=0.8, linewidth=2, color='#A23B72')
    axes[0, 0].set_title('Test Set: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index', fontsize=11)
    axes[0, 0].set_ylabel('Price (USD)', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # Subplot 2: Scatter plot (prediction accuracy)
    axes[0, 1].scatter(y_test_real, y_pred_test_real, alpha=0.5, s=20, color='#F18F01')
    axes[0, 1].plot([y_test_real.min(), y_test_real.max()], 
                     [y_test_real.min(), y_test_real.max()], 
                     'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    axes[0, 1].set_title(f'Prediction Accuracy (R² = {test_r2:.4f})', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Actual Price (USD)', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Price (USD)', fontsize=11)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # Subplot 3: Prediction error distribution
    errors = y_test_real - y_pred_test_real
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='#06A77D')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 0].set_title(f'Prediction Error Distribution (MAE = ${test_mae:.2f})', 
                          fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Error (Actual - Predicted) USD', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    
    # Subplot 4: Error over time
    axes[1, 1].plot(np.abs(errors), alpha=0.7, linewidth=1.5, color='#D62246')
    axes[1, 1].axhline(test_mae, color='blue', linestyle='--', linewidth=2, 
                       label=f'Mean MAE = ${test_mae:.2f}')
    axes[1, 1].set_title('Absolute Error Over Time', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Sample Index', fontsize=11)
    axes[1, 1].set_ylabel('Absolute Error (USD)', fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/model_evaluation.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/model_evaluation.png")
    
    # Additional plot: Training vs Test comparison
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    metrics_comparison = {
        'RMSE': [train_rmse, test_rmse],
        'MAE': [train_mae, test_mae],
        'MAPE (%)': [train_mape, test_mape]
    }
    
    x_pos = np.arange(len(metrics_comparison))
    width = 0.35
    
    train_vals = [metrics_comparison[k][0] for k in metrics_comparison.keys()]
    test_vals = [metrics_comparison[k][1] for k in metrics_comparison.keys()]
    
    ax.bar(x_pos - width/2, train_vals, width, label='Training', color='#2E86AB', alpha=0.8)
    ax.bar(x_pos + width/2, test_vals, width, label='Test', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Test Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_comparison.keys())
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('reports/figures/train_vs_test_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/train_vs_test_comparison.png")
    
    # Save metrics to file
    metrics = {
        "train": {
            "mse": float(train_mse),
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "r2": float(train_r2),
            "mape": float(train_mape)
        },
        "test": {
            "mse": float(test_mse),
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2),
            "mape": float(test_mape),
            "directional_accuracy": float(directional_accuracy)
        },
        "data_info": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features": feature_list,
            "lookback": int(X.shape[1])
        }
    }
    
    with open("reports/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("   ✓ Saved: reports/evaluation_metrics.json")
    
    # Create a text report with UTF-8 encoding
    with open("reports/evaluation_report.txt", "w", encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("STOCKLENS AI - MODEL EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("DATA INFORMATION:\n")
        f.write(f"  Training Samples  : {len(X_train):,}\n")
        f.write(f"  Test Samples      : {len(X_test):,}\n")
        f.write(f"  Features Used     : {len(feature_list)}\n")
        f.write(f"  Lookback Period   : {X.shape[1]} days\n\n")
        
        f.write("-"*60 + "\n")
        f.write("TRAINING SET PERFORMANCE:\n")
        f.write("-"*60 + "\n")
        f.write(f"  MSE  (Mean Squared Error)     : ${train_mse:.2f}\n")
        f.write(f"  RMSE (Root Mean Squared Error): ${train_rmse:.2f}\n")
        f.write(f"  MAE  (Mean Absolute Error)    : ${train_mae:.2f}\n")
        f.write(f"  R²   (R-squared Score)        : {train_r2:.4f}\n")
        f.write(f"  MAPE (Mean Absolute % Error)  : {train_mape:.2f}%\n\n")
        
        f.write("-"*60 + "\n")
        f.write("TEST SET PERFORMANCE:\n")
        f.write("-"*60 + "\n")
        f.write(f"  MSE  (Mean Squared Error)     : ${test_mse:.2f}\n")
        f.write(f"  RMSE (Root Mean Squared Error): ${test_rmse:.2f}\n")
        f.write(f"  MAE  (Mean Absolute Error)    : ${test_mae:.2f}\n")
        f.write(f"  R²   (R-squared Score)        : {test_r2:.4f}\n")
        f.write(f"  MAPE (Mean Absolute % Error)  : {test_mape:.2f}%\n\n")
        
        f.write("-"*60 + "\n")
        f.write("DIRECTIONAL ACCURACY:\n")
        f.write("-"*60 + "\n")
        f.write(f"  Directional Accuracy: {directional_accuracy:.2f}%\n")
        f.write(f"  (How often we correctly predicted UP vs DOWN)\n\n")
        
        f.write("="*60 + "\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("="*60 + "\n")
        f.write("  RMSE:\n")
        f.write("    • < $5   : Excellent\n")
        f.write("    • < $10  : Good\n")
        f.write("    • < $20  : Acceptable\n\n")
        f.write("  R² Score:\n")
        f.write("    • > 0.90 : Strong predictive power\n")
        f.write("    • > 0.70 : Moderate predictive power\n")
        f.write("    • < 0.50 : Weak predictive power\n\n")
        f.write("  MAPE:\n")
        f.write("    • < 5%   : High accuracy\n")
        f.write("    • < 10%  : Good accuracy\n")
        f.write("    • > 15%  : Poor accuracy\n\n")
        f.write("  Directional Accuracy:\n")
        f.write("    • > 55%  : Better than random\n")
        f.write("    • > 60%  : Good trend prediction\n")
        f.write("    • > 70%  : Excellent trend prediction\n")
    
    print("   ✓ Saved: reports/evaluation_report.txt")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  1. reports/figures/model_evaluation.png")
    print("  2. reports/figures/train_vs_test_comparison.png")
    print("  3. reports/evaluation_metrics.json")
    print("  4. reports/evaluation_report.txt")
    
    print("\nModel Performance Summary:")
    print(f"  Test RMSE: ${test_rmse:.2f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
    
    print("\nInterpretation:")
    if test_rmse < 20:
        print("  ✓ RMSE is acceptable for stock prediction")
    else:
        print("  ⚠ RMSE is high - consider retraining with more data")
    
    if directional_accuracy > 70:
        print("  ✓ Excellent trend prediction!")
    elif directional_accuracy > 55:
        print("  ✓ Good directional accuracy")
    else:
        print("  ⚠ Poor trend prediction")
    
    return metrics


if __name__ == "__main__":
    evaluate_model()