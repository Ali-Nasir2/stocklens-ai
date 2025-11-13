import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

# Configuration
MODEL_PATH = "models/stocklens_hybrid_best.keras"
SCALER_PATH = "models/scaler.pkl"
FEATURE_LIST_PATH = "models/feature_list.json"
FEATURE_SPLIT_PATH = "models/feature_split.json"
X_PATH = "data/interim/X.npy"
Y_PATH = "data/interim/y.npy"

def evaluate_model():
    """
    Evaluate the trained model on test data and generate comprehensive metrics
    """
    print("="*60)
    print("STOCKLENS AI - HYBRID MODEL EVALUATION (v2)")
    print("="*60)
    
    # Load model and data
    print("\n[1/6] Loading model and data...")
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    
    with open(FEATURE_LIST_PATH, 'r') as f:
        feature_list = json.load(f)
    
    with open(FEATURE_SPLIT_PATH, 'r') as f:
        feature_split = json.load(f)
    
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    print(f"   ✓ Model loaded from {MODEL_PATH}")
    print(f"   ✓ Data shape: X={X.shape}, y={y.shape}")
    print(f"   ✓ Features: {feature_list}")
    
    # Split features into price and indicator streams
    price_indices = feature_split['price_indices']
    indicator_indices = feature_split['indicator_indices']
    
    X_price = X[:, :, price_indices]
    X_indicators = X[:, :, indicator_indices]
    
    print(f"   ✓ Price stream: {X_price.shape}")
    print(f"   ✓ Indicator stream: {X_indicators.shape}")
    
    # Split data (use last 20% as test set - temporal split)
    print("\n[2/6] Splitting data (chronological)...")
    split_idx = int(len(X) * 0.8)
    
    X_price_train, X_price_test = X_price[:split_idx], X_price[split_idx:]
    X_ind_train, X_ind_test = X_indicators[:split_idx], X_indicators[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   ✓ Train samples: {len(y_train)} (first 80%)")
    print(f"   ✓ Test samples: {len(y_test)} (last 20% - future data)")
    
    # Make predictions
    print("\n[3/6] Generating predictions...")
    y_pred_train = model.predict([X_price_train, X_ind_train], verbose=0).flatten()
    y_pred_test = model.predict([X_price_test, X_ind_test], verbose=0).flatten()
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
    print("TEST SET PERFORMANCE (Chronological Future Data)")
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
    
    # Calculate improvement ratio
    print("\n" + "-"*60)
    print("GENERALIZATION ANALYSIS")
    print("-"*60)
    
    test_train_rmse_ratio = test_rmse / train_rmse
    test_train_r2_diff = train_r2 - test_r2
    
    print(f"  Test/Train RMSE Ratio: {test_train_rmse_ratio:.2f}x")
    print(f"  R² Drop (Train→Test):  {test_train_r2_diff:.4f}")
    
    if test_train_rmse_ratio < 1.5:
        print("  ✓ Excellent generalization!")
    elif test_train_rmse_ratio < 2.5:
        print("  ✓ Good generalization")
    else:
        print("  ⚠ Moderate overfitting detected")
    
    # Directional accuracy
    print("\n" + "-"*60)
    print("DIRECTIONAL ACCURACY (Next-Day Trend)")
    print("-"*60)
    
    test_actual_direction = np.diff(y_test_real)
    test_pred_direction = y_pred_test_real[1:] - y_test_real[:-1]
    
    directional_accuracy = np.mean((test_actual_direction > 0) == (test_pred_direction > 0)) * 100
    
    # Calculate separate up/down accuracies
    up_mask = test_actual_direction > 0
    down_mask = test_actual_direction <= 0
    
    up_accuracy = np.mean((test_actual_direction[up_mask] > 0) == (test_pred_direction[up_mask] > 0)) * 100 if np.any(up_mask) else 0
    down_accuracy = np.mean((test_actual_direction[down_mask] <= 0) == (test_pred_direction[down_mask] <= 0)) * 100 if np.any(down_mask) else 0
    
    print(f"  Overall Directional Accuracy: {directional_accuracy:.2f}%")
    print(f"  UP days accuracy:   {up_accuracy:.2f}%")
    print(f"  DOWN days accuracy: {down_accuracy:.2f}%")
    print(f"  (How often we correctly predicted UP vs DOWN)")
    
    # Generate visualizations
    print("\n[6/6] Generating visualizations...")
    os.makedirs("reports/figures", exist_ok=True)
    
    # Plot 1: Actual vs Predicted (Test Set)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Subplot 1: Test predictions over time
    axes[0, 0].plot(y_test_real, label='Actual', alpha=0.8, linewidth=2, color='#2E86AB')
    axes[0, 0].plot(y_pred_test_real, label='Predicted', alpha=0.8, linewidth=2, color='#A23B72')
    axes[0, 0].set_title(f'Test Set: Actual vs Predicted (R²={test_r2:.3f})', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index', fontsize=11)
    axes[0, 0].set_ylabel('Price (USD)', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # Subplot 2: Scatter plot
    axes[0, 1].scatter(y_test_real, y_pred_test_real, alpha=0.5, s=20, color='#F18F01')
    axes[0, 1].plot([y_test_real.min(), y_test_real.max()], 
                     [y_test_real.min(), y_test_real.max()], 
                     'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    axes[0, 1].set_title(f'Prediction Accuracy (RMSE=${test_rmse:.2f})', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Actual Price (USD)', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Price (USD)', fontsize=11)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # Subplot 3: Prediction error distribution
    errors = y_test_real - y_pred_test_real
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='#06A77D')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 0].axvline(errors.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean Error: ${errors.mean():.2f}')
    axes[1, 0].set_title(f'Error Distribution (MAE=${test_mae:.2f})', 
                          fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Error (Actual - Predicted) USD', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    
    # Subplot 4: Error over time
    axes[1, 1].plot(np.abs(errors), alpha=0.7, linewidth=1.5, color='#D62246')
    axes[1, 1].axhline(test_mae, color='blue', linestyle='--', linewidth=2, 
                       label=f'Mean MAE = ${test_mae:.2f}')
    axes[1, 1].set_title(f'Absolute Error Over Time (Dir. Acc={directional_accuracy:.1f}%)', 
                          fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Sample Index', fontsize=11)
    axes[1, 1].set_ylabel('Absolute Error (USD)', fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/hybrid_model_evaluation_v2.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/hybrid_model_evaluation_v2.png")
    
    # Additional plot: Training vs Test comparison
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    metrics_comparison = {
        'RMSE ($)': [train_rmse, test_rmse],
        'MAE ($)': [train_mae, test_mae],
        'R² Score': [train_r2, test_r2]
    }
    
    x_pos = np.arange(len(metrics_comparison))
    width = 0.35
    
    train_vals = [metrics_comparison[k][0] for k in metrics_comparison.keys()]
    test_vals = [metrics_comparison[k][1] for k in metrics_comparison.keys()]
    
    bars1 = ax.bar(x_pos - width/2, train_vals, width, label='Training', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, test_vals, width, label='Test', color='#A23B72', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Hybrid Model: Training vs Test (Ratio={test_train_rmse_ratio:.2f}x)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_comparison.keys())
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('reports/figures/hybrid_train_vs_test_v2.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/hybrid_train_vs_test_v2.png")
    
    # Save metrics to file
    metrics = {
        "model_type": "Hybrid Multi-Branch (LSTM + GRU) v2",
        "anti_overfitting": {
            "dropout": 0.5,
            "l2_regularization": 0.01,
            "learning_rate": 0.0001,
            "batch_size": 64
        },
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
            "directional_accuracy": float(directional_accuracy),
            "up_accuracy": float(up_accuracy),
            "down_accuracy": float(down_accuracy)
        },
        "generalization": {
            "test_train_rmse_ratio": float(test_train_rmse_ratio),
            "r2_drop": float(test_train_r2_diff)
        },
        "data_info": {
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "features": feature_list,
            "price_features": feature_split['price_features'],
            "indicator_features": feature_split['indicator_features'],
            "lookback": int(X.shape[1])
        }
    }
    
    with open("reports/hybrid_evaluation_metrics_v2.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("   ✓ Saved: reports/hybrid_evaluation_metrics_v2.json")
    
    # Create a comprehensive text report
    with open("reports/hybrid_evaluation_report_v2.txt", "w", encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("STOCKLENS AI - HYBRID MODEL EVALUATION REPORT v2\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write("  Type: Multi-Branch Hybrid (LSTM + GRU)\n")
        f.write(f"  Price Features: {', '.join(feature_split['price_features'])}\n")
        f.write(f"  Indicator Features: {', '.join(feature_split['indicator_features'])}\n\n")
        
        f.write("ANTI-OVERFITTING MEASURES:\n")
        f.write("  • Dropout Rate:        0.5 (50%)\n")
        f.write("  • L2 Regularization:   0.01\n")
        f.write("  • Learning Rate:       0.0001\n")
        f.write("  • Batch Size:          64\n")
        f.write("  • Validation Split:    25% (chronological)\n\n")
        
        f.write("DATA INFORMATION:\n")
        f.write(f"  Training Samples  : {len(y_train):,} (first 80% chronologically)\n")
        f.write(f"  Test Samples      : {len(y_test):,} (last 20% - future data)\n")
        f.write(f"  Total Features    : {len(feature_list)}\n")
        f.write(f"  Lookback Period   : {X.shape[1]} days\n\n")
        
        f.write("-"*70 + "\n")
        f.write("TRAINING SET PERFORMANCE:\n")
        f.write("-"*70 + "\n")
        f.write(f"  MSE  (Mean Squared Error)     : ${train_mse:.2f}\n")
        f.write(f"  RMSE (Root Mean Squared Error): ${train_rmse:.2f}\n")
        f.write(f"  MAE  (Mean Absolute Error)    : ${train_mae:.2f}\n")
        f.write(f"  R²   (R-squared Score)        : {train_r2:.4f}\n")
        f.write(f"  MAPE (Mean Absolute % Error)  : {train_mape:.2f}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write("TEST SET PERFORMANCE (Chronological Future Data):\n")
        f.write("-"*70 + "\n")
        f.write(f"  MSE  (Mean Squared Error)     : ${test_mse:.2f}\n")
        f.write(f"  RMSE (Root Mean Squared Error): ${test_rmse:.2f}\n")
        f.write(f"  MAE  (Mean Absolute Error)    : ${test_mae:.2f}\n")
        f.write(f"  R²   (R-squared Score)        : {test_r2:.4f}\n")
        f.write(f"  MAPE (Mean Absolute % Error)  : {test_mape:.2f}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write("GENERALIZATION ANALYSIS:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Test/Train RMSE Ratio: {test_train_rmse_ratio:.2f}x\n")
        f.write(f"  R² Drop (Train→Test):  {test_train_r2_diff:.4f}\n\n")
        
        if test_train_rmse_ratio < 1.5:
            f.write("  ✓ Excellent generalization - model performs consistently!\n\n")
        elif test_train_rmse_ratio < 2.5:
            f.write("  ✓ Good generalization - acceptable performance gap\n\n")
        else:
            f.write("  ⚠ Moderate overfitting - consider more regularization\n\n")
        
        f.write("-"*70 + "\n")
        f.write("DIRECTIONAL ACCURACY:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Overall Accuracy: {directional_accuracy:.2f}%\n")
        f.write(f"  UP days:          {up_accuracy:.2f}%\n")
        f.write(f"  DOWN days:        {down_accuracy:.2f}%\n")
        f.write(f"  (How often we correctly predicted trend direction)\n\n")
        
        f.write("="*70 + "\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("="*70 + "\n")
        f.write("  RMSE (Stock Prediction):\n")
        f.write("    • < $30  : Excellent for multi-stock model\n")
        f.write("    • < $60  : Good - acceptable for trading\n")
        f.write("    • < $100 : Fair - usable with caution\n")
        f.write("    • > $100 : Poor - needs improvement\n\n")
        
        f.write("  R² Score:\n")
        f.write("    • > 0.70 : Strong predictive power\n")
        f.write("    • > 0.50 : Moderate predictive power\n")
        f.write("    • > 0.30 : Weak but useful\n")
        f.write("    • < 0.30 : Poor predictive power\n\n")
        
        f.write("  MAPE:\n")
        f.write("    • < 10%  : Excellent accuracy\n")
        f.write("    • < 20%  : Good accuracy\n")
        f.write("    • < 30%  : Acceptable\n")
        f.write("    • > 30%  : Poor accuracy\n\n")
        
        f.write("  Directional Accuracy:\n")
        f.write("    • > 70%  : Excellent trend prediction (tradeable!)\n")
        f.write("    • > 60%  : Good trend prediction\n")
        f.write("    • > 55%  : Better than random\n")
        f.write("    • < 55%  : No predictive value\n\n")
        
        f.write("  Test/Train RMSE Ratio:\n")
        f.write("    • < 1.5  : Excellent generalization\n")
        f.write("    • < 2.5  : Good generalization\n")
        f.write("    • < 3.5  : Moderate overfitting\n")
        f.write("    • > 3.5  : Severe overfitting\n\n")
        
        f.write("="*70 + "\n")
        f.write("FINAL VERDICT:\n")
        f.write("="*70 + "\n")
        
        # Overall assessment
        if test_r2 > 0.50 and test_rmse < 60 and directional_accuracy > 70:
            f.write("  ✓✓✓ MODEL IS PRODUCTION READY!\n")
            f.write("  This model shows strong predictive power and excellent trend detection.\n")
            f.write("  Suitable for algorithmic trading strategies.\n")
        elif test_r2 > 0.30 and test_rmse < 100 and directional_accuracy > 60:
            f.write("  ✓✓ MODEL IS USABLE WITH CAUTION\n")
            f.write("  This model has moderate predictive power.\n")
            f.write("  Best used as one signal among multiple indicators.\n")
        else:
            f.write("  ⚠ MODEL NEEDS IMPROVEMENT\n")
            f.write("  Consider: More data, feature engineering, or architecture changes.\n")
    
    print("   ✓ Saved: reports/hybrid_evaluation_report_v2.txt")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  1. reports/figures/hybrid_model_evaluation_v2.png")
    print("  2. reports/figures/hybrid_train_vs_test_v2.png")
    print("  3. reports/hybrid_evaluation_metrics_v2.json")
    print("  4. reports/hybrid_evaluation_report_v2.txt")
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"  Test RMSE:              ${test_rmse:.2f}")
    print(f"  Test R²:                {test_r2:.4f}")
    print(f"  Test/Train RMSE Ratio:  {test_train_rmse_ratio:.2f}x")
    print(f"  Directional Accuracy:   {directional_accuracy:.2f}%")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # RMSE assessment
    if test_rmse < 30:
        print("  ✓✓✓ RMSE is EXCELLENT!")
    elif test_rmse < 60:
        print("  ✓✓ RMSE is GOOD - acceptable for trading")
    elif test_rmse < 100:
        print("  ✓ RMSE is FAIR - usable with caution")
    else:
        print("  ⚠ RMSE is HIGH - needs improvement")
    
    # R² assessment
    if test_r2 > 0.70:
        print("  ✓✓✓ Strong predictive power!")
    elif test_r2 > 0.50:
        print("  ✓✓ Moderate predictive power")
    elif test_r2 > 0.30:
        print("  ✓ Weak but useful predictive power")
    else:
        print("  ⚠ Poor predictive power")
    
    # Generalization assessment
    if test_train_rmse_ratio < 1.5:
        print("  ✓✓✓ Excellent generalization!")
    elif test_train_rmse_ratio < 2.5:
        print("  ✓✓ Good generalization")
    else:
        print("  ⚠ Moderate overfitting detected")
    
    # Directional assessment
    if directional_accuracy > 70:
        print("  ✓✓✓ Excellent trend prediction - TRADEABLE!")
    elif directional_accuracy > 60:
        print("  ✓✓ Good trend prediction")
    elif directional_accuracy > 55:
        print("  ✓ Better than random")
    else:
        print("  ⚠ Poor trend prediction")
    
    # Overall verdict
    print("\n" + "="*70)
    if test_r2 > 0.50 and test_rmse < 60 and directional_accuracy > 70:
        print("✓✓✓ MODEL IS PRODUCTION READY!")
        print("="*70)
        print("\nNext Step: Run live predictions!")
        print("  python -m src.load_and_predict")
    elif test_r2 > 0.30 and test_rmse < 100 and directional_accuracy > 60:
        print("✓✓ MODEL IS USABLE (with caution)")
        print("="*70)
        print("\nRecommendation: Use as one signal among multiple indicators")
    else:
        print("⚠ MODEL NEEDS IMPROVEMENT")
        print("="*70)
        print("\nConsider: More training epochs, more data, or feature engineering")
    
    return metrics


if __name__ == "__main__":
    evaluate_model()