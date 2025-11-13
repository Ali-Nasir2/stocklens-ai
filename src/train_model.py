import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, Concatenate, 
    Bidirectional, BatchNormalization, Attention, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import json

print("="*60)
print("STOCKLENS AI - HYBRID MODEL TRAINING (21 Stocks)")
print("="*60)

# Load data
print("\n[1/5] Loading data...")
X = np.load("data/interim/X.npy")
y = np.load("data/interim/y.npy")

with open("models/feature_list.json", "r") as f:
    features = json.load(f)

print(f"   ✓ Data loaded: X={X.shape}, y={y.shape}")
print(f"   ✓ Features: {features}")

# Print data statistics to check for bias
print(f"\n[DATA CHECK] Target (y) Statistics:")
print(f"   Mean: {y.mean():.6f}")
print(f"   Std:  {y.std():.6f}")
print(f"   Min:  {y.min():.6f}")
print(f"   Max:  {y.max():.6f}")

# Check for UP/DOWN balance
price_changes = np.diff(y)
up_moves = np.sum(price_changes > 0)
down_moves = np.sum(price_changes < 0)
print(f"\n[BALANCE CHECK] Price movements:")
print(f"   UP moves:    {up_moves:,} ({up_moves/len(price_changes)*100:.1f}%)")
print(f"   DOWN moves:  {down_moves:,} ({down_moves/len(price_changes)*100:.1f}%)")

# ==========================================
# FEATURE SEPARATION
# ==========================================
print("\n[2/5] Separating features into groups...")

# Group 1: Price/Volume (OHLCV) - High impact on prediction
price_features = ['Close', 'High', 'Low', 'Open']

# Group 2: Technical Indicators - Pattern recognition
# 🔧 UPDATED to match new build_features.py output
indicator_features = []
for f in features:
    if f not in price_features:  # Everything else is an indicator
        indicator_features.append(f)

price_indices = [features.index(f) for f in price_features if f in features]
indicator_indices = [features.index(f) for f in indicator_features if f in features]

print(f"   ✓ Price features ({len(price_indices)}): {[features[i] for i in price_indices]}")
print(f"   ✓ Indicators ({len(indicator_indices)}): {[features[i] for i in indicator_indices]}")

# Split data into two streams
X_price = X[:, :, price_indices]        
X_indicators = X[:, :, indicator_indices]

print(f"   ✓ Price stream: {X_price.shape}")
print(f"   ✓ Indicator stream: {X_indicators.shape}")

# ==========================================
# BUILD MULTI-BRANCH MODEL (v4 - 21 Stocks)
# ==========================================
print("\n[3/5] Building hybrid architecture (v4 - 21 stocks)...")

# ============ BRANCH 1: Price Stream (LSTM) ============
price_input = Input(shape=(X_price.shape[1], X_price.shape[2]), name='price_input')

price_lstm1 = Bidirectional(
    LSTM(128, return_sequences=True, kernel_regularizer=l2(0.003)),  # 🔧 Further reduced L2
    name='price_lstm1'
)(price_input)
price_norm1 = LayerNormalization(name='price_norm1')(price_lstm1)
price_drop1 = Dropout(0.25, name='price_dropout1')(price_norm1)  # 🔧 Reduced from 0.3

price_lstm2 = Bidirectional(
    LSTM(64, return_sequences=True, kernel_regularizer=l2(0.003)),
    name='price_lstm2'
)(price_drop1)
price_norm2 = LayerNormalization(name='price_norm2')(price_lstm2)
price_drop2 = Dropout(0.25, name='price_dropout2')(price_norm2)

price_attention = Attention(name='price_attention')([price_drop2, price_drop2])

price_lstm3 = LSTM(32, kernel_regularizer=l2(0.003), name='price_lstm3')(price_attention)
price_output = Dropout(0.15, name='price_dropout3')(price_lstm3)  # 🔧 Reduced from 0.2

# ============ BRANCH 2: Indicators Stream (GRU) ============
indicator_input = Input(shape=(X_indicators.shape[1], X_indicators.shape[2]), name='indicator_input')

indicator_gru1 = Bidirectional(
    GRU(96, return_sequences=True, kernel_regularizer=l2(0.003)),
    name='indicator_gru1'
)(indicator_input)
indicator_norm1 = LayerNormalization(name='indicator_norm1')(indicator_gru1)
indicator_drop1 = Dropout(0.25, name='indicator_dropout1')(indicator_norm1)

indicator_gru2 = Bidirectional(
    GRU(48, return_sequences=True, kernel_regularizer=l2(0.003)),
    name='indicator_gru2'
)(indicator_drop1)
indicator_norm2 = LayerNormalization(name='indicator_norm2')(indicator_gru2)
indicator_drop2 = Dropout(0.25, name='indicator_dropout2')(indicator_norm2)

indicator_gru3 = GRU(24, kernel_regularizer=l2(0.003), name='indicator_gru3')(indicator_drop2)
indicator_output = Dropout(0.15, name='indicator_dropout3')(indicator_gru3)

# ============ MERGE BOTH BRANCHES ============
merged = Concatenate(name='merge_branches')([price_output, indicator_output])

dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.003), name='dense1')(merged)
dense1_norm = LayerNormalization(name='dense1_norm')(dense1)
dense1_drop = Dropout(0.25, name='dense1_dropout')(dense1_norm)

dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.003), name='dense2')(dense1_drop)
dense2_norm = LayerNormalization(name='dense2_norm')(dense2)
dense2_drop = Dropout(0.15, name='dense2_dropout')(dense2_norm)

# Output layer
output = Dense(1, name='output')(dense2_drop)

# Create model
model = Model(inputs=[price_input, indicator_input], outputs=output, name='StockLens_Hybrid_v4_21stocks')

# ==========================================
# COMPILE MODEL
# ==========================================
model.compile(
    optimizer=Adam(learning_rate=0.0001),  
    loss='mse',
    metrics=['mae', 'mse']
)

print("\n" + "="*60)
print("MODEL ARCHITECTURE v4 (21 DIVERSE STOCKS)")
print("="*60)
print("Improvements:")
print("  • Training on 21 diverse stocks (vs 5)")
print("  • Dropout: 0.3 → 0.25 (more learning)")
print("  • L2: 0.005 → 0.003 (less penalty)")
print("  • Better sector diversity")
print("="*60)
model.summary()

# ==========================================
# CALLBACKS
# ==========================================
print("\n[4/5] Setting up callbacks...")
os.makedirs("models", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        min_delta=0.0001,
        verbose=1
    ),
    
    ModelCheckpoint(
        "models/stocklens_hybrid_best.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1
    )
]

# ==========================================
# TRAIN MODEL
# ==========================================
print("\n[5/5] Training model on 21 diverse stocks...")
print(f"Training samples: {len(X):,}")
print("-"*60)

history = model.fit(
    [X_price, X_indicators],
    y,
    epochs=75,  # 🔧 Increased from 50 (more data = more epochs needed)
    batch_size=64,
    validation_split=0.25,
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# SAVE MODEL & HISTORY
# ==========================================
print("\n[SAVE] Saving final model...")
model.save("models/stocklens_hybrid_final.keras")

# Save training history
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'mae': [float(x) for x in history.history['mae']],
    'val_mae': [float(x) for x in history.history['val_mae']],
}

with open("models/training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)

# Save feature split info
split_info = {
    'price_features': [features[i] for i in price_indices],
    'indicator_features': [features[i] for i in indicator_indices],
    'price_indices': price_indices,
    'indicator_indices': indicator_indices
}

with open("models/feature_split.json", "w") as f:
    json.dump(split_info, f, indent=2)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nSaved files:")
print("  ✓ models/stocklens_hybrid_best.keras")
print("  ✓ models/stocklens_hybrid_final.keras")
print("  ✓ models/training_history.json")
print("  ✓ models/feature_split.json")

print("\nBest Performance:")
print(f"  Training Loss:   {min(history.history['loss']):.6f}")
print(f"  Validation Loss: {min(history.history['val_loss']):.6f}")
print(f"  Validation MAE:  {min(history.history['val_mae']):.6f}")

# Calculate train/val ratio
train_val_ratio = min(history.history['val_loss']) / min(history.history['loss'])
print(f"\nOverfitting Check:")
print(f"  Val/Train Loss Ratio: {train_val_ratio:.2f}")
if train_val_ratio < 1.5:
    print("  ✓ Excellent generalization!")
elif train_val_ratio < 2.5:
    print("  ✓ Good generalization")
else:
    print("  ⚠ Some overfitting detected")

print("\n" + "="*60)
print("v4 IMPROVEMENTS (21 STOCKS):")
print("="*60)
print("✓ 21 diverse stocks across sectors")
print("✓ ~39K training samples (vs ~9K)")
print("✓ Lower dropout (0.25) for more learning")
print("✓ Lower L2 (0.003) for less penalty")
print("✓ 75 epochs (vs 50) for larger dataset")
print("="*60)

print("\n[OK] Model ready for testing on unknown stocks!")
print("Run: python -m src.test_unknown_stock")