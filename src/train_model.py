import numpy as np, tensorflow as tf, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load data
X = np.load("data/interim/X.npy")
y = np.load("data/interim/y.npy")

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    GRU(64, return_sequences=True),
    Dropout(0.2),
    GRU(32),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Callbacks
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss"),
    ModelCheckpoint("models/stocklens_lstm_gru_best.h5", save_best_only=True, monitor="val_loss")
]

# Train
history = model.fit(X, y, epochs=50, batch_size=64, validation_split=0.1, callbacks=callbacks)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/stocklens_lstm_gru_final.h5")
print("[OK] Model trained and saved.")

