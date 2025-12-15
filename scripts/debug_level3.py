#!/usr/bin/env python3
"""
Debug script để kiểm tra level 3 training issues
"""

import pandas as pd
import numpy as np
import sys
import os

def check_level3_data():
    """Kiểm tra data level 3"""
    print("🔍 Checking Level 3 Data...")

    try:
        # Load data
        df = pd.read_pickle('dataset/splits/level3/train_balanced.pkl')
        print(f"Data shape: {df.shape}")

        # Find label column
        label_cols = [c for c in df.columns if 'label' in c and 'encoded' in c]
        if not label_cols:
            print("❌ No encoded label column found!")
            return

        label_col = label_cols[0]
        print(f"Label column: {label_col}")

        # Check labels
        labels = df[label_col].values
        unique_labels = np.unique(labels)
        print(f"Unique labels: {unique_labels}")
        print(f"Label range: {np.min(labels)} to {np.max(labels)}")
        print(f"Label distribution: {np.bincount(labels)}")

        # Check for feature columns
        feature_cols = [c for c in df.columns if not 'label' in c.lower()]
        print(f"Feature columns: {len(feature_cols)}")

        # Check for NaN/inf
        X = df[feature_cols].values
        print(f"Features shape: {X.shape}")
        print(f"NaN values: {np.isnan(X).sum()}")
        print(f"Inf values: {np.isinf(X).sum()}")

        # Check data types
        print(f"Data types: {df.dtypes.value_counts()}")

        return True

    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False

def test_simple_model():
    """Test với model đơn giản"""
    print("\n🧪 Testing Simple Model...")

    try:
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        # Load data
        df = pd.read_pickle('dataset/splits/level3/train_balanced.pkl')
        label_col = [c for c in df.columns if 'label' in c and 'encoded' in c][0]
        feature_cols = [c for c in df.columns if not 'label' in c.lower()]

        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values

        # Simple preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Re-index labels from 0
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        print(f"After preprocessing: X={X_scaled.shape}, y_unique={np.unique(y_encoded)}")

        # Simple model
        input_shape = (X_scaled.shape[1], 1)
        X_reshaped = X_scaled.reshape(-1, input_shape[0], 1)

        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test on small batch
        X_test = X_reshaped[:32]
        y_test = y_encoded[:32]

        print("Testing forward pass...")
        predictions = model.predict(X_test, verbose=0)
        print(f"Predictions shape: {predictions.shape}")

        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss/accuracy: {loss}")

        # Train for 1 epoch
        print("Training for 1 epoch...")
        history = model.fit(
            X_reshaped[:100], y_encoded[:100],
            validation_split=0.2,
            epochs=1,
            batch_size=16,
            verbose=1
        )

        return True

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔧 Debug Level 3 Training Issues")
    print("="*50)

    success = True

    # Check data
    if not check_level3_data():
        success = False

    # Test simple model
    if not test_simple_model():
        success = False

    if success:
        print("\n✅ Debug completed successfully")
        print("💡 If simple model works but complex model doesn't:")
        print("   - Check model architecture (gradient flow)")
        print("   - Check loss function compatibility")
        print("   - Try reducing model complexity")
    else:
        print("\n❌ Debug found issues - check output above")

if __name__ == "__main__":
    main()

