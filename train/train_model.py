#!/usr/bin/env python3
"""
XL Fitness AI Overseer — Model Trainer
Mac Mini training pipeline step 2.

Trains LSTM on extracted sequences → saves model → ready to deploy to Pi
Usage: python3 train_model.py <sequences.json> <output_model.json>
"""

import json
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

def load_sequences(path):
    with open(path) as f:
        seqs = json.load(f)
    print(f"Loaded {len(seqs)} sequences from {path}")
    counts = {}
    for s in seqs:
        counts[s['label']] = counts.get(s['label'],0)+1
    print(f"Label distribution: {counts}")
    return seqs

def prepare_features(sequences):
    """Extract features and labels."""
    X, y = [], []
    for s in sequences:
        features = [
            s['min_angle'],
            s['rom'],
            s['std_angle'],
            s['duration_s'],
            s['avg_angle'],
        ]
        X.append(features)
        y.append(s['label'])
    return np.array(X), np.array(y)

def train_mlp(X_train, X_test, y_train_enc, y_test_enc, n_classes):
    """Train MLP (fast, for < 500 sequences)."""
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train_enc,
        validation_data=(X_test, y_test_enc),
        epochs=100, batch_size=16, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )
    
    _, acc = model.evaluate(X_test, y_test_enc, verbose=0)
    print(f"MLP Test Accuracy: {acc*100:.1f}%")
    return model, acc

def export_model_json(model, scaler, label_encoder, accuracy, sequences, output_path):
    """Export model to JSON format (for browser demo and Pi inference)."""
    weights = [w.numpy().tolist() for w in model.weights[::2]]  # weights only
    biases = [w.numpy().tolist() for w in model.weights[1::2]]  # biases only
    
    model_data = {
        'version': 5,
        'type': 'MLP',
        'classes': label_encoder.classes_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'weights': weights,
        'biases': biases,
        'features': ['min_angle', 'rom', 'std_angle', 'duration_s', 'avg_angle'],
        'cv_accuracy': round(accuracy*100, 1),
        'trained_on': {
            label: int(np.sum(np.array([s['label'] for s in sequences]) == label))
            for label in label_encoder.classes_
        }
    }
    model_data['trained_on']['total'] = len(sequences)
    
    with open(output_path, 'w') as f:
        json.dump(model_data, f)
    
    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ Model saved: {output_path} ({size_kb:.0f}KB)")

def export_tflite(model, output_path):
    """Export to TFLite for Pi deployment."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = output_path.replace('.json', '.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite saved: {tflite_path} ({len(tflite_model)/1024:.0f}KB)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 train_model.py <sequences.json> <output_model.json>")
        sys.exit(1)
    
    sequences_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Load data
    sequences = load_sequences(sequences_path)
    X, y = prepare_features(sequences)
    
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Classes: {le.classes_}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)}...")
    
    # Train
    n_classes = len(le.classes_)
    model, accuracy = train_mlp(X_train, X_test, y_train, y_test, n_classes)
    
    # Classification report
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Export
    export_model_json(model, scaler, le, accuracy, sequences, output_path)
    export_tflite(model, output_path)
    
    print(f"\n🎯 Model v5 ready: {accuracy*100:.1f}% accuracy on {len(sequences)} sequences")
    print(f"Deploy JSON to: pose/model.json (browser demo)")
    print(f"Deploy TFLite to: pi/ folder (Raspberry Pi)")
