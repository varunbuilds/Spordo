#!/usr/bin/env python3
"""
Advanced LSTM Training with Data Augmentation and Improved Learning
Addresses severe class imbalance with SMOTE-like technique for time series
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import json

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

def augment_sequence(sequence, num_augmentations=3):
    """
    Augment pose sequences with variations
    """
    augmented = [sequence]
    
    for _ in range(num_augmentations):
        aug_seq = sequence.copy()
        
        # Time warping (speed up/slow down)
        speed_factor = np.random.uniform(0.9, 1.1)
        indices = np.linspace(0, len(sequence)-1, int(len(sequence) * speed_factor))
        indices = np.clip(indices, 0, len(sequence)-1).astype(int)
        aug_seq = sequence[indices]
        
        # Add small noise
        noise = np.random.normal(0, 0.02, aug_seq.shape)
        aug_seq = aug_seq + noise
        
        # Random temporal dropout (simulate dropped frames)
        if np.random.random() < 0.3:
            drop_idx = np.random.choice(len(aug_seq), size=max(1, len(aug_seq)//10), replace=False)
            # Forward fill dropped frames
            for idx in sorted(drop_idx):
                if idx > 0:
                    aug_seq[idx] = aug_seq[idx-1]
        
        augmented.append(aug_seq)
    
    return augmented

def balance_dataset_with_augmentation(sequences, labels, target_samples_per_class=150):
    """
    Balance dataset by augmenting minority classes
    """
    class_counts = Counter(labels)
    unique_labels = list(class_counts.keys())
    
    balanced_sequences = []
    balanced_labels = []
    
    for label in unique_labels:
        # Get all sequences for this class
        class_indices = [i for i, l in enumerate(labels) if l == label]
        class_sequences = [sequences[i] for i in class_indices]
        
        current_count = len(class_sequences)
        
        if current_count < target_samples_per_class:
            # Need augmentation
            augmentations_needed = target_samples_per_class - current_count
            aug_per_sample = max(1, augmentations_needed // current_count)
            
            print(f"  {label:15s}: {current_count:3d} → {target_samples_per_class:3d} (augmenting {augmentations_needed:3d})")
            
            # Add original sequences
            balanced_sequences.extend(class_sequences)
            balanced_labels.extend([label] * current_count)
            
            # Augment
            augmented_count = 0
            while augmented_count < augmentations_needed:
                for seq in class_sequences:
                    if augmented_count >= augmentations_needed:
                        break
                    aug_sequences = augment_sequence(seq, num_augmentations=1)
                    balanced_sequences.extend(aug_sequences[1:])  # Skip original
                    balanced_labels.extend([label] * (len(aug_sequences) - 1))
                    augmented_count += len(aug_sequences) - 1
        else:
            # Already have enough samples
            print(f"  {label:15s}: {current_count:3d} (no augmentation needed)")
            balanced_sequences.extend(class_sequences)
            balanced_labels.extend([label] * current_count)
    
    return balanced_sequences, balanced_labels

class AdvancedCricketShotTrainer:
    def __init__(self, data_path='data/pose_sequences.pkl', model_dir='models'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        
        self.sequence_length = 60
        self.num_features = 8
        self.num_classes = None
        
        os.makedirs(model_dir, exist_ok=True)
    
    def load_and_prepare_data(self, use_augmentation=True):
        """Load and prepare data with optional augmentation"""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        sequences = data['sequences']
        labels = data['labels']
        
        print(f"✓ Loaded {len(sequences)} sequences")
        
        # Show original distribution
        class_counts = Counter(labels)
        print(f"\nOriginal Distribution:")
        for cls, count in sorted(class_counts.items()):
            percentage = (count / len(labels)) * 100
            print(f"  {cls:15s}: {count:4d} ({percentage:5.1f}%)")
        
        # Balance with augmentation
        if use_augmentation:
            print(f"\n" + "="*70)
            print("BALANCING DATASET WITH AUGMENTATION")
            print("="*70)
            sequences, labels = balance_dataset_with_augmentation(
                sequences, labels, target_samples_per_class=150
            )
            
            print(f"\nBalanced Distribution:")
            class_counts = Counter(labels)
            for cls, count in sorted(class_counts.items()):
                percentage = (count / len(labels)) * 100
                print(f"  {cls:15s}: {count:4d} ({percentage:5.1f}%)")
        
        # Prepare sequences
        X = []
        for seq in sequences:
            if len(seq) >= self.sequence_length:
                X.append(seq[:self.sequence_length])
            else:
                padding = np.zeros((self.sequence_length - len(seq), self.num_features))
                X.append(np.vstack([seq, padding]))
        
        X = np.array(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        y_categorical = keras.utils.to_categorical(y_encoded)
        
        self.num_classes = len(self.label_encoder.classes_)
        print(f"\n✓ Prepared data: {X.shape}, Classes: {self.num_classes}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_categorical, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        y_temp_labels = np.argmax(y_temp, axis=1)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_labels
        )
        
        print(f"✓ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self):
        """Build optimized LSTM model"""
        print("\n" + "="*70)
        print("BUILDING MODEL")
        print("="*70)
        
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            
            # Bidirectional LSTM layers
            layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.2)),
            layers.BatchNormalization(),
            
            layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.2)),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile with lower initial learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        self.model = model
        model.summary()
        return model
    
    def create_callbacks(self):
        """Create training callbacks"""
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'cricket_lstm_advanced.weights.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                mode='max'
            )
        ]
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=100, batch_size=32):
        """Train the model"""
        print("\n" + "="*70)
        print("TRAINING MODEL")
        print("="*70)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.create_callbacks(),
            verbose=1
        )
        
        print(f"\n✓ Training completed!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model"""
        print("\n" + "="*70)
        print("EVALUATING MODEL")
        print("="*70)
        
        loss, accuracy, top3_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Results:")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Top-3 Accuracy: {top3_accuracy*100:.2f}%")
        print(f"  Loss: {loss:.4f}")
        
        # Per-class evaluation
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print(f"\nPer-Class Accuracy:")
        for idx, cls_name in enumerate(self.label_encoder.classes_):
            mask = y_true_classes == idx
            if mask.sum() > 0:
                cls_acc = (y_pred_classes[mask] == idx).mean()
                print(f"  {cls_name:15s}: {cls_acc*100:5.1f}% ({mask.sum()} samples)")
        
        return accuracy, y_pred_classes, y_true_classes
    
    def plot_results(self, y_true, y_pred):
        """Plot training history and confusion matrix"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Val')
        axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Val')
        axes[1].set_title('Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Top-3 Accuracy
        axes[2].plot(self.history.history['top_3_accuracy'], label='Train')
        axes[2].plot(self.history.history['val_top_3_accuracy'], label='Val')
        axes[2].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_advanced.png', dpi=150)
        print(f"\n✓ Saved training history")
        plt.close()
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix - Advanced Model', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix_advanced.png', dpi=150)
        print(f"✓ Saved confusion matrix")
        plt.close()
    
    def save_model(self):
        """Save model and metadata"""
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        model_path = os.path.join(self.model_dir, 'cricket_lstm_advanced.h5')
        self.model.save(model_path)
        print(f"✓ Saved model to {model_path}")
        
        metadata = {
            'classes': self.label_encoder.classes_.tolist(),
            'num_classes': self.num_classes,
            'sequence_length': self.sequence_length,
            'num_features': self.num_features,
            'model_type': 'advanced_lstm_augmented'
        }
        
        with open(os.path.join(self.model_dir, 'cricket_lstm_advanced.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        with open(os.path.join(self.model_dir, 'cricket_lstm_advanced.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata")

def main():
    print("\n" + "="*70)
    print("ADVANCED CRICKET SHOT CLASSIFICATION")
    print("="*70)
    print("Features:")
    print("  ✓ Data augmentation (time warping, noise, dropout)")
    print("  ✓ Balanced dataset (150 samples per class)")
    print("  ✓ Optimized architecture (lighter, faster)")
    print("  ✓ Lower learning rate (0.0005)")
    print("  ✓ Longer patience (better convergence)")
    print("="*70)
    
    trainer = AdvancedCricketShotTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_and_prepare_data(use_augmentation=True)
    trainer.build_model()
    trainer.train_model(X_train, X_val, y_train, y_val, epochs=100, batch_size=32)
    accuracy, y_pred, y_true = trainer.evaluate_model(X_test, y_test)
    trainer.plot_results(y_true, y_pred)
    trainer.save_model()
    
    print("\n" + "="*70)
    print(f"FINAL TEST ACCURACY: {accuracy*100:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
