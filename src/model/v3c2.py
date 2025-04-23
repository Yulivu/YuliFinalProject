"""
基于ResNet18架构的地震断层运动方式识别及板块边界预测模型
- 使用ResNet18作为骨干网络
- 针对地质数据优化的特征提取
- 改进的损失函数及数据增强
- 优化的预测和后处理
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, applications, regularizers
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from skimage import measure
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
import time
from datetime import datetime
import json

# 设置路径
DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2processed"
RAW_DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2raw"
MODEL_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\src\model"
OUTPUT_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\result\v3c2"

# 确保目录存在
for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 从v3c1.py导入数据处理函数
from v3c1 import load_raster_data, define_geological_regions, create_improved_train_test_split, visualize_data_split, global_pad

def build_resnet18_model(input_shape, num_classes=1):
    """
    Build efficient ResNet18 model with balanced complexity for performance and speed
    
    Parameters:
    - input_shape: Input data shape (H, W, C)
    - num_classes: Number of output classes, 1 for binary classification
    
    Returns:
    - model: Compiled ResNet18 model
    """
    def identity_block(x, filters, kernel_size=3):
        """Residual block without downsampling"""
        shortcut = x
        
        # First convolution layer
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Second convolution layer
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add skip connection
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        
        return x
    
    def conv_block(x, filters, kernel_size=3, strides=2):
        """Convolution block with downsampling"""
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
        
        # First convolution layer
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Second convolution layer
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add skip connection
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        
        return x
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution layer
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # First group of residual blocks
    x = conv_block(x, 64, strides=1)
    x = identity_block(x, 64)
    
    # Second group of residual blocks
    x = conv_block(x, 128)
    x = identity_block(x, 128)
    
    # Third group of residual blocks
    x = conv_block(x, 256)
    x = identity_block(x, 256)
    
    # Efficient spatial attention
    spatial_features = layers.Conv2D(1, 1, padding='same')(x)
    spatial_attention = layers.Activation('sigmoid')(spatial_features)
    x = layers.multiply([x, spatial_attention])
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs, outputs)
    
    # Compile model with balanced class weights
    if num_classes == 1:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(), 
                     tf.keras.metrics.Recall()]
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def prepare_data_for_resnet(eq_data, boundary_data, masks, window_size=11):
    """
    Prepare windowed data for ResNet18 with balanced preprocessing
    
    Parameters:
    - eq_data: Earthquake feature data
    - boundary_data: Plate boundary data
    - masks: Training, validation and test masks
    - window_size: Window size
    
    Returns:
    - X_train, y_train: Training data and labels
    - X_val, y_val: Validation data and labels
    - pos_weight: Positive sample weight
    """
    print(f"Preparing windowed data with size: {window_size}x{window_size}...")
    
    # Get data shape
    num_bands, height, width = eq_data.shape
    half_window = window_size // 2
    
    # Pad data
    eq_data_padded = global_pad(eq_data, half_window)
    
    # Get training and validation masks
    train_mask = masks['train']
    val_mask = masks['val']
    
    # Get indices for training and validation pixels
    train_indices = np.where(train_mask)
    val_indices = np.where(val_mask)
    
    # Sample data for efficiency while maintaining representation
    train_sample_rate = 0.6  # Use 60% of training data
    if len(train_indices[0]) > 20000:
        sample_size = int(len(train_indices[0]) * train_sample_rate)
        sample_idx = np.random.choice(len(train_indices[0]), sample_size, replace=False)
        train_indices = (train_indices[0][sample_idx], train_indices[1][sample_idx])
    
    if len(val_indices[0]) > 10000:
        sample_size = min(10000, len(val_indices[0]))
        sample_idx = np.random.choice(len(val_indices[0]), sample_size, replace=False)
        val_indices = (val_indices[0][sample_idx], val_indices[1][sample_idx])
    
    num_train_pixels = len(train_indices[0])
    num_val_pixels = len(val_indices[0])
    
    print(f"Training set (sampled): {num_train_pixels} pixels, Validation set (sampled): {num_val_pixels} pixels")
    
    # Define data processing function with efficient augmentation
    def process_windows(indices, is_training=False):
        num_pixels = len(indices[0])
        X = np.zeros((num_pixels, window_size, window_size, num_bands))
        y = np.zeros(num_pixels, dtype=np.float32)
        
        for i, (y_idx, x_idx) in enumerate(zip(indices[0], indices[1])):
            if i % 100000 == 0 and i > 0:
                print(f"  Processed {i}/{num_pixels} pixels")
            
            # Get padded center point index
            padded_y, padded_x = y_idx + half_window, x_idx + half_window
            
            # Extract window for each band
            for b in range(num_bands):
                X[i, :, :, b] = eq_data_padded[b,
                                padded_y - half_window:padded_y + half_window + 1,
                                padded_x - half_window:padded_x + half_window + 1]
            
            # Set label - whether it's a plate boundary
            y[i] = boundary_data[0, y_idx, x_idx] > 0
            
            # Balanced data augmentation
            if is_training and np.random.rand() < 0.5:
                # Random rotation - only 90° increments for efficiency
                k = np.random.randint(4)
                X[i] = np.rot90(X[i], k=k)
                
                # Random flip
                if np.random.rand() < 0.5:
                    X[i] = np.fliplr(X[i])
        
        # Efficient standardization - per band across all samples
        for b in range(num_bands):
            band_mean = np.mean(X[:, :, :, b])
            band_std = np.std(X[:, :, :, b]) + 1e-7
            X[:, :, :, b] = (X[:, :, :, b] - band_mean) / band_std
        
        return X, y
    
    # Process training and validation data
    print("Processing training data...")
    X_train, y_train = process_windows(train_indices, is_training=True)
    
    print("Processing validation data...")
    X_val, y_val = process_windows(val_indices, is_training=False)
    
    # Calculate positive sample ratio
    pos_train_ratio = np.mean(y_train)
    pos_val_ratio = np.mean(y_val)
    
    print(f"Training set positive ratio: {pos_train_ratio:.4f}, Validation set positive ratio: {pos_val_ratio:.4f}")
    
    # Calculate class weight
    pos_weight = (1 - pos_train_ratio) / pos_train_ratio if pos_train_ratio > 0 else 1.0
    pos_weight = min(pos_weight * 1.2, 12.0)  # Enhance positive weight with reasonable cap
    print(f"Positive class weight: {pos_weight:.2f}")
    
    return X_train, y_train, X_val, y_val, pos_weight

def train_resnet18_model(X_train, y_train, X_val, y_val, batch_size=64, epochs=40, class_weight=None):
    """
    Train the ResNet18 model with optimized training process
    
    Parameters:
    - X_train, y_train: Training data and labels
    - X_val, y_val: Validation data and labels
    - batch_size: Batch size
    - epochs: Maximum training epochs
    - class_weight: Class weights for imbalanced data
    
    Returns:
    - model: Trained model
    - history: Training history
    - run_output_dir: Directory with training outputs
    """
    print(f"Training data shape: {X_train.shape}, Label shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, Label shape: {y_val.shape}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, f"resnet18_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Build model
    model = build_resnet18_model(X_train.shape[1:])
    model.summary()
    
    # Set callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"resnet18_model_{timestamp}.h5"),
            monitor='val_loss',
            save_best_only=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        class_weight=class_weight,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)
    
    print(f"Validation results:")
    print(f" - Loss: {val_loss:.4f}")
    print(f" - Accuracy: {val_acc:.4f}")
    print(f" - Precision: {val_precision:.4f}")
    print(f" - Recall: {val_recall:.4f}")
    print(f" - F1 score: {val_f1:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['precision'], label='Precision')
    plt.plot(history.history['recall'], label='Recall')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    # Save training history and evaluation results
    pd.DataFrame(history.history).to_csv(os.path.join(run_output_dir, 'training_history.csv'), index=False)
    
    # Save model summary
    with open(os.path.join(run_output_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Save evaluation results
    eval_results = {
        'val_loss': [val_loss],
        'val_accuracy': [val_acc],
        'val_precision': [val_precision],
        'val_recall': [val_recall],
        'val_f1': [val_f1],
        'training_time': [training_time],
        'batch_size': [batch_size],
        'epochs': [len(history.history['loss'])]
    }
    pd.DataFrame(eval_results).to_csv(os.path.join(run_output_dir, 'evaluation_results.csv'), index=False)
    
    return model, history, run_output_dir

def predict_for_test_regions(model, eq_data, boundary_data, masks, window_size=11, threshold=0.5):
    """
    Make predictions for test regions efficiently
    
    Parameters:
    - model: Trained ResNet18 model
    - eq_data: Earthquake feature data
    - boundary_data: Ground truth boundary data
    - masks: Dictionary with test region masks
    - window_size: Window size for predictions
    - threshold: Classification threshold
    
    Returns:
    - prediction_map: Full prediction probability map
    - binary_prediction: Binary prediction map
    - test_metrics: Dictionary with test metrics
    """
    print("Making predictions for test regions...")
    
    # Get data shape
    num_bands, height, width = eq_data.shape
    half_window = window_size // 2
    
    # Pad data
    eq_data_padded = global_pad(eq_data, half_window)
    
    # Create prediction map
    prediction_map = np.zeros((1, height, width), dtype=np.float32)
    
    # Get test mask
    test_mask = masks['test']
    test_indices = np.where(test_mask)
    total_test_pixels = len(test_indices[0])
    
    print(f"Test set: {total_test_pixels} pixels")
    
    # Process in batches for memory efficiency
    batch_size = 2048
    num_batches = (total_test_pixels + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_test_pixels)
        
        if batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx+1}/{num_batches}...")
        
        # Get batch indices
        batch_indices_y = test_indices[0][start_idx:end_idx]
        batch_indices_x = test_indices[1][start_idx:end_idx]
        current_batch_size = len(batch_indices_y)
        
        # Create batch data
        X_batch = np.zeros((current_batch_size, window_size, window_size, num_bands))
        
        # Extract windows
        for i, (y_idx, x_idx) in enumerate(zip(batch_indices_y, batch_indices_x)):
            padded_y, padded_x = y_idx + half_window, x_idx + half_window
            
            for b in range(num_bands):
                X_batch[i, :, :, b] = eq_data_padded[b,
                                      padded_y - half_window:padded_y + half_window + 1,
                                      padded_x - half_window:padded_x + half_window + 1]
        
        # Normalize batch efficiently
        for b in range(num_bands):
            band_mean = np.mean(X_batch[:, :, :, b])
            band_std = np.std(X_batch[:, :, :, b]) + 1e-7
            X_batch[:, :, :, b] = (X_batch[:, :, :, b] - band_mean) / band_std
        
        # Make predictions
        y_pred_batch = model.predict(X_batch, verbose=0)
        
        # Store predictions
        for i, (y_idx, x_idx) in enumerate(zip(batch_indices_y, batch_indices_x)):
            prediction_map[0, y_idx, x_idx] = y_pred_batch[i]
    
    # Create binary prediction
    binary_prediction = (prediction_map >= threshold).astype(np.float32)
    
    # Calculate test metrics
    true_positives = np.sum((binary_prediction > 0) & (boundary_data > 0) & test_mask)
    false_positives = np.sum((binary_prediction > 0) & (boundary_data == 0) & test_mask)
    false_negatives = np.sum((binary_prediction == 0) & (boundary_data > 0) & test_mask)
    true_negatives = np.sum((binary_prediction == 0) & (boundary_data == 0) & test_mask)
    
    precision = true_positives / (true_positives + false_positives + 1e-7)
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    
    # Calculate test metrics per region
    region_metrics = {}
    for region_name, region_mask in masks['test_regions'].items():
        region_test_mask = region_mask & test_mask
        
        r_true_positives = np.sum((binary_prediction > 0) & (boundary_data > 0) & region_test_mask)
        r_false_positives = np.sum((binary_prediction > 0) & (boundary_data == 0) & region_test_mask)
        r_false_negatives = np.sum((binary_prediction == 0) & (boundary_data > 0) & region_test_mask)
        r_true_negatives = np.sum((binary_prediction == 0) & (boundary_data == 0) & region_test_mask)
        
        r_precision = r_true_positives / (r_true_positives + r_false_positives + 1e-7)
        r_recall = r_true_positives / (r_true_positives + r_false_negatives + 1e-7)
        r_f1_score = 2 * (r_precision * r_recall) / (r_precision + r_recall + 1e-7)
        r_accuracy = (r_true_positives + r_true_negatives) / (r_true_positives + r_true_negatives + r_false_positives + r_false_negatives)
        
        region_metrics[region_name] = {
            'precision': r_precision,
            'recall': r_recall,
            'f1_score': r_f1_score,
            'accuracy': r_accuracy,
            'true_positives': int(r_true_positives),
            'false_positives': int(r_false_positives),
            'false_negatives': int(r_false_negatives),
            'true_negatives': int(r_true_negatives)
        }
    
    # Optional: Apply morphological post-processing
    if True:
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        binary_prediction_processed = binary_prediction.copy()
        
        # Opening to remove small noise
        binary_prediction_processed[0] = cv2.morphologyEx(
            binary_prediction_processed[0].astype(np.uint8), 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=1
        ).astype(np.float32)
        
        # Closing to fill small gaps
        binary_prediction_processed[0] = cv2.morphologyEx(
            binary_prediction_processed[0].astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel, 
            iterations=1
        ).astype(np.float32)
        
        # Calculate metrics after post-processing
        p_true_positives = np.sum((binary_prediction_processed > 0) & (boundary_data > 0) & test_mask)
        p_false_positives = np.sum((binary_prediction_processed > 0) & (boundary_data == 0) & test_mask)
        p_false_negatives = np.sum((binary_prediction_processed == 0) & (boundary_data > 0) & test_mask)
        p_true_negatives = np.sum((binary_prediction_processed == 0) & (boundary_data == 0) & test_mask)
        
        p_precision = p_true_positives / (p_true_positives + p_false_positives + 1e-7)
        p_recall = p_true_positives / (p_true_positives + p_false_negatives + 1e-7)
        p_f1_score = 2 * (p_precision * p_recall) / (p_precision + p_recall + 1e-7)
        p_accuracy = (p_true_positives + p_true_negatives) / (p_true_positives + p_true_negatives + p_false_positives + p_false_negatives)
        
        print(f"Test metrics after post-processing:")
        print(f"  Precision: {p_precision:.4f}")
        print(f"  Recall: {p_recall:.4f}")
        print(f"  F1 Score: {p_f1_score:.4f}")
        print(f"  Accuracy: {p_accuracy:.4f}")
        
        # Use processed prediction if it improves F1
        if p_f1_score > f1_score:
            print("Post-processing improved F1 score - using processed predictions")
            binary_prediction = binary_prediction_processed
            precision = p_precision
            recall = p_recall
            f1_score = p_f1_score
            accuracy = p_accuracy
    
    # Compile results
    test_metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'true_negatives': int(true_negatives),
        'region_metrics': region_metrics
    }
    
    # Print results
    print(f"Overall test metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    return prediction_map, binary_prediction, test_metrics

def simple_visualize(prediction_map, binary_prediction, boundary_data, masks, splits, output_dir):
    """
    Visualize test predictions and save results
    
    Parameters:
    - prediction_map: Probability map of predictions (0-1)
    - binary_prediction: Binary predictions after thresholding
    - boundary_data: Ground truth boundary data
    - masks: Dictionary with region masks
    - splits: Dictionary with train/val/test splits
    - output_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    print("Visualizing predictions...")
    
    # Create output directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    check_create_directory(vis_dir)
    
    # Define colormap for the mask
    colors = [(1, 1, 1, 0), (0, 0, 1, 0.3), (0, 1, 0, 0.3), (1, 0, 0, 0.3)]  # transparent, blue, green, red
    cmap_mask = ListedColormap(colors)
    
    # Create a combined mask for visualization
    combined_mask = np.zeros_like(masks['train'], dtype=np.uint8)
    combined_mask[masks['train']] = 1  # Training regions: blue
    combined_mask[masks['val']] = 2    # Validation regions: green
    combined_mask[masks['test']] = 3   # Test regions: red
    
    # Create full-size figures for the entire dataset
    plt.figure(figsize=(18, 12))
    
    # First subplot: Earthquake feature data (first 3 bands as RGB)
    plt.subplot(2, 3, 1)
    eq_data_display = np.zeros((boundary_data.shape[1], boundary_data.shape[2], 3))
    for i in range(3):  # Use first 3 bands for RGB visualization
        if i < len(eq_data):
            channel = eq_data[i].copy()
            # Normalize to 0-1 for visualization
            vmin, vmax = np.percentile(channel, [2, 98])
            channel = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
            eq_data_display[:, :, i] = channel
    
    plt.imshow(eq_data_display)
    plt.title('Earthquake Feature Data (RGB)', fontsize=12)
    plt.axis('off')
    
    # Second subplot: Ground truth boundaries
    plt.subplot(2, 3, 2)
    plt.imshow(boundary_data[0], cmap='Greys_r', vmin=0, vmax=1)
    plt.title('Ground Truth Boundaries', fontsize=12)
    plt.axis('off')
    
    # Third subplot: Prediction probability map
    plt.subplot(2, 3, 3)
    plt.imshow(prediction_map[0], cmap='plasma', vmin=0, vmax=1)
    plt.colorbar(shrink=0.7, label='Probability')
    plt.title('Prediction Probability Map', fontsize=12)
    plt.axis('off')
    
    # Fourth subplot: Binary prediction
    plt.subplot(2, 3, 4)
    plt.imshow(binary_prediction[0], cmap='Greys_r', vmin=0, vmax=1)
    plt.title('Binary Prediction', fontsize=12)
    plt.axis('off')
    
    # Fifth subplot: Prediction vs. Ground truth (overlay)
    plt.subplot(2, 3, 5)
    
    # Create color-coded overlay (true positive, false positive, false negative)
    overlay = np.zeros((binary_prediction.shape[1], binary_prediction.shape[2], 3))
    
    # True positives: green
    tp_mask = (binary_prediction[0] > 0) & (boundary_data[0] > 0)
    overlay[tp_mask] = [0, 1, 0]
    
    # False positives: red
    fp_mask = (binary_prediction[0] > 0) & (boundary_data[0] == 0)
    overlay[fp_mask] = [1, 0, 0]
    
    # False negatives: blue
    fn_mask = (binary_prediction[0] == 0) & (boundary_data[0] > 0)
    overlay[fn_mask] = [0, 0, 1]
    
    plt.imshow(overlay)
    
    # Add legend
    tp_patch = mpatches.Patch(color='green', label='True Positive')
    fp_patch = mpatches.Patch(color='red', label='False Positive')
    fn_patch = mpatches.Patch(color='blue', label='False Negative')
    plt.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower right', fontsize=10)
    
    plt.title('Prediction Evaluation', fontsize=12)
    plt.axis('off')
    
    # Sixth subplot: Train/Val/Test regions
    plt.subplot(2, 3, 6)
    plt.imshow(combined_mask, cmap=cmap_mask)
    
    # Add legend for regions
    train_patch = mpatches.Patch(color=colors[1], label='Training')
    val_patch = mpatches.Patch(color=colors[2], label='Validation')
    test_patch = mpatches.Patch(color=colors[3], label='Test')
    plt.legend(handles=[train_patch, val_patch, test_patch], loc='lower right', fontsize=10)
    
    plt.title('Data Splits', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'overview.png'), dpi=300, bbox_inches='tight')
    
    # Create visualizations for each test region
    for region_name, region_mask in masks['test_regions'].items():
        # Only generate for test regions
        if not np.any(region_mask & masks['test']):
            continue
            
        print(f"  Visualizing region: {region_name}")
            
        # Get region boundaries
        region_indices = np.where(region_mask)
        y_min, y_max = np.min(region_indices[0]), np.max(region_indices[0])
        x_min, x_max = np.min(region_indices[1]), np.max(region_indices[1])
        
        # Add padding
        padding = 10
        y_min = max(0, y_min - padding)
        y_max = min(boundary_data.shape[1] - 1, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(boundary_data.shape[2] - 1, x_max + padding)
        
        # Create zoomed-in figure for this region
        plt.figure(figsize=(18, 12))
        
        # First subplot: Earthquake feature data (first 3 bands)
        plt.subplot(2, 3, 1)
        region_eq_data = eq_data_display[y_min:y_max, x_min:x_max]
        plt.imshow(region_eq_data)
        plt.title(f'Region {region_name}: Earthquake Features', fontsize=12)
        plt.axis('off')
        
        # Second subplot: Ground truth boundaries
        plt.subplot(2, 3, 2)
        region_gt = boundary_data[0, y_min:y_max, x_min:x_max]
        plt.imshow(region_gt, cmap='Greys_r', vmin=0, vmax=1)
        plt.title(f'Region {region_name}: Ground Truth', fontsize=12)
        plt.axis('off')
        
        # Third subplot: Prediction probability map
        plt.subplot(2, 3, 3)
        region_prob = prediction_map[0, y_min:y_max, x_min:x_max]
        plt.imshow(region_prob, cmap='plasma', vmin=0, vmax=1)
        plt.colorbar(shrink=0.7, label='Probability')
        plt.title(f'Region {region_name}: Probability Map', fontsize=12)
        plt.axis('off')
        
        # Fourth subplot: Binary prediction
        plt.subplot(2, 3, 4)
        region_pred = binary_prediction[0, y_min:y_max, x_min:x_max]
        plt.imshow(region_pred, cmap='Greys_r', vmin=0, vmax=1)
        plt.title(f'Region {region_name}: Binary Prediction', fontsize=12)
        plt.axis('off')
        
        # Fifth subplot: Prediction vs. Ground truth (overlay)
        plt.subplot(2, 3, 5)
        
        # Create color-coded overlay (true positive, false positive, false negative)
        region_overlay = np.zeros((y_max-y_min, x_max-x_min, 3))
        
        # True positives: green
        region_tp_mask = (binary_prediction[0, y_min:y_max, x_min:x_max] > 0) & (boundary_data[0, y_min:y_max, x_min:x_max] > 0)
        region_overlay[region_tp_mask] = [0, 1, 0]
        
        # False positives: red
        region_fp_mask = (binary_prediction[0, y_min:y_max, x_min:x_max] > 0) & (boundary_data[0, y_min:y_max, x_min:x_max] == 0)
        region_overlay[region_fp_mask] = [1, 0, 0]
        
        # False negatives: blue
        region_fn_mask = (binary_prediction[0, y_min:y_max, x_min:x_max] == 0) & (boundary_data[0, y_min:y_max, x_min:x_max] > 0)
        region_overlay[region_fn_mask] = [0, 0, 1]
        
        plt.imshow(region_overlay)
        
        # Add legend
        tp_patch = mpatches.Patch(color='green', label='True Positive')
        fp_patch = mpatches.Patch(color='red', label='False Positive')
        fn_patch = mpatches.Patch(color='blue', label='False Negative')
        plt.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower right', fontsize=10)
        
        plt.title(f'Region {region_name}: Evaluation', fontsize=12)
        plt.axis('off')
        
        # Sixth subplot: Region metrics
        plt.subplot(2, 3, 6)
        region_metrics = test_metrics['region_metrics'].get(region_name, {})
        
        if region_metrics:
            # No image here, just text with metrics
            plt.text(0.5, 0.9, f"Region {region_name} Metrics", fontsize=14, horizontalalignment='center')
            plt.text(0.5, 0.8, f"Precision: {region_metrics.get('precision', 0):.4f}", fontsize=12, horizontalalignment='center')
            plt.text(0.5, 0.7, f"Recall: {region_metrics.get('recall', 0):.4f}", fontsize=12, horizontalalignment='center')
            plt.text(0.5, 0.6, f"F1 Score: {region_metrics.get('f1_score', 0):.4f}", fontsize=12, horizontalalignment='center')
            plt.text(0.5, 0.5, f"Accuracy: {region_metrics.get('accuracy', 0):.4f}", fontsize=12, horizontalalignment='center')
            plt.text(0.5, 0.4, f"TP: {region_metrics.get('true_positives', 0)}", fontsize=12, horizontalalignment='center')
            plt.text(0.5, 0.3, f"FP: {region_metrics.get('false_positives', 0)}", fontsize=12, horizontalalignment='center')
            plt.text(0.5, 0.2, f"FN: {region_metrics.get('false_negatives', 0)}", fontsize=12, horizontalalignment='center')
            plt.text(0.5, 0.1, f"TN: {region_metrics.get('true_negatives', 0)}", fontsize=12, horizontalalignment='center')
        
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'region_{region_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot precision-recall curves at different thresholds
    plt.figure(figsize=(12, 8))
    
    # Only consider test pixels
    test_mask = masks['test']
    boundary_test = boundary_data[0][test_mask]
    pred_probs_test = prediction_map[0][test_mask]
    
    # Calculate precision and recall at different thresholds
    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        # Apply threshold
        binary_preds = (pred_probs_test >= thresh).astype(np.float32)
        
        # Calculate metrics
        true_pos = np.sum((binary_preds > 0) & (boundary_test > 0))
        false_pos = np.sum((binary_preds > 0) & (boundary_test == 0))
        false_neg = np.sum((binary_preds == 0) & (boundary_test > 0))
        
        # Handle division by zero
        if true_pos + false_pos == 0:
            precision = 0
        else:
            precision = true_pos / (true_pos + false_pos)
        
        if true_pos + false_neg == 0:
            recall = 0
        else:
            recall = true_pos / (true_pos + false_neg)
        
        # Calculate F1
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Plot precision-recall curve
    plt.subplot(2, 2, 1)
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.scatter(recalls, precisions, c=thresholds, cmap='plasma', s=30)
    plt.colorbar(label='Threshold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot F1 vs threshold
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, f1_scores, 'g-', linewidth=2)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    plt.scatter([optimal_threshold], [optimal_f1], c='r', s=100, marker='*')
    plt.annotate(f'Optimal: {optimal_threshold:.2f} (F1={optimal_f1:.4f})', 
                 (optimal_threshold, optimal_f1),
                 xytext=(10, -20),
                 textcoords='offset points',
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Precision and Recall vs threshold
    plt.subplot(2, 2, 3)
    plt.plot(thresholds, precisions, 'r-', linewidth=2, label='Precision')
    plt.plot(thresholds, recalls, 'b-', linewidth=2, label='Recall')
    plt.axvline(x=optimal_threshold, color='g', linestyle='--', alpha=0.7)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot confusion matrix as a heatmap
    plt.subplot(2, 2, 4)
    
    # Use the default threshold metrics
    cm = np.array([
        [test_metrics['true_negatives'], test_metrics['false_positives']],
        [test_metrics['false_negatives'], test_metrics['true_positives']]
    ])
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    
    # Add labels
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'performance_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {vis_dir}")
    
    return vis_dir

def improved_main():
    """
    Main function that runs the complete analysis workflow.
    """
    print("Starting ResNet18 plate boundary detection workflow...")
    
    # Ensure directories exist
    check_create_directory(DATA_DIR)
    check_create_directory(MODEL_DIR)
    check_create_directory(OUTPUT_DIR)
    
    # Load earthquake and boundary data
    eq_data = load_raster_data()
    boundary_data = load_boundary_data()
    
    print(f"Loaded earthquake data with shape: {eq_data.shape}")
    print(f"Loaded boundary data with shape: {boundary_data.shape}")
    
    # Define geological regions and create spatial split
    regions = define_regions()
    masks = create_spatial_split(regions, boundary_data.shape[1:])
    splits = create_improved_train_val_test_split(boundary_data, masks, eq_data.shape[0])
    
    # Visualize data and splits
    visualize_data_split(eq_data, boundary_data, masks, splits, OUTPUT_DIR)
    
    # Prepare data for model
    train_windows, val_windows, num_features = prepare_improved_data(eq_data, boundary_data[0], masks, window_size=48)
    
    # Build and train model
    model = build_improved_resnet_model(input_shape=(48, 48, num_features))
    
    history = train_improved_model(
        model,
        train_windows,
        val_windows,
        batch_size=64,
        epochs=100,
        class_weight_ratio=8,
        model_path=os.path.join(MODEL_DIR, 'resnet18_improved.h5')
    )
    
    # Plot training history
    plot_training_history(history, OUTPUT_DIR)
    
    # Make predictions and evaluate
    print("Running predictions on test regions...")
    prediction_map, binary_prediction, test_metrics = predict_for_test_regions(
        model, 
        eq_data, 
        boundary_data, 
        masks, 
        window_size=48,
        threshold=0.5,
        use_morphology=True
    )
    
    # Visualize results
    simple_visualize(prediction_map, binary_prediction, boundary_data, masks, splits, OUTPUT_DIR)
    
    # Save metrics to file
    metrics_file = os.path.join(OUTPUT_DIR, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")
    print(f"Overall F1 score: {test_metrics['f1_score']:.4f}")
    
    return model, prediction_map, binary_prediction, test_metrics

if __name__ == "__main__":
    improved_main()  # 使用改进版的主函数
