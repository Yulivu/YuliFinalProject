"""
UNet-based model for tectonic plate boundary detection
- Implements UNet architecture for improved segmentation performance
- Uses the same data loading and processing methods from v2c2.py
- Enhanced visualization and evaluation
"""

import os
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

# Set paths
DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2processed"
RAW_DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2raw"
MODEL_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\src\model"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(r"C:\Users\debuf\Desktop\YuliFinalProject\result\unet", f"Unet_{timestamp}")

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

def global_pad(data, padding_size):
    if padding_size <= 0:
        return data
    channels, height, width = data.shape
    padded_data = np.zeros((channels, height + 2 * padding_size, width + 2 * padding_size), dtype=data.dtype)
    padded_data[:, padding_size:padding_size + height, padding_size:padding_size + width] = data
    padded_data[:, padding_size:padding_size + height, :padding_size] = data[:, :, -padding_size:]
    padded_data[:, padding_size:padding_size + height, padding_size + width:] = data[:, :, :padding_size]
    padded_data[:, :padding_size, padding_size:padding_size + width] = np.flip(data[:, :padding_size, :], axis=1)
    padded_data[:, padding_size + height:, padding_size:padding_size + width] = np.flip(data[:, -padding_size:, :], axis=1)
    padded_data[:, :padding_size, :padding_size] = np.flip(data[:, :padding_size, -padding_size:], axis=1)
    padded_data[:, :padding_size, padding_size + width:] = np.flip(data[:, :padding_size, :padding_size], axis=1)
    padded_data[:, padding_size + height:, :padding_size] = np.flip(data[:, -padding_size:, -padding_size:], axis=1)
    padded_data[:, padding_size + height:, padding_size + width:] = np.flip(data[:, -padding_size:, :padding_size], axis=1)
    return padded_data

def load_raster_data():
    """Load raster data, returning earthquake features and plate boundary rasters"""
    print("Loading raster data...")

    # Load earthquake features raster
    earthquake_raster_path = os.path.join(DATA_DIR, "earthquake_features_raster.tif")
    with rasterio.open(earthquake_raster_path) as eq_src:
        eq_data = eq_src.read()
        eq_meta = eq_src.meta
        eq_transform = eq_src.transform
        eq_crs = eq_src.crs

    # Load plate boundary raster
    boundaries_raster_path = os.path.join(DATA_DIR, "plate_boundaries_raster.tif")
    with rasterio.open(boundaries_raster_path) as bound_src:
        boundary_data = bound_src.read()

    # Adjust sizes to ensure both rasters have the same dimensions
    min_height = min(eq_data.shape[1], boundary_data.shape[1])
    min_width = min(eq_data.shape[2], boundary_data.shape[2])

    eq_data = eq_data[:, :min_height, :min_width]
    boundary_data = boundary_data[:, :min_height, :min_width]

    # Apply transformation: rotate 180° clockwise and flip horizontally
    transformed_eq_data = np.zeros_like(eq_data)
    transformed_boundary_data = np.zeros_like(boundary_data)
    
    for i in range(eq_data.shape[0]):
        transformed_eq_data[i] = np.fliplr(np.rot90(eq_data[i], k=2))
    
    for i in range(boundary_data.shape[0]):
        transformed_boundary_data[i] = np.fliplr(np.rot90(boundary_data[i], k=2))

    print(f"Earthquake features raster shape: {transformed_eq_data.shape}, Plate boundary raster shape: {transformed_boundary_data.shape}")

    return transformed_eq_data, transformed_boundary_data, eq_meta, eq_transform, eq_crs

def define_geological_regions(eq_data, boundary_data, n_regions_h=4, n_regions_w=4, random_state=42):
    """Define region-based splits based on geological features"""
    print(f"Defining {n_regions_h}x{n_regions_w} geological regions...")

    # Get data shape
    num_bands, height, width = eq_data.shape
    np.random.seed(random_state)

    # Create region mask
    region_mask = np.zeros((height, width), dtype=np.int32)
    region_stats = {}
    region_height = height // n_regions_h
    region_width = width // n_regions_w
    region_id = 0

    for i in range(n_regions_h):
        for j in range(n_regions_w):
            # Define region boundaries
            h_start = i * region_height
            h_end = (i + 1) * region_height if i < n_regions_h - 1 else height
            w_start = j * region_width
            w_end = (j + 1) * region_width if j < n_regions_w - 1 else width

            # Mark region
            region_mask[h_start:h_end, w_start:w_end] = region_id

            # Calculate region statistics
            region_pixels = (h_end - h_start) * (w_end - w_start)
            boundary_pixels = np.sum(boundary_data[0, h_start:h_end, w_start:w_end] > 0)
            boundary_percentage = boundary_pixels / region_pixels * 100

            # Calculate earthquake feature statistics
            earthquake_stats = {}
            for b in range(num_bands):
                band_data = eq_data[b, h_start:h_end, w_start:w_end]
                earthquake_stats[f'band_{b}_mean'] = np.mean(band_data)
                earthquake_stats[f'band_{b}_std'] = np.std(band_data)

            # Store region statistics
            region_stats[region_id] = {
                'region_id': region_id,
                'row': i, 'col': j,
                'h_start': h_start, 'h_end': h_end,
                'w_start': w_start, 'w_end': w_end,
                'total_pixels': region_pixels,
                'boundary_pixels': boundary_pixels,
                'boundary_percentage': boundary_percentage,
                'earthquake_stats': earthquake_stats
            }

            region_id += 1

    return region_mask, region_stats

def create_improved_train_test_split(region_mask, region_stats, test_size=0.25, val_size=0.15,
                                     min_boundary_percentage=0.5, random_state=42):
    """Create improved train-validation-test split ensuring test regions contain sufficient boundary information"""
    print("Creating improved train-validation-test split...")
    np.random.seed(random_state)

    # Get all region IDs
    all_regions = list(region_stats.keys())
    total_regions = len(all_regions)

    # Calculate test and validation region counts
    n_test_regions = max(1, int(total_regions * test_size))
    n_val_regions = max(1, int(total_regions * val_size))

    # Sort regions by boundary percentage
    regions_by_boundary = sorted(
        all_regions,
        key=lambda r: region_stats[r]['boundary_percentage'],
        reverse=True
    )

    # Ensure test regions include enough boundaries
    high_boundary_regions = [r for r in regions_by_boundary
                             if region_stats[r]['boundary_percentage'] >= min_boundary_percentage]

    if len(high_boundary_regions) >= n_test_regions:
        # If enough high-boundary regions exist, randomly select from them
        test_regions = np.random.choice(high_boundary_regions, size=n_test_regions, replace=False)
    else:
        # Otherwise, use all high-boundary regions and randomly select from the rest
        test_regions = high_boundary_regions.copy()
        remaining_needed = n_test_regions - len(test_regions)
        remaining_regions = [r for r in all_regions if r not in test_regions]
        if remaining_needed > 0 and remaining_regions:
            additional_regions = np.random.choice(remaining_regions,
                                                  size=min(remaining_needed, len(remaining_regions)),
                                                  replace=False)
            test_regions = np.concatenate([test_regions, additional_regions])

    # Select validation set from remaining regions
    remaining_regions = [r for r in all_regions if r not in test_regions]
    val_regions = np.random.choice(
        remaining_regions,
        size=min(n_val_regions, len(remaining_regions)),
        replace=False
    )

    # Remaining regions form the training set
    train_regions = [r for r in all_regions if r not in test_regions and r not in val_regions]

    # Create masks
    height, width = region_mask.shape
    train_mask = np.zeros((height, width), dtype=bool)
    val_mask = np.zeros((height, width), dtype=bool)
    test_mask = np.zeros((height, width), dtype=bool)

    for r in train_regions:
        train_mask[region_mask == r] = True
    for r in val_regions:
        val_mask[region_mask == r] = True
    for r in test_regions:
        test_mask[region_mask == r] = True

    # Save split information
    splits = {
        'train_regions': train_regions,
        'val_regions': val_regions,
        'test_regions': test_regions,
        'region_stats': region_stats
    }

    # Create mask dictionary
    masks = {'train': train_mask, 'val': val_mask, 'test': test_mask, 'region': region_mask}

    print(f"Train/validation/test regions: {len(train_regions)}/{len(val_regions)}/{len(test_regions)} regions")

    return masks, splits

def prepare_windowed_data(eq_data, boundary_data, masks, window_size=7, patch_mode=True):
    """Prepare windowed data for model training using patches"""
    print(f"Preparing windowed data with size {window_size}x{window_size}...")

    num_bands, height, width = eq_data.shape
    half_window = window_size // 2

    # Determine sample counts for each set
    train_count = np.sum(masks['train'])
    val_count = np.sum(masks['val'])
    test_count = np.sum(masks['test'])
    print(f"Train/validation/test samples: {train_count}/{val_count}/{test_count}")

    # Get train/validation/test pixel indices
    train_indices = np.where(masks['train'])
    val_indices = np.where(masks['val'])
    test_indices = np.where(masks['test'])

    # Create index mappings for later spatial position recovery
    train_index_map = {(y, x): i for i, (y, x) in enumerate(zip(train_indices[0], train_indices[1]))}
    val_index_map = {(y, x): i for i, (y, x) in enumerate(zip(val_indices[0], val_indices[1]))}
    test_index_map = {(y, x): i for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1]))}

    # Use custom global padding to extend data, correctly handling lat/lon boundaries
    eq_data_padded = global_pad(eq_data, half_window)

    if patch_mode:
        # In patch mode, we directly create patches for UNet input
        # Initialize arrays for patches
        X_train = np.zeros((train_count, window_size, window_size, num_bands))
        X_val = np.zeros((val_count, window_size, window_size, num_bands))
        X_test = np.zeros((test_count, window_size, window_size, num_bands))
        
        # Initialize arrays for target masks
        y_train = np.zeros((train_count, window_size, window_size, 1))
        y_val = np.zeros((val_count, window_size, window_size, 1))
        y_test = np.zeros((test_count, window_size, window_size, 1))
        
        # Pad boundary data
        boundary_data_padded = global_pad(boundary_data, half_window)
        
        # Processing function for patch creation
        def process_patches(indices, X_patches, y_patches):
            count = len(indices[0])
            for i, (y, x) in enumerate(zip(indices[0], indices[1])):
                if i % 50000 == 0 and i > 0:
                    print(f"  Processed {i}/{count} samples")
                    
                padded_y, padded_x = y + half_window, x + half_window
                
                # Extract feature patch from all bands
                for b in range(num_bands):
                    X_patches[i, :, :, b] = eq_data_padded[b,
                                           padded_y - half_window:padded_y + half_window + 1,
                                           padded_x - half_window:padded_x + half_window + 1]
                
                # Extract target patch
                y_patches[i, :, :, 0] = boundary_data_padded[0,
                                        padded_y - half_window:padded_y + half_window + 1,
                                        padded_x - half_window:padded_x + half_window + 1]
        
        # Fill train, validation, and test patches
        print("Preparing training samples...")
        process_patches(train_indices, X_train, y_train)
        
        print("Preparing validation samples...")
        process_patches(val_indices, X_val, y_val)
        
        print("Preparing test samples...")
        process_patches(test_indices, X_test, y_test)
        
        # Binarize targets
        y_train = (y_train > 0).astype(np.float32)
        y_val = (y_val > 0).astype(np.float32)
        y_test = (y_test > 0).astype(np.float32)
        
    else:
        # In non-patch mode (used by the original CNN), we extract windows centered on each pixel
        # Initialize windowed data
        X_train = np.zeros((train_count, window_size, window_size, num_bands))
        X_val = np.zeros((val_count, window_size, window_size, num_bands))
        X_test = np.zeros((test_count, window_size, window_size, num_bands))

        # Processing function for windows
        def process_windows(indices, windowed_data):
            count = len(indices[0])
            for i, (y, x) in enumerate(zip(indices[0], indices[1])):
                if i % 50000 == 0 and i > 0:
                    print(f"  Processed {i}/{count} samples")
                    
                padded_y, padded_x = y + half_window, x + half_window
                for b in range(num_bands):
                    windowed_data[i, :, :, b] = eq_data_padded[b,
                                               padded_y - half_window:padded_y + half_window + 1,
                                               padded_x - half_window:padded_x + half_window + 1]

        # Fill train, validation, and test windows
        print("Preparing training samples...")
        process_windows(train_indices, X_train)
        
        print("Preparing validation samples...")
        process_windows(val_indices, X_val)
        
        print("Preparing test samples...")
        process_windows(test_indices, X_test)

        # Get target values
        y_train = boundary_data[0][train_indices]
        y_val = boundary_data[0][val_indices]
        y_test = boundary_data[0][test_indices]

        # Binarize target values
        y_train = (y_train > 0).astype(np.float32)
        y_val = (y_val > 0).astype(np.float32)
        y_test = (y_test > 0).astype(np.float32)

    # Calculate class weight to handle imbalance
    pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1.0
    print(f"Positive sample weight: {pos_weight:.2f} (for handling class imbalance)")

    # Create indices to spatial position mapping dictionary
    indices_mapping = {
        'train': {'indices': train_indices, 'index_map': train_index_map},
        'val': {'indices': val_indices, 'index_map': val_index_map},
        'test': {'indices': test_indices, 'index_map': test_index_map}
    }

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            pos_weight, indices_mapping)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)

def build_unet_model(input_shape, dropout_rate=0.5, l2_reg=1e-4, base_filters=32):
    reg = tf.keras.regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape)
    # Encoder path
    conv1 = layers.Conv2D(base_filters, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(base_filters, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = layers.Dropout(dropout_rate)(pool1)
    # Level 2
    conv2 = layers.Conv2D(base_filters*2, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(base_filters*2, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = layers.Dropout(dropout_rate)(pool2)
    # Bottom level
    conv3 = layers.Conv2D(base_filters*4, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(base_filters*4, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Dropout(dropout_rate)(conv3)
    # Decoder path
    up2 = layers.Conv2DTranspose(base_filters*2, (2, 2), strides=(2, 2), padding='same')(conv3)
    merge2 = layers.concatenate([conv2, up2], axis=3)
    up_conv2 = layers.Conv2D(base_filters*2, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(merge2)
    up_conv2 = layers.BatchNormalization()(up_conv2)
    up_conv2 = layers.Conv2D(base_filters*2, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(up_conv2)
    up_conv2 = layers.BatchNormalization()(up_conv2)
    up_conv2 = layers.Dropout(dropout_rate/2)(up_conv2)
    up1 = layers.UpSampling2D(size=(2, 2))(up_conv2)
    target_size = conv1.shape[1:3]
    up1 = layers.Resizing(target_size[0], target_size[1])(up1)
    up1 = layers.Conv2D(base_filters, (3, 3), activation='relu', padding='same')(up1)
    merge1 = layers.concatenate([conv1, up1], axis=3)
    up_conv1 = layers.Conv2D(base_filters, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(merge1)
    up_conv1 = layers.BatchNormalization()(up_conv1)
    up_conv1 = layers.Conv2D(base_filters, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(up_conv1)
    up_conv1 = layers.BatchNormalization()(up_conv1)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(up_conv1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss=dice_loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    print(f"UNet base_filters: {base_filters}, Dropout: {dropout_rate}, L2: {l2_reg}")
    print(f"Model parameters: {model.count_params()}")
    return model

def main():
    print("Starting UNet-based plate boundary prediction (training only)...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eq_data, boundary_data, eq_meta, eq_transform, eq_crs = load_raster_data()
    region_mask, region_stats = define_geological_regions(eq_data, boundary_data)
    masks, splits = create_improved_train_test_split(region_mask, region_stats)
    window_size = 33
    X_train, X_val, X_test, y_train, y_val, y_test, pos_weight, indices_mapping = prepare_windowed_data(
        eq_data, boundary_data, masks, window_size, patch_mode=True)
    # 使用 base_filters=32
    base_filters = 32
    print(f"\n--- Training UNet with base_filters={base_filters} ---")
    input_shape = X_train.shape[1:]
    model = build_unet_model(input_shape, dropout_rate=0.5, l2_reg=1e-4, base_filters=base_filters)
    model.summary()
    model_id = f"unet_{base_filters}_{timestamp}"
    MODELS_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\src\model\models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_id}.h5")
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15),
        callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
    ]
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1, 1, 1)
        y_val = y_val.reshape(-1, 1, 1, 1)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        class_weight=None,
        verbose=1
    )
    # 保存训练历史到CSV
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(OUTPUT_DIR, f'{model_id}_training_history.csv')
    history_df.to_csv(history_csv_path, index=False)
    # 训练过程可视化
    metrics_to_plot = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'precision', 'val_precision', 'recall', 'val_recall', 'auc', 'val_auc', 'learning_rate']
    plt.figure(figsize=(16, 10))
    for i, metric in enumerate(metrics_to_plot):
        if metric in history_df.columns:
            plt.plot(history_df[metric], label=metric)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_id}_training_history.png'), dpi=200)
    plt.close()
    # 生成评估报告
    best_epoch = int(np.argmin(history.history['val_loss']))
    report_lines = [
        f"Test Region Evaluation Report",
        f"-----------------------------",
        f"Best Epoch: {best_epoch}",
        f"Train Loss: {history.history['loss'][best_epoch]:.4f}",
        f"Val Loss: {history.history['val_loss'][best_epoch]:.4f}",
        f"Train Accuracy: {history.history['accuracy'][best_epoch]:.4f}",
        f"Val Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}",
        f"Train Precision: {history.history['precision'][best_epoch]:.4f}",
        f"Val Precision: {history.history['val_precision'][best_epoch]:.4f}",
        f"Train Recall: {history.history['recall'][best_epoch]:.4f}",
        f"Val Recall: {history.history['val_recall'][best_epoch]:.4f}",
        f"Train AUC: {history.history['auc'][best_epoch]:.4f}",
        f"Val AUC: {history.history['val_auc'][best_epoch]:.4f}",
        f"Model saved to: {model_path}"
    ]
    report_path = os.path.join(OUTPUT_DIR, f'Test_Region_Evaluation_Report_{base_filters}.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Training completed. Model saved to: {model_path}")
    print(f"Training history saved to: {history_csv_path}")
    print(f"Evaluation report saved to: {report_path}")
    return None

if __name__ == "__main__":
    main()