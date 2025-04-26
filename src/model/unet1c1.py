"""
UNet-based model for tectonic plate boundary detection
- Implements UNet architecture for improved segmentation performance
- Uses the same data loading and processing methods from v2c2.py
- Enhanced visualization and evaluation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from skimage import measure
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
import time
import math
from datetime import datetime

# Set paths - adjust according to your environment
DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2processed"
RAW_DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2raw"
MODEL_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\src\model"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(r"C:\Users\debuf\Desktop\YuliFinalProject\result\unet", f"Unet_{timestamp}")
PLATE_BOUNDARIES_SHP = os.path.join(RAW_DATA_DIR, "plate", "plate_boundaries.shp")

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU setup error: {e}")


def global_pad(data, padding_size):
    """Implement custom padding for global geographic data, handling latitude/longitude boundaries"""
    if padding_size <= 0:
        return data
        
    channels, height, width = data.shape
    padded_data = np.zeros((channels, height + 2 * padding_size, width + 2 * padding_size), dtype=data.dtype)

    # Fill central region
    padded_data[:, padding_size:padding_size + height, padding_size:padding_size + width] = data

    # Handle longitude boundaries (east-west) - wrap padding
    padded_data[:, padding_size:padding_size + height, :padding_size] = data[:, :, -padding_size:]  # Left boundary
    padded_data[:, padding_size:padding_size + height, padding_size + width:] = data[:, :, :padding_size]  # Right boundary

    # Handle latitude boundaries (north-south) - reflect padding
    padded_data[:, :padding_size, padding_size:padding_size + width] = np.flip(data[:, :padding_size, :], axis=1)  # Top boundary
    padded_data[:, padding_size + height:, padding_size:padding_size + width] = np.flip(data[:, -padding_size:, :], axis=1)  # Bottom boundary

    # Handle corners
    padded_data[:, :padding_size, :padding_size] = np.flip(data[:, :padding_size, -padding_size:], axis=1)  # Top-left
    padded_data[:, :padding_size, padding_size + width:] = np.flip(data[:, :padding_size, :padding_size], axis=1)  # Top-right
    padded_data[:, padding_size + height:, :padding_size] = np.flip(data[:, -padding_size:, -padding_size:], axis=1)  # Bottom-left
    padded_data[:, padding_size + height:, padding_size + width:] = np.flip(data[:, -padding_size:, :padding_size], axis=1)  # Bottom-right

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


def visualize_data_split(masks, splits, output_path):
    """Visualize data split, showing train, validation, and test regions"""
    print("Visualizing data split...")

    region_mask = masks['region']
    train_regions = splits['train_regions']
    val_regions = splits['val_regions']
    test_regions = splits['test_regions']
    region_stats = splits['region_stats']

    # Create region type mask (0=train, 1=validation, 2=test)
    split_type_mask = np.zeros_like(region_mask)
    for r in val_regions:
        split_type_mask[region_mask == r] = 1
    for r in test_regions:
        split_type_mask[region_mask == r] = 2

    # Create custom colormap
    colors = ['#4363d8', '#42d4f4', '#f58231']  # blue=train, cyan=validation, orange=test
    cmap = LinearSegmentedColormap.from_list('split_cmap', colors, N=3)

    # Rotate 180° and flip horizontally for consistency with other visualizations
    split_type_mask_transformed = np.rot90(split_type_mask, k=2)  # Rotate 180°
    split_type_mask_transformed = np.fliplr(split_type_mask_transformed)  # Flip horizontally

    # Create image
    plt.figure(figsize=(12, 9))
    plt.imshow(split_type_mask_transformed, cmap=cmap, interpolation='nearest')

    # Add region boundaries and labels
    for region_id, stats in region_stats.items():
        y_start, y_end = stats['h_start'], stats['h_end']
        x_start, x_end = stats['w_start'], stats['w_end']
        
        # Transform coordinates
        height, width = split_type_mask.shape
        y_start_transformed = height - y_end
        y_end_transformed = height - y_start
        x_start_transformed = width - x_end
        x_end_transformed = width - x_start
        
        # Draw boundary
        plt.plot([x_start_transformed, x_end_transformed, x_end_transformed, x_start_transformed, x_start_transformed],
                 [y_start_transformed, y_start_transformed, y_end_transformed, y_end_transformed, y_start_transformed],
                 'k-', linewidth=0.8, alpha=0.6)
        
        # Determine region type and text color
        if region_id in test_regions:
            region_type, text_color = "Test", 'black'
        elif region_id in val_regions:
            region_type, text_color = "Val", 'black'
        else:
            region_type, text_color = "Train", 'white'
            
        # Add label - calculate transformed center
        center_y_transformed = (y_start_transformed + y_end_transformed) // 2
        center_x_transformed = (x_start_transformed + x_end_transformed) // 2
        plt.text(center_x_transformed, center_y_transformed, f"R{region_id}\n{region_type}\n{stats['boundary_percentage']:.1f}%",
                 color=text_color, ha='center', va='center', fontweight='bold', fontsize=9)

    # Add legend
    legend_elements = [
        Patch(facecolor=colors[0], label='Training area'),
        Patch(facecolor=colors[1], label='Validation area'),
        Patch(facecolor=colors[2], label='Test area')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title('Data segmentation - regional distribution')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Data split visualization saved to {output_path}")


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


def build_unet_model(input_shape, dropout_rate=0.3):
    """Build UNet model for image segmentation"""
    print("Building UNet model...")
    
    # Set L2 regularization coefficient
    l2_reg = 1e-4
    reg = tf.keras.regularizers.l2(l2_reg)

    # UNet architecture with skip connections
    inputs = layers.Input(shape=input_shape)
    
    # Encoder path (contracting)
    # Level 1
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = layers.Dropout(dropout_rate)(pool1)
    
    # Level 2
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = layers.Dropout(dropout_rate)(pool2)
    
    # Bottom level
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Dropout(dropout_rate)(conv3)
    
    # Decoder path (expanding)
    # Level 2 up
    up2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
    merge2 = layers.concatenate([conv2, up2], axis=3)
    up_conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(merge2)
    up_conv2 = layers.BatchNormalization()(up_conv2)
    up_conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(up_conv2)
    up_conv2 = layers.BatchNormalization()(up_conv2)
    up_conv2 = layers.Dropout(dropout_rate / 2)(up_conv2)
    
    # Print shapes for debugging
    print(f"Conv1 shape: {conv1.shape}, Up_conv2 shape: {up_conv2.shape}")
    
    # Level 1 up - with ResizeLayer to match dimensions exactly
    up1 = layers.UpSampling2D(size=(2, 2))(up_conv2)
    print(f"Up1 after upsampling shape: {up1.shape}")
    
    # Use custom Resizing to match exactly
    target_size = conv1.shape[1:3]
    up1 = layers.Resizing(target_size[0], target_size[1])(up1)
    up1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    
    print(f"After adjustments - Conv1 shape: {conv1.shape}, Up1 shape: {up1.shape}")
    
    merge1 = layers.concatenate([conv1, up1], axis=3)
    up_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(merge1)
    up_conv1 = layers.BatchNormalization()(up_conv1)
    up_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(up_conv1)
    up_conv1 = layers.BatchNormalization()(up_conv1)
    
    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(up_conv1)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    return model


def weighted_binary_crossentropy(pos_weight):
    """Create weighted binary cross-entropy loss function for handling class imbalance"""
    def loss(y_true, y_pred):
        # Avoid log(0) errors
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # Weighted binary cross-entropy
        loss_pos = -pos_weight * y_true * tf.math.log(y_pred)
        loss_neg = -(1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss_pos + loss_neg)
    return loss


def improved_postprocessing(prediction_map, threshold=0.5):
    """Enhanced post-processing strategy for better boundary continuity"""
    # Basic binarization
    binary_prediction = (prediction_map >= threshold).astype(np.uint8)
    
    # First remove small noise points
    binary_prediction = ndimage.binary_opening(binary_prediction, structure=np.ones((3, 3)))
    
    # Then connect close areas
    binary_prediction = ndimage.binary_closing(binary_prediction, structure=np.ones((5, 5)))
    
    # Remove small regions
    labeled_array, num_features = ndimage.label(binary_prediction)
    component_sizes = np.bincount(labeled_array.ravel())
    small_size = 10  # Adjust according to data
    too_small = component_sizes < small_size
    too_small[0] = False  # Keep background
    binary_prediction = ~np.isin(labeled_array, np.where(too_small))
    
    # Another closing operation to fill gaps in boundaries
    binary_prediction = ndimage.binary_closing(binary_prediction, structure=np.ones((3, 3)))
    
    # Finally thin the boundaries for better precision
    binary_prediction = ndimage.binary_erosion(binary_prediction, structure=np.ones((2, 2)))
    
    return binary_prediction.astype(np.uint8)


def train_and_evaluate(X_train, X_val, y_train, y_val, pos_weight=1.0,
                       epochs=50, batch_size=32, patience=10, patch_mode=True):
    """Train and evaluate the model, using class weights to handle imbalance"""
    print("Starting model training...")
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    # Build UNet model
    input_shape = X_train.shape[1:]
    model = build_unet_model(input_shape)
    model.summary()

    # Use class weights
    class_weights = {0: 1.0, 1: pos_weight}
    print(f"Using class weights: {class_weights}")

    # Generate timestamp as model ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"unet_plate_boundary_{timestamp}"
    model_path = os.path.join(MODEL_DIR, f"{model_id}.h5")

    # Set up callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=patience),
        callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience // 2, 
                                    min_lr=1e-6, verbose=1)
    ]

    # Reshape targets for UNet if in patch mode
    if patch_mode:
        # For UNet, targets should have shape (samples, height, width, 1)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1, 1, 1)
            y_val = y_val.reshape(-1, 1, 1, 1)
    else:
        # For the original approach, targets should have shape (samples,)
        if len(y_train.shape) > 1:
            y_train = y_train.reshape(-1)
            y_val = y_val.reshape(-1)

    # Train model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        class_weight=None if patch_mode else class_weights,  # Only use class weights in non-patch mode
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed, time: {training_time:.2f} seconds")

    # Load best model
    model = tf.keras.models.load_model(model_path)

    # Evaluate model
    if patch_mode:
        # In patch mode, we evaluate using the center pixel of each patch
        y_val_center = y_val[:, y_val.shape[1]//2, y_val.shape[2]//2, 0]
        val_pred = model.predict(X_val)
        val_pred_center = val_pred[:, val_pred.shape[1]//2, val_pred.shape[2]//2, 0]
        
        # Calculate metrics on center pixels
        val_pred_binary = (val_pred_center >= 0.5).astype(np.float32)
        val_accuracy = np.mean(val_pred_binary == y_val_center)
        val_precision = np.sum((val_pred_binary == 1) & (y_val_center == 1)) / (np.sum(val_pred_binary == 1) + 1e-7)
        val_recall = np.sum((val_pred_binary == 1) & (y_val_center == 1)) / (np.sum(y_val_center == 1) + 1e-7)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)
        
        print(f"Validation set metrics: Accuracy={val_accuracy:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}, F1={val_f1:.4f}")
    else:
        val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(X_val, y_val, verbose=0)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        print(f"Validation set metrics: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}, F1={val_f1:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Plot loss curve
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss', fontsize=12)
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy', fontsize=12)
    plt.legend()

    # Plot precision and recall
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Prec')
    plt.plot(history.history['val_precision'], label='Val Prec')
    plt.plot(history.history['recall'], label='Rec')
    plt.plot(history.history['val_recall'], label='Val Rec')
    plt.title('Precision/Recall', fontsize=12)
    plt.legend()

    plt.tight_layout()
    history_plot_path = os.path.join(OUTPUT_DIR, f'{model_id}_training_history.png')
    plt.savefig(history_plot_path, dpi=300)
    plt.close()

    # Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, f'{model_id}_training_history.csv'), index=False)

    return model, history, model_id


def predict_for_test_regions(model, eq_data, masks, splits, window_size=7, batch_size=32, patch_mode=True):
    """Predict for test regions using the UNet model"""
    print("Starting prediction for test regions...")

    num_bands, height, width = eq_data.shape
    half_window = window_size // 2
    test_mask = masks['test']

    # Only predict for test region pixels
    test_indices = np.where(test_mask)
    num_test_pixels = len(test_indices[0])
    print(f"Test region: {num_test_pixels} pixels")

    # Use custom global padding to handle lat/lon boundaries
    eq_data_padded = global_pad(eq_data, half_window)

    # Initialize prediction map
    prediction_map = np.zeros((height, width), dtype=np.float32)

    if patch_mode:
        # For UNet, we need to predict each patch and extract the center pixel
        # Since UNet works with patches, we need to be careful about memory usage
        batch_size = min(batch_size, 128)  # Limit batch size to avoid memory issues
        
        # Process test pixels in batches
        for batch_start in range(0, num_test_pixels, batch_size):
            batch_end = min(batch_start + batch_size, num_test_pixels)
            batch_indices = (test_indices[0][batch_start:batch_end], test_indices[1][batch_start:batch_end])
            batch_size_actual = batch_end - batch_start
            
            # Create batch of patches
            X_batch = np.zeros((batch_size_actual, window_size, window_size, num_bands))
            
            # Collect patch data
            for i, (y, x) in enumerate(zip(batch_indices[0], batch_indices[1])):
                padded_y, padded_x = y + half_window, x + half_window
                for b in range(num_bands):
                    X_batch[i, :, :, b] = eq_data_padded[b,
                                         padded_y - half_window:padded_y + half_window + 1,
                                         padded_x - half_window:padded_x + half_window + 1]
            
            # Predict batch
            pred_batch = model.predict(X_batch, verbose=0)
            
            # For UNet, extract predictions for each center pixel
            for i, (y, x) in enumerate(zip(batch_indices[0], batch_indices[1])):
                prediction_map[y, x] = pred_batch[i, half_window, half_window, 0]
            
            if batch_start % (20 * batch_size) == 0:
                print(f"  Processed {batch_start}/{num_test_pixels} test pixels")
    else:
        # For the original approach, create windows for each test pixel
        X_test_windowed = np.zeros((num_test_pixels, window_size, window_size, num_bands))

        # Collect test region pixels' windowed data
        for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1])):
            if i % 50000 == 0 and i > 0:
                print(f"  Processed {i}/{num_test_pixels} test pixels")

            padded_y, padded_x = y + half_window, x + half_window
            for b in range(num_bands):
                X_test_windowed[i, :, :, b] = eq_data_padded[b,
                                             padded_y - half_window:padded_y + half_window + 1,
                                             padded_x - half_window:padded_x + half_window + 1]

        # Use batch processing for prediction
        print("Performing batch prediction for test region...")
        predictions = []
        total_batches = math.ceil(num_test_pixels / batch_size)
        
        for i in range(0, num_test_pixels, batch_size):
            batch_end = min(i + batch_size, num_test_pixels)
            batch_predictions = model.predict(X_test_windowed[i:batch_end], verbose=0)
            predictions.append(batch_predictions)
            if (i // batch_size) % 20 == 0:
                print(f"  Completed {i // batch_size}/{total_batches} batches")

        # Merge all batch predictions and fill prediction map
        all_predictions = np.vstack(predictions)
        for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1])):
            prediction_map[y, x] = all_predictions[i, 0]

    print("Test region prediction completed")
    return prediction_map


def visualize_test_predictions(prediction_map, boundary_data, masks, splits, eq_meta,
                               threshold=0.5, output_path_prefix=None):
    """Visualize test region prediction results"""
    print("Visualizing test region prediction results...")

    if output_path_prefix is None:
        output_path_prefix = os.path.join(OUTPUT_DIR, "test_prediction")

    # Get test mask and region information
    test_mask = masks['test']
    test_regions = splits['test_regions']
    region_mask = masks['region']
    region_stats = splits['region_stats']

    # Apply improved post-processing to prediction map for better boundary continuity
    binary_prediction = improved_postprocessing(prediction_map, threshold)

    # Extract contours
    contours = measure.find_contours(prediction_map, threshold)

    # Calculate performance metrics for test region
    test_indices = np.where(test_mask)
    test_true = (boundary_data[0][test_indices] > 0).astype(np.int32)
    test_pred = binary_prediction[test_indices].astype(np.int32)  # Use post-processed prediction

    # Calculate precision, recall, and F1
    true_positives = np.sum((test_true == 1) & (test_pred == 1))
    false_positives = np.sum((test_true == 0) & (test_pred == 1))
    false_negatives = np.sum((test_true == 1) & (test_pred == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Create evaluation report for test region
    report = f"""Test Region Evaluation Report:
    -----------------
    Number of test regions: {len(test_regions)}
    Total test pixels: {np.sum(test_mask)}
    Boundary pixel ratio: {np.sum(test_true) / np.sum(test_mask) * 100:.2f}%
    
    Performance metrics:
    - Precision: {precision:.4f}
    - Recall: {recall:.4f}
    - F1 score: {f1:.4f}
    """

    # Save report to file
    with open(f"{output_path_prefix}_report.txt", 'w') as f:
        f.write(report)

    print(report)

    # Create custom colormaps
    boundary_cmap = LinearSegmentedColormap.from_list('boundary_cmap', ['white', 'crimson'], N=256)
    prediction_cmap = LinearSegmentedColormap.from_list('prediction_cmap', 
                                                        ['navy', 'deepskyblue', 'gold'], N=256)

    # Get shape dimensions
    height, width = test_mask.shape

    # Create overall test region prediction comparison
    plt.figure(figsize=(16, 12))
    
    # Only show test region
    test_prediction = np.ma.masked_where(~test_mask, prediction_map)
    test_original = np.ma.masked_where(~test_mask, boundary_data[0])
    
    # Create background - show test region as light gray
    background = np.zeros_like(test_mask, dtype=np.float32)
    background[test_mask] = 0.1
    plt.imshow(background, cmap='gray', alpha=0.3)

    # Show original boundary - red
    plt.imshow(test_original, cmap=boundary_cmap, alpha=0.7, vmin=0, vmax=1)

    # Plot predicted boundary contours - blue
    for contour in contours:
        # Check if contour is in test region
        contour_points = [(int(p[0]), int(p[1])) for p in contour 
                          if 0 <= int(p[0]) < test_mask.shape[0] and 0 <= int(p[1]) < test_mask.shape[1]]
        if any(test_mask[y, x] for y, x in contour_points):
            plt.plot(contour[:, 1], contour[:, 0], 'b-', linewidth=1)

    # Draw boundary for each test region
    for region_id in test_regions:
        stats = region_stats[region_id]
        y_start, y_end = stats['h_start'], stats['h_end']
        x_start, x_end = stats['w_start'], stats['w_end']
        
        plt.plot([x_start, x_end, x_end, x_start, x_start],
                 [y_start, y_start, y_end, y_end, y_start],
                 'y-', linewidth=2, alpha=0.7)

    plt.title(f'Test Region Evaluation - Enhanced Processing - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}', 
              fontsize=14)

    # Add legend
    legend_elements = [
        Patch(facecolor='crimson', alpha=0.7, label='Original Boundary'),
        Line2D([0], [0], color='blue', lw=2, label='Predicted Boundary'),
        Patch(facecolor='yellow', alpha=0.3, label='Test Region'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_overview.png", dpi=300)
    plt.close()

    # Create prediction heatmap
    plt.figure(figsize=(16, 12))
    plt.imshow(background, cmap='gray', alpha=0.3)
    plt.imshow(test_prediction, cmap=prediction_cmap, alpha=0.7, vmin=0, vmax=1)
    
    # Show original boundary contours
    original_contours = measure.find_contours(boundary_data[0], 0.5)
    for contour in original_contours:
        contour_points = [(int(p[0]), int(p[1])) for p in contour 
                         if 0 <= int(p[0]) < test_mask.shape[0] and 0 <= int(p[1]) < test_mask.shape[1]]
        if any(test_mask[y, x] for y, x in contour_points):
            plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5)
    
    plt.title('Test Region Prediction Heatmap - Enhanced Model', fontsize=14)
    plt.colorbar(label='Prediction Probability')
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_heatmap.png", dpi=300)
    plt.close()
    
    # Create individual visualizations for each test region
    print(f"Creating individual visualizations for {len(test_regions)} test regions...")
    for region_id in test_regions:
        stats = region_stats[region_id]
        y_start, y_end = stats['h_start'], stats['h_end']
        x_start, x_end = stats['w_start'], stats['w_end']
        
        # Create region mask
        region_mask_local = np.zeros_like(test_mask, dtype=bool)
        region_mask_local[y_start:y_end, x_start:x_end] = True
        region_test_mask = region_mask_local & test_mask
        
        if np.sum(region_test_mask) == 0:
            print(f"  Region {region_id} is not in test set, skipping visualization")
            continue
            
        # Calculate region performance metrics
        region_indices = np.where(region_test_mask)
        region_true = (boundary_data[0][region_indices] > 0).astype(np.int32)
        region_pred = binary_prediction[region_indices].astype(np.int32)  # Use post-processed prediction
        
        region_tp = np.sum((region_true == 1) & (region_pred == 1))
        region_fp = np.sum((region_true == 0) & (region_pred == 1))
        region_fn = np.sum((region_true == 1) & (region_pred == 0))
        
        region_precision = region_tp / (region_tp + region_fp) if (region_tp + region_fp) > 0 else 0
        region_recall = region_tp / (region_tp + region_fn) if (region_tp + region_fn) > 0 else 0
        region_f1 = 2 * region_precision * region_recall / (region_precision + region_recall) if (region_precision + region_recall) > 0 else 0
        
        # Create region visualization - original vs predicted boundaries
        plt.figure(figsize=(10, 8))
        
        # Crop region data
        region_data = boundary_data[0][y_start:y_end, x_start:x_end]
        region_pred_data = prediction_map[y_start:y_end, x_start:x_end]
        region_test_area = test_mask[y_start:y_end, x_start:x_end]
        
        # Create masked data
        region_original = np.ma.masked_where(~region_test_area, region_data)
        
        # Show region original boundary
        plt.imshow(region_original, cmap=boundary_cmap, alpha=0.7, vmin=0, vmax=1)
        
        # Get contours in region
        region_binary = binary_prediction[y_start:y_end, x_start:x_end]
        region_contours = measure.find_contours(region_pred_data, threshold)
        for contour in region_contours:
            y_contour = contour[:, 0]
            x_contour = contour[:, 1]
            contour_in_test = any(region_test_area[int(min(y, region_test_area.shape[0]-1)), 
                                               int(min(x, region_test_area.shape[1]-1))] 
                               for y, x in zip(y_contour, x_contour)
                               if 0 <= int(y) < region_test_area.shape[0] and 
                                  0 <= int(x) < region_test_area.shape[1])
            if contour_in_test:
                plt.plot(x_contour, y_contour, 'b-', linewidth=1.5)
                
        plt.title(f'Region {region_id} - Boundary: {stats["boundary_percentage"]:.1f}% - Enhanced Processing\n'
                 f'Precision: {region_precision:.3f}, Recall: {region_recall:.3f}, F1: {region_f1:.3f}')
        
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_region_{region_id}.png", dpi=300)
        plt.close()
        
        # Create region heatmap
        plt.figure(figsize=(10, 8))
        region_pred_masked = np.ma.masked_where(~region_test_area, region_pred_data)
        plt.imshow(region_pred_masked, cmap=prediction_cmap, vmin=0, vmax=1)
        
        # Show original boundary contours
        original_contours = measure.find_contours(region_data, 0.5)
        for contour in original_contours:
            contour_in_test = any(region_test_area[int(min(y, region_test_area.shape[0]-1)), 
                                                int(min(x, region_test_area.shape[1]-1))] 
                                for y, x in zip(contour[:, 0], contour[:, 1])
                                if 0 <= int(y) < region_test_area.shape[0] and 
                                   0 <= int(x) < region_test_area.shape[1])
            if contour_in_test:
                plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5)
                
        plt.colorbar(label='Prediction Probability')
        plt.title(f'Region {region_id} - Enhanced Model - Prediction Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_region_{region_id}_heatmap.png", dpi=300)
        plt.close()

    # Save prediction results as GeoTIFF
    pred_meta = eq_meta.copy()
    pred_meta.update({'count': 1, 'dtype': 'float32', 'nodata': None})
    with rasterio.open(f"{output_path_prefix}.tif", 'w', **pred_meta) as dst:
        dst.write(prediction_map.astype(np.float32), 1)

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'threshold': threshold, 'true_positives': true_positives,
        'false_positives': false_positives, 'false_negatives': false_negatives
    }


def main():
    """Main function"""
    print("Starting UNet-based plate boundary prediction...")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Load data
    eq_data, boundary_data, eq_meta, eq_transform, eq_crs = load_raster_data()

    # Define geological regions
    region_mask, region_stats = define_geological_regions(eq_data, boundary_data)

    # Split data
    masks, splits = create_improved_train_test_split(region_mask, region_stats)

    # Visualize data split
    visualize_data_split(masks, splits, os.path.join(run_output_dir, "data_split.png"))

    # Prepare windowed data - UNet mode (patch_mode=True)
    window_size = 33  # Increased window size from 17 to 33 to capture more context
    X_train, X_val, X_test, y_train, y_val, y_test, pos_weight, indices_mapping = prepare_windowed_data(
        eq_data, boundary_data, masks, window_size, patch_mode=True)

    # Train model
    model, history, model_id = train_and_evaluate(
        X_train, X_val, y_train, y_val,
        pos_weight=pos_weight,
        epochs=100,  # Increase epochs, rely on early stopping
        batch_size=8,  # Smaller batch size due to larger window size (33x33)
        patience=15,  # Longer patience value
        patch_mode=True
    )

    # Predict for test regions
    prediction_map = predict_for_test_regions(
        model, eq_data, masks, splits, window_size=window_size, patch_mode=True)

    # Visualize test region prediction results
    output_prefix = os.path.join(run_output_dir, f"test_prediction_{model_id}")
    metrics = visualize_test_predictions(
        prediction_map, boundary_data, masks, splits, eq_meta,
        threshold=0.5, output_path_prefix=output_prefix
    )

    # Save configuration info
    config = {
        'timestamp': timestamp,
        'window_size': window_size,
        'improved_postprocessing': True,
        'improvements': [
            'Increased window size from 17 to 33 for better context',
            'Enhanced post-processing to reduce fragmentation',
            'Removed small isolated regions',
            'Connected close boundaries',
            'Thinned boundaries for better precision'
        ],
        'train_regions': len(splits['train_regions']),
        'val_regions': len(splits['val_regions']),
        'test_regions': len(splits['test_regions']),
        'metrics': metrics,
        'model_id': model_id,
        'model_type': 'UNet'
    }

    with open(os.path.join(run_output_dir, 'config.txt'), 'w') as f:
        f.write(str(config))

    print(f"Plate boundary prediction completed! F1 score: {metrics['f1']:.4f}")
    return model, prediction_map, metrics


if __name__ == "__main__":
    main()