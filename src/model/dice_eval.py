import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import tensorflow as tf
from skimage import measure
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import math
from datetime import datetime
from skimage.morphology import binary_dilation, skeletonize
from skimage.metrics import hausdorff_distance
import matplotlib.cm as cm
from scipy.spatial.distance import cdist

# 路径设置
DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2processed"
RAW_DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2raw"
MODEL_PATH = r"C:\Users\debuf\Desktop\YuliFinalProject\src\model\models\unet_v2_64.h5"
NEW_EVAL_ROOT = r"C:\Users\debuf\Desktop\YuliFinalProject\result\new_eval"

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
    earthquake_raster_path = os.path.join(DATA_DIR, "earthquake_features_raster.tif")
    with rasterio.open(earthquake_raster_path) as eq_src:
        eq_data = eq_src.read()
        eq_meta = eq_src.meta
        eq_transform = eq_src.transform
        eq_crs = eq_src.crs
    boundaries_raster_path = os.path.join(DATA_DIR, "plate_boundaries_raster.tif")
    with rasterio.open(boundaries_raster_path) as bound_src:
        boundary_data = bound_src.read()
    min_height = min(eq_data.shape[1], boundary_data.shape[1])
    min_width = min(eq_data.shape[2], boundary_data.shape[2])
    eq_data = eq_data[:, :min_height, :min_width]
    boundary_data = boundary_data[:, :min_height, :min_width]
    transformed_eq_data = np.zeros_like(eq_data)
    transformed_boundary_data = np.zeros_like(boundary_data)
    for i in range(eq_data.shape[0]):
        transformed_eq_data[i] = np.fliplr(np.rot90(eq_data[i], k=2))
    for i in range(boundary_data.shape[0]):
        transformed_boundary_data[i] = np.fliplr(np.rot90(boundary_data[i], k=2))
    return transformed_eq_data, transformed_boundary_data, eq_meta, eq_transform, eq_crs

def define_geological_regions(eq_data, boundary_data, n_regions_h=4, n_regions_w=4, random_state=42):
    num_bands, height, width = eq_data.shape
    np.random.seed(random_state)
    region_mask = np.zeros((height, width), dtype=np.int32)
    region_stats = {}
    region_height = height // n_regions_h
    region_width = width // n_regions_w
    region_id = 0
    for i in range(n_regions_h):
        for j in range(n_regions_w):
            h_start = i * region_height
            h_end = (i + 1) * region_height if i < n_regions_h - 1 else height
            w_start = j * region_width
            w_end = (j + 1) * region_width if j < n_regions_w - 1 else width
            region_mask[h_start:h_end, w_start:w_end] = region_id
            region_pixels = (h_end - h_start) * (w_end - w_start)
            boundary_pixels = np.sum(boundary_data[0, h_start:h_end, w_start:w_end] > 0)
            boundary_percentage = boundary_pixels / region_pixels * 100
            earthquake_stats = {}
            for b in range(num_bands):
                band_data = eq_data[b, h_start:h_end, w_start:w_end]
                earthquake_stats[f'band_{b}_mean'] = np.mean(band_data)
                earthquake_stats[f'band_{b}_std'] = np.std(band_data)
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
    np.random.seed(random_state)
    all_regions = list(region_stats.keys())
    total_regions = len(all_regions)
    n_test_regions = max(1, int(total_regions * test_size))
    n_val_regions = max(1, int(total_regions * val_size))
    regions_by_boundary = sorted(
        all_regions,
        key=lambda r: region_stats[r]['boundary_percentage'],
        reverse=True
    )
    high_boundary_regions = [r for r in regions_by_boundary
                             if region_stats[r]['boundary_percentage'] >= min_boundary_percentage]
    if len(high_boundary_regions) >= n_test_regions:
        test_regions = np.random.choice(high_boundary_regions, size=n_test_regions, replace=False)
    else:
        test_regions = high_boundary_regions.copy()
        remaining_needed = n_test_regions - len(test_regions)
        remaining_regions = [r for r in all_regions if r not in test_regions]
        if remaining_needed > 0 and remaining_regions:
            additional_regions = np.random.choice(remaining_regions,
                                                  size=min(remaining_needed, len(remaining_regions)),
                                                  replace=False)
            test_regions = np.concatenate([test_regions, additional_regions])
    remaining_regions = [r for r in all_regions if r not in test_regions]
    val_regions = np.random.choice(
        remaining_regions,
        size=min(n_val_regions, len(remaining_regions)),
        replace=False
    )
    train_regions = [r for r in all_regions if r not in test_regions and r not in val_regions]
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
    splits = {
        'train_regions': train_regions,
        'val_regions': val_regions,
        'test_regions': test_regions,
        'region_stats': region_stats
    }
    masks = {'train': train_mask, 'val': val_mask, 'test': test_mask, 'region': region_mask}
    return masks, splits

def prepare_windowed_data(eq_data, boundary_data, masks, window_size=7, patch_mode=True):
    num_bands, height, width = eq_data.shape
    half_window = window_size // 2
    train_count = np.sum(masks['train'])
    val_count = np.sum(masks['val'])
    test_count = np.sum(masks['test'])
    train_indices = np.where(masks['train'])
    val_indices = np.where(masks['val'])
    test_indices = np.where(masks['test'])
    train_index_map = {(y, x): i for i, (y, x) in enumerate(zip(train_indices[0], train_indices[1]))}
    val_index_map = {(y, x): i for i, (y, x) in enumerate(zip(val_indices[0], val_indices[1]))}
    test_index_map = {(y, x): i for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1]))}
    eq_data_padded = global_pad(eq_data, half_window)
    if patch_mode:
        X_train = np.zeros((train_count, window_size, window_size, num_bands))
        X_val = np.zeros((val_count, window_size, window_size, num_bands))
        X_test = np.zeros((test_count, window_size, window_size, num_bands))
        y_train = np.zeros((train_count, window_size, window_size, 1))
        y_val = np.zeros((val_count, window_size, window_size, 1))
        y_test = np.zeros((test_count, window_size, window_size, 1))
        boundary_data_padded = global_pad(boundary_data, half_window)
        def process_patches(indices, X_patches, y_patches):
            count = len(indices[0])
            for i, (y, x) in enumerate(zip(indices[0], indices[1])):
                padded_y, padded_x = y + half_window, x + half_window
                for b in range(num_bands):
                    X_patches[i, :, :, b] = eq_data_padded[b,
                                           padded_y - half_window:padded_y + half_window + 1,
                                           padded_x - half_window:padded_x + half_window + 1]
                y_patches[i, :, :, 0] = boundary_data_padded[0,
                                        padded_y - half_window:padded_y + half_window + 1,
                                        padded_x - half_window:padded_x + half_window + 1]
        process_patches(train_indices, X_train, y_train)
        process_patches(val_indices, X_val, y_val)
        process_patches(test_indices, X_test, y_test)
        y_train = (y_train > 0).astype(np.float32)
        y_val = (y_val > 0).astype(np.float32)
        y_test = (y_test > 0).astype(np.float32)
    else:
        X_train = np.zeros((train_count, window_size, window_size, num_bands))
        X_val = np.zeros((val_count, window_size, window_size, num_bands))
        X_test = np.zeros((test_count, window_size, window_size, num_bands))
        def process_windows(indices, windowed_data):
            count = len(indices[0])
            for i, (y, x) in enumerate(zip(indices[0], indices[1])):
                padded_y, padded_x = y + half_window, x + half_window
                for b in range(num_bands):
                    windowed_data[i, :, :, b] = eq_data_padded[b,
                                               padded_y - half_window:padded_y + half_window + 1,
                                               padded_x - half_window:padded_x + half_window + 1]
        process_windows(train_indices, X_train)
        process_windows(val_indices, X_val)
        process_windows(test_indices, X_test)
        y_train = boundary_data[0][train_indices]
        y_val = boundary_data[0][val_indices]
        y_test = boundary_data[0][test_indices]
        y_train = (y_train > 0).astype(np.float32)
        y_val = (y_val > 0).astype(np.float32)
        y_test = (y_test > 0).astype(np.float32)
    pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1.0
    indices_mapping = {
        'train': {'indices': train_indices, 'index_map': train_index_map},
        'val': {'indices': val_indices, 'index_map': val_index_map},
        'test': {'indices': test_indices, 'index_map': test_index_map}
    }
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            pos_weight, indices_mapping)

def improved_postprocessing(prediction_map, threshold=0.5):
    binary_prediction = (prediction_map >= threshold).astype(np.uint8)
    binary_prediction = ndimage.binary_opening(binary_prediction, structure=np.ones((3, 3)))
    binary_prediction = ndimage.binary_closing(binary_prediction, structure=np.ones((5, 5)))
    labeled_array, num_features = ndimage.label(binary_prediction)
    component_sizes = np.bincount(labeled_array.ravel())
    small_size = 10
    too_small = component_sizes < small_size
    too_small[0] = False
    binary_prediction = ~np.isin(labeled_array, np.where(too_small))
    binary_prediction = ndimage.binary_closing(binary_prediction, structure=np.ones((3, 3)))
    binary_prediction = ndimage.binary_erosion(binary_prediction, structure=np.ones((2, 2)))
    return binary_prediction.astype(np.uint8)

def predict_for_test_regions(model, eq_data, masks, splits, window_size=7, batch_size=32, patch_mode=True):
    num_bands, height, width = eq_data.shape
    half_window = window_size // 2
    test_mask = masks['test']
    test_indices = np.where(test_mask)
    num_test_pixels = len(test_indices[0])
    eq_data_padded = global_pad(eq_data, half_window)
    prediction_map = np.zeros((height, width), dtype=np.float32)
    if patch_mode:
        batch_size = min(batch_size, 128)
        for batch_start in range(0, num_test_pixels, batch_size):
            batch_end = min(batch_start + batch_size, num_test_pixels)
            batch_indices = (test_indices[0][batch_start:batch_end], test_indices[1][batch_start:batch_end])
            batch_size_actual = batch_end - batch_start
            X_batch = np.zeros((batch_size_actual, window_size, window_size, num_bands))
            for i, (y, x) in enumerate(zip(batch_indices[0], batch_indices[1])):
                padded_y, padded_x = y + half_window, x + half_window
                for b in range(num_bands):
                    X_batch[i, :, :, b] = eq_data_padded[b,
                                         padded_y - half_window:padded_y + half_window + 1,
                                         padded_x - half_window:padded_x + half_window + 1]
            pred_batch = model.predict(X_batch, verbose=0)
            for i, (y, x) in enumerate(zip(batch_indices[0], batch_indices[1])):
                prediction_map[y, x] = pred_batch[i, half_window, half_window, 0]
    else:
        X_test_windowed = np.zeros((num_test_pixels, window_size, window_size, num_bands))
        for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1])):
            padded_y, padded_x = y + half_window, x + half_window
            for b in range(num_bands):
                X_test_windowed[i, :, :, b] = eq_data_padded[b,
                                             padded_y - half_window:padded_y + half_window + 1,
                                             padded_x - half_window:padded_x + half_window + 1]
        predictions = []
        total_batches = math.ceil(num_test_pixels / batch_size)
        for i in range(0, num_test_pixels, batch_size):
            batch_end = min(i + batch_size, num_test_pixels)
            batch_predictions = model.predict(X_test_windowed[i:batch_end], verbose=0)
            predictions.append(batch_predictions)
        all_predictions = np.vstack(predictions)
        for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1])):
            prediction_map[y, x] = all_predictions[i, 0]
    print("预测概率均值：", np.mean(prediction_map))
    return prediction_map

def visualize_test_predictions_with_score(prediction_map, boundary_data, masks, splits, eq_meta,
                               threshold=0.5, output_path_prefix=None,
                               f1=None, fuzzy_f1=None, precision=None, fuzzy_precision=None, recall=None, fuzzy_recall=None, hausdorff=None):
    test_mask = masks['test']
    test_indices = np.where(test_mask)
    test_true = (boundary_data[0][test_indices] > 0).astype(np.int32)
    binary_prediction = improved_postprocessing(prediction_map, threshold)
    test_pred = binary_prediction[test_indices].astype(np.int32)
    true_positives = np.sum((test_true == 1) & (test_pred == 1))
    false_positives = np.sum((test_true == 0) & (test_pred == 1))
    false_negatives = np.sum((test_true == 1) & (test_pred == 0))
    precision_ = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall_ = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_ = 2 * precision_ * recall_ / (precision_ + recall_) if (precision_ + recall_) > 0 else 0
    # Use provided metrics if available
    if f1 is not None: f1_ = f1
    if fuzzy_f1 is None or fuzzy_precision is None or fuzzy_recall is None or hausdorff is None:
        fuzzy_precision, fuzzy_recall, fuzzy_f1 = 0, 0, 0
        hausdorff = 0
    # Main heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    background = np.zeros_like(test_mask, dtype=np.float32)
    background[test_mask] = 0.1
    ax.imshow(background, cmap='gray', alpha=0.3)
    test_prediction = np.ma.masked_where(~test_mask, prediction_map)
    im = ax.imshow(test_prediction, cmap=LinearSegmentedColormap.from_list('prediction_cmap', ['navy', 'deepskyblue', 'gold'], N=256), alpha=0.7, vmin=0, vmax=1)
    original_contours = measure.find_contours(boundary_data[0], 0.5)
    for contour in original_contours:
        ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5)
    title = (
        f"F1: {f1_:.2f} | Fuzzy F1: {fuzzy_f1:.2f} | "
        f"Precision: {precision_:.2f} | Fuzzy Precision: {fuzzy_precision:.2f} | "
        f"Recall: {recall_:.2f} | Fuzzy Recall: {fuzzy_recall:.2f} | "
        f"Hausdorff: {hausdorff:.2f}"
    )
    ax.set_title(title, fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label('Prediction Probability')
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_heatmap.png", dpi=300)
    plt.close()
    # Region heatmaps (can be similarly updated if needed)
    # 生成每个测试区域的小图
    for region_id in splits['test_regions']:
        region_info = splits['region_stats'][region_id]
        h_start, h_end = region_info['h_start'], region_info['h_end']
        w_start, w_end = region_info['w_start'], region_info['w_end']
        region_prediction = prediction_map[h_start:h_end, w_start:w_end]
        region_boundary = boundary_data[0, h_start:h_end, w_start:w_end]
        region_mask = test_mask[h_start:h_end, w_start:w_end]
        region_true = (region_boundary > 0).astype(np.int32)
        region_pred = (region_prediction >= threshold).astype(np.int32)
        region_true_positives = np.sum((region_true == 1) & (region_pred == 1))
        region_false_positives = np.sum((region_true == 0) & (region_pred == 1))
        region_false_negatives = np.sum((region_true == 1) & (region_pred == 0))
        region_precision = region_true_positives / (region_true_positives + region_false_positives) if (region_true_positives + region_false_positives) > 0 else 0
        region_recall = region_true_positives / (region_true_positives + region_false_negatives) if (region_true_positives + region_false_negatives) > 0 else 0
        region_f1 = 2 * region_precision * region_recall / (region_precision + region_recall) if (region_precision + region_recall) > 0 else 0
        region_fuzzy_precision, region_fuzzy_recall, region_fuzzy_f1 = fuzzy_f1_score(region_true, region_pred, dilation_radius=2)
        region_hausdorff = hausdorff_distance(region_true, region_pred) if np.any(region_true) and np.any(region_pred) else 0
        region_title = (
            f"Region {region_id} - F1: {region_f1:.2f} | Fuzzy F1: {region_fuzzy_f1:.2f} | "
            f"Precision: {region_precision:.2f} | Fuzzy Precision: {region_fuzzy_precision:.2f} | "
            f"Recall: {region_recall:.2f} | Fuzzy Recall: {region_fuzzy_recall:.2f} | "
            f"Hausdorff: {region_hausdorff:.2f}"
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        background = np.zeros_like(region_mask, dtype=np.float32)
        background[region_mask] = 0.1
        ax.imshow(background, cmap='gray', alpha=0.3)
        region_prediction_masked = np.ma.masked_where(~region_mask, region_prediction)
        im = ax.imshow(region_prediction_masked, cmap=LinearSegmentedColormap.from_list('prediction_cmap', ['navy', 'deepskyblue', 'gold'], N=256), alpha=0.7, vmin=0, vmax=1)
        original_contours = measure.find_contours(region_boundary, 0.5)
        for contour in original_contours:
            ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5)
        ax.set_title(region_title, fontsize=10)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label('Prediction Probability')
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_region_{region_id}.png", dpi=300)
        plt.close()
    return {
        'precision': precision_, 'recall': recall_, 'f1': f1_,
        'threshold': threshold, 'true_positives': true_positives,
        'false_positives': false_positives, 'false_negatives': false_negatives
    }

def fuzzy_f1_score(y_true, y_pred, dilation_radius=2):
    structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1))
    y_true_dil = binary_dilation(y_true, footprint=structure)
    y_pred_dil = binary_dilation(y_pred, footprint=structure)
    tp = np.sum(y_pred & y_true_dil)
    fp = np.sum(y_pred & (~y_true_dil))
    fn = np.sum((~y_pred) & y_true_dil)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def visualize_error_heatmap(y_true, y_pred, output_path):
    from scipy.ndimage import distance_transform_edt
    dist_true = distance_transform_edt(~y_true)
    dist_pred = distance_transform_edt(~y_pred)
    error_map = np.zeros_like(y_true, dtype=np.float32)
    error_map[y_pred > 0] = dist_true[y_pred > 0]
    error_map[y_true > 0] = dist_pred[y_true > 0]
    vmax = np.percentile(error_map[error_map > 0], 95) if np.any(error_map > 0) else 1
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.ones_like(y_true), cmap='gray', alpha=0.3)
    im = ax.imshow(np.ma.masked_where(error_map == 0, error_map), cmap='plasma', alpha=0.7, vmin=0, vmax=vmax)
    ax.set_title('Boundary Distance Error Map')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label('Distance Error')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_distance_histogram(y_true, y_pred, output_path):
    from scipy.ndimage import distance_transform_edt
    sk_true = skeletonize(y_true)
    sk_pred = skeletonize(y_pred)
    true_pts = np.argwhere(sk_true)
    pred_pts = np.argwhere(sk_pred)
    dists = cdist(true_pts, pred_pts)
    min_dists = np.min(dists, axis=1)
    plt.figure(figsize=(6,4))
    plt.hist(min_dists, bins=50, color='orange', alpha=0.7)
    plt.xlabel('Distance to Closest Predicted Point')
    plt.ylabel('Count')
    plt.title('Distance Distribution (GT to Pred)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)

def evaluate_model(model_path, output_tag):
    print(f"\n正在评估模型: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss})
    print(model.summary())
    eq_data, boundary_data, eq_meta, eq_transform, eq_crs = load_raster_data()
    region_mask, region_stats = define_geological_regions(eq_data, boundary_data)
    masks, splits = create_improved_train_test_split(region_mask, region_stats)
    window_size = 33
    X_train, X_val, X_test, y_train, y_val, y_test, pos_weight, indices_mapping = prepare_windowed_data(
        eq_data, boundary_data, masks, window_size, patch_mode=True)
    prediction_map = predict_for_test_regions(
        model, eq_data, masks, splits, window_size=window_size, patch_mode=True)
    threshold = 0.30
    binary_prediction = improved_postprocessing(prediction_map, threshold)
    test_mask = masks['test']
    test_indices = np.where(test_mask)
    test_true = (boundary_data[0][test_indices] > 0).astype(np.int32)
    test_pred = binary_prediction[test_indices].astype(np.int32)
    tp = np.sum((test_true == 1) & (test_pred == 1))
    fp = np.sum((test_true == 0) & (test_pred == 1))
    fn = np.sum((test_true == 1) & (test_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    test_true_mask = (boundary_data[0] > 0) & test_mask
    test_pred_mask = (binary_prediction > 0) & test_mask
    fuzzy_precision, fuzzy_recall, fuzzy_f1 = fuzzy_f1_score(test_true_mask, test_pred_mask, dilation_radius=2)
    try:
        hausdorff = hausdorff_distance(test_true_mask, test_pred_mask)
    except Exception as e:
        hausdorff = -1
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fuzzy_precision': fuzzy_precision,
        'fuzzy_recall': fuzzy_recall,
        'fuzzy_f1': fuzzy_f1,
        'hausdorff_distance': float(hausdorff),
        'threshold': threshold
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(NEW_EVAL_ROOT, f"run_{output_tag}_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(run_output_dir, 'metrics.csv'), index=False)
    output_prefix = os.path.join(run_output_dir, f"test_prediction_unet_plate_boundary_{timestamp}")
    visualize_test_predictions_with_score(
        prediction_map, boundary_data, masks, splits, eq_meta,
        threshold=threshold, output_path_prefix=output_prefix,
        f1=f1, fuzzy_f1=fuzzy_f1, precision=precision, fuzzy_precision=fuzzy_precision, recall=recall, fuzzy_recall=fuzzy_recall, hausdorff=hausdorff
    )
    visualize_error_heatmap(test_true_mask, test_pred_mask, output_prefix + '_error_heatmap.png')
    plot_distance_histogram(test_true_mask, test_pred_mask, output_prefix + '_distance_histogram.png')
    config = {
        'timestamp': timestamp,
        'window_size': window_size,
        'metrics': metrics,
        'model_path': model_path
    }
    with open(os.path.join(run_output_dir, 'config.txt'), 'w') as f:
        f.write(str(config))
    print(f"{output_tag} 评估结果：F1={f1:.4f}，Fuzzy F1={fuzzy_f1:.4f}，Hausdorff={hausdorff:.2f}，结果保存在: {run_output_dir}")

if __name__ == "__main__":
    model_list = [
        (r'C:\\Users\\debuf\\Desktop\\YuliFinalProject\\src\\model\\models\\unet_v2_32.h5', 'base_filters=32'),
        (r'C:\\Users\\debuf\\Desktop\\YuliFinalProject\\src\\model\\models\\unet_v2_64.h5', 'base_filters=64')
    ]
    for model_path, tag in model_list:
        evaluate_model(model_path, tag)
