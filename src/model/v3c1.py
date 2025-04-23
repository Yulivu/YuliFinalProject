"""
基于ResNet思想的地震断层运动方式识别及板块边界预测模型
- 基于v2c2的改进
- 使用ResNet风格的残差连接提高特征提取能力
- 优化可视化效果和测试集评估
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
from tensorflow.keras import layers, models, callbacks, applications
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from skimage import measure
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
import time
import math
from datetime import datetime

# 设置路径 - 根据您的环境进行调整
DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2processed"
RAW_DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2raw"
MODEL_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\src\model"
OUTPUT_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\result\v3c1"
PLATE_BOUNDARIES_SHP = os.path.join(RAW_DATA_DIR, "plate", "plate_boundaries.shp")

# 确保目录存在
for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# 设置随机种子以确保结果可重复
np.random.seed(42)
tf.random.set_seed(42)

# 设置GPU内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU设置错误: {e}")


def global_pad(data, padding_size):
    """为全球地理数据实现自定义填充，处理经纬度边界"""
    if padding_size <= 0:
        return data
        
    channels, height, width = data.shape
    padded_data = np.zeros((channels, height + 2 * padding_size, width + 2 * padding_size), dtype=data.dtype)

    # 中央区域填充
    padded_data[:, padding_size:padding_size + height, padding_size:padding_size + width] = data

    # 处理经度边界（东西方向）- 环绕填充
    padded_data[:, padding_size:padding_size + height, :padding_size] = data[:, :, -padding_size:]  # 左边界
    padded_data[:, padding_size:padding_size + height, padding_size + width:] = data[:, :, :padding_size]  # 右边界

    # 处理纬度边界（南北方向）- 反射填充
    padded_data[:, :padding_size, padding_size:padding_size + width] = np.flip(data[:, :padding_size, :], axis=1)  # 上边界
    padded_data[:, padding_size + height:, padding_size:padding_size + width] = np.flip(data[:, -padding_size:, :], axis=1)  # 下边界

    # 处理四个角落
    padded_data[:, :padding_size, :padding_size] = np.flip(data[:, :padding_size, -padding_size:], axis=1)  # 左上角
    padded_data[:, :padding_size, padding_size + width:] = np.flip(data[:, :padding_size, :padding_size], axis=1)  # 右上角
    padded_data[:, padding_size + height:, :padding_size] = np.flip(data[:, -padding_size:, -padding_size:], axis=1)  # 左下角
    padded_data[:, padding_size + height:, padding_size + width:] = np.flip(data[:, -padding_size:, :padding_size], axis=1)  # 右下角

    return padded_data


def load_raster_data():
    """Load raster data, return earthquake features and plate boundary rasters"""
    print("Loading raster data...")

    # 加载地震特征栅格
    earthquake_raster_path = os.path.join(DATA_DIR, "earthquake_features_raster.tif")
    with rasterio.open(earthquake_raster_path) as eq_src:
        eq_data = eq_src.read()
        eq_meta = eq_src.meta
        eq_transform = eq_src.transform
        eq_crs = eq_src.crs

    # 加载板块边界栅格
    boundaries_raster_path = os.path.join(DATA_DIR, "plate_boundaries_raster.tif")
    with rasterio.open(boundaries_raster_path) as bound_src:
        boundary_data = bound_src.read()

    # 调整尺寸以确保两个栅格具有相同的尺寸
    min_height = min(eq_data.shape[1], boundary_data.shape[1])
    min_width = min(eq_data.shape[2], boundary_data.shape[2])

    eq_data = eq_data[:, :min_height, :min_width]
    boundary_data = boundary_data[:, :min_height, :min_width]

    print(f"Earthquake feature raster shape: {eq_data.shape}, Plate boundary raster shape: {boundary_data.shape}")

    return eq_data, boundary_data, eq_meta, eq_transform, eq_crs


def define_geological_regions(eq_data, boundary_data, n_regions_h=4, n_regions_w=4, random_state=42):
    """Define regions based on geological features"""
    print(f"Defining {n_regions_h}x{n_regions_w} geological regions...")

    # 获取数据形状
    num_bands, height, width = eq_data.shape
    np.random.seed(random_state)

    # 创建区域掩码
    region_mask = np.zeros((height, width), dtype=np.int32)
    region_stats = {}
    region_height = height // n_regions_h
    region_width = width // n_regions_w
    region_id = 0

    for i in range(n_regions_h):
        for j in range(n_regions_w):
            # 定义区域边界
            h_start = i * region_height
            h_end = (i + 1) * region_height if i < n_regions_h - 1 else height
            w_start = j * region_width
            w_end = (j + 1) * region_width if j < n_regions_w - 1 else width

            # 标记区域
            region_mask[h_start:h_end, w_start:w_end] = region_id

            # 计算该区域的统计信息
            region_pixels = (h_end - h_start) * (w_end - w_start)
            boundary_pixels = np.sum(boundary_data[0, h_start:h_end, w_start:w_end] > 0)
            boundary_percentage = boundary_pixels / region_pixels * 100

            # 计算地震特征平均值和标准差
            earthquake_stats = {}
            for b in range(num_bands):
                band_data = eq_data[b, h_start:h_end, w_start:w_end]
                earthquake_stats[f'band_{b}_mean'] = np.mean(band_data)
                earthquake_stats[f'band_{b}_std'] = np.std(band_data)

            # 存储区域统计信息
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
    """Create an improved train-validation-test split, ensuring test regions contain enough boundary information"""
    print("Creating improved train-validation-test split...")
    np.random.seed(random_state)

    # 获取所有区域ID
    all_regions = list(region_stats.keys())
    total_regions = len(all_regions)

    # 计算测试和验证区域数量
    n_test_regions = max(1, int(total_regions * test_size))
    n_val_regions = max(1, int(total_regions * val_size))

    # 按边界百分比排序区域
    regions_by_boundary = sorted(
        all_regions,
        key=lambda r: region_stats[r]['boundary_percentage'],
        reverse=True
    )

    # 确保测试区域包含足够的边界
    high_boundary_regions = [r for r in regions_by_boundary
                             if region_stats[r]['boundary_percentage'] >= min_boundary_percentage]

    if len(high_boundary_regions) >= n_test_regions:
        # 如果有足够的高边界区域，从中随机选择
        test_regions = np.random.choice(high_boundary_regions, size=n_test_regions, replace=False)
    else:
        # 否则，使用所有高边界区域并从其余区域中随机选择
        test_regions = high_boundary_regions.copy()
        remaining_needed = n_test_regions - len(test_regions)
        remaining_regions = [r for r in all_regions if r not in test_regions]
        if remaining_needed > 0 and remaining_regions:
            additional_regions = np.random.choice(remaining_regions,
                                                  size=min(remaining_needed, len(remaining_regions)),
                                                  replace=False)
            test_regions = np.concatenate([test_regions, additional_regions])

    # 从剩余区域中选择验证集
    remaining_regions = [r for r in all_regions if r not in test_regions]
    val_regions = np.random.choice(
        remaining_regions,
        size=min(n_val_regions, len(remaining_regions)),
        replace=False
    )

    # 剩余区域为训练集
    train_regions = [r for r in all_regions if r not in test_regions and r not in val_regions]

    # 创建掩码
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

    # 保存划分信息
    splits = {
        'train_regions': train_regions,
        'val_regions': val_regions,
        'test_regions': test_regions,
        'region_stats': region_stats
    }

    # 创建掩码字典
    masks = {'train': train_mask, 'val': val_mask, 'test': test_mask, 'region': region_mask}

    print(f"Train/validation/test regions: {len(train_regions)}/{len(val_regions)}/{len(test_regions)} regions")

    return masks, splits


def visualize_data_split(masks, splits, output_path):
    """Visualize the data split, showing training, validation and test regions"""
    print("Visualizing data split...")
    
    region_mask = masks['region']
    train_mask = masks['train']
    val_mask = masks['val']
    test_mask = masks['test']
    
    # Create region type map
    split_map = np.zeros_like(region_mask, dtype=np.uint8)
    split_map[train_mask] = 1  # Training
    split_map[val_mask] = 2    # Validation
    split_map[test_mask] = 3   # Testing
    
    # Set visualization colors
    cmap = plt.cm.get_cmap('viridis', 4)
    colors = [cmap(i) for i in range(4)]
    colors[0] = (0.8, 0.8, 0.8, 1.0)  # Gray - Unused
    colors[1] = (0.2, 0.7, 0.2, 1.0)  # Green - Training
    colors[2] = (0.9, 0.7, 0.1, 1.0)  # Yellow - Validation
    colors[3] = (0.8, 0.2, 0.2, 1.0)  # Red - Testing
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_split', colors, N=4)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(split_map, cmap=custom_cmap, interpolation='nearest')
    
    # Add region labels
    for r_id, stats in splits['region_stats'].items():
        y_center = (stats['h_start'] + stats['h_end']) // 2
        x_center = (stats['w_start'] + stats['w_end']) // 2
        r_type = "Train" if r_id in splits['train_regions'] else ("Val" if r_id in splits['val_regions'] else "Test" if r_id in splits['test_regions'] else "Unused")
        boundary_pct = stats['boundary_percentage']
        
        # Region ID and boundary percentage label
        plt.text(x_center, y_center, f"{r_id}\n({boundary_pct:.1f}%)",
                 ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.7))
    
    # Create legend
    legend_elements = [
        Patch(facecolor=colors[1], edgecolor='black', label=f'Training ({len(splits["train_regions"])} regions)'),
        Patch(facecolor=colors[2], edgecolor='black', label=f'Validation ({len(splits["val_regions"])} regions)'),
        Patch(facecolor=colors[3], edgecolor='black', label=f'Testing ({len(splits["test_regions"])} regions)')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=3, fontsize=10)
    
    plt.title('Dataset Split - Based on Geological Regions', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Data split visualization saved to: {output_path}")


def prepare_windowed_data(eq_data, boundary_data, masks, window_size=7):
    """Prepare windowed data for training"""
    print(f"Preparing windowed data, window size: {window_size}x{window_size}...")
    
    # 获取数据形状
    num_bands, height, width = eq_data.shape
    half_window = window_size // 2
    
    # 使用自定义全球填充，正确处理经纬度边界
    eq_data_padded = global_pad(eq_data, half_window)
    
    # 获取训练和验证掩码
    train_mask = masks['train']
    val_mask = masks['val']
    
    # 获取训练和验证像素的索引
    train_indices = np.where(train_mask)
    val_indices = np.where(val_mask)
    
    num_train_pixels = len(train_indices[0])
    num_val_pixels = len(val_indices[0])
    
    print(f"Training set: {num_train_pixels} pixels, Validation set: {num_val_pixels} pixels")
    
    # 处理窗口数据的函数
    def process_windows(indices, windowed_data):
        num_pixels = len(indices[0])
        X = np.zeros((num_pixels, window_size, window_size, num_bands))
        y = np.zeros(num_pixels, dtype=np.float32)
        
        for i, (y_idx, x_idx) in enumerate(zip(indices[0], indices[1])):
            if i % 100000 == 0 and i > 0:
                print(f"  Processed {i}/{num_pixels} pixels")
            
            # 获取中心点对应的填充后索引
            padded_y, padded_x = y_idx + half_window, x_idx + half_window
            
            # 提取每个波段的窗口
            for b in range(num_bands):
                X[i, :, :, b] = eq_data_padded[b,
                                padded_y - half_window:padded_y + half_window + 1,
                                padded_x - half_window:padded_x + half_window + 1]
            
            # 设置标签 - 是否为板块边界
            y[i] = boundary_data[0, y_idx, x_idx] > 0
        
        return X, y
    
    # 处理训练和验证数据
    print("Processing training data...")
    X_train, y_train = process_windows(train_indices, eq_data_padded)
    
    print("Processing validation data...")
    X_val, y_val = process_windows(val_indices, eq_data_padded)
    
    # 计算正样本比例
    pos_train_ratio = np.mean(y_train)
    pos_val_ratio = np.mean(y_val)
    
    print(f"Training set positive ratio: {pos_train_ratio:.4f}, Validation set positive ratio: {pos_val_ratio:.4f}")
    
    # 计算类别权重（用于处理类别不平衡）
    pos_weight = (1 - pos_train_ratio) / pos_train_ratio if pos_train_ratio > 0 else 1.0
    print(f"Positive class weight: {pos_weight:.2f}")
    
    return X_train, X_val, y_train, y_val, pos_weight


def build_resnet_model(input_shape, dropout_rate=0.3):
    """构建基于ResNet思想的CNN模型用于板块边界识别"""
    print("构建基于ResNet思想的CNN模型...")
    
    # 设置L2正则化系数
    l2_reg = 1e-4
    reg = tf.keras.regularizers.l2(l2_reg)

    # 输入层
    inputs = layers.Input(shape=input_shape)
    
    # 自定义小型ResNet架构，适用于小窗口尺寸
    
    # 第一个卷积块
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    
    # 第一个残差连接
    shortcut = layers.Conv2D(64, (1, 1), padding='same', kernel_regularizer=reg)(inputs)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    
    # 第二个卷积块
    y = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg)(y)
    y = layers.BatchNormalization()(y)
    
    # 第二个残差连接
    shortcut = layers.Conv2D(128, (1, 1), padding='same', kernel_regularizer=reg)(x)
    y = layers.add([y, shortcut])
    y = layers.Activation('relu')(y)
    
    # 第三个卷积块
    z = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(y)
    z = layers.BatchNormalization()(z)
    z = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg)(z)
    z = layers.BatchNormalization()(z)
    
    # 第三个残差连接
    shortcut = layers.Conv2D(256, (1, 1), padding='same', kernel_regularizer=reg)(y)
    z = layers.add([z, shortcut])
    z = layers.Activation('relu')(z)
    
    # 全局池化
    z = layers.GlobalAveragePooling2D()(z)
    
    # 全连接层
    z = layers.Dense(256, activation='relu', kernel_regularizer=reg)(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(dropout_rate)(z)
    
    # 输出层
    outputs = layers.Dense(1, activation='sigmoid')(z)
    
    # 创建模型
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
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
    """创建带权重的二元交叉熵损失函数，用于处理类别不平衡"""
    def loss(y_true, y_pred):
        # 避免log(0)错误
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # 带权重的二元交叉熵
        loss_pos = -pos_weight * y_true * tf.math.log(y_pred)
        loss_neg = -(1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss_pos + loss_neg)
    return loss


def train_and_evaluate(X_train, X_val, y_train, y_val, pos_weight=1.0,
                       epochs=50, batch_size=64, patience=10):
    """训练和评估模型，使用类别权重处理不平衡问题"""
    print("Starting training of ResNet-style model...")
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    # 生成时间戳作为模型ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"resnet_plate_boundary_model_{timestamp}"
    
    # 为当前运行创建子文件夹
    run_output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Results will be saved to: {run_output_dir}")
    
    model_path = os.path.join(MODEL_DIR, f"{model_id}.h5")

    # 构建ResNet模型
    model = build_resnet_model(X_train.shape[1:])
    model.summary()

    # 使用类别权重
    class_weights = {0: 1.0, 1: pos_weight}
    print(f"Using class weights: {class_weights}")

    # 设置回调函数
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=patience),
        callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience // 2, 
                                   min_lr=1e-6, verbose=1)
    ]

    # 训练模型
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed, time taken: {training_time:.2f} seconds")

    # 加载最佳模型
    model = tf.keras.models.load_model(model_path)

    # 评估模型
    val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(X_val, y_val, verbose=0)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
    
    print(f"Validation metrics: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}, F1={val_f1:.4f}")

    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss', fontsize=12)
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy', fontsize=12)
    plt.legend()

    # 绘制精确率和召回率
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Prec')
    plt.plot(history.history['val_precision'], label='Val Prec')
    plt.plot(history.history['recall'], label='Rec')
    plt.plot(history.history['val_recall'], label='Val Rec')
    plt.title('Precision/Recall', fontsize=12)
    plt.legend()

    plt.tight_layout()
    history_plot_path = os.path.join(run_output_dir, 'training_history.png')
    plt.savefig(history_plot_path, dpi=300)
    plt.close()

    # 保存训练历史到CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(run_output_dir, 'training_history.csv'), index=False)

    return model, history, model_id, run_output_dir


def predict_for_test_regions(model, eq_data, masks, splits, window_size=7, batch_size=256):
    """仅对测试区域进行预测"""
    print("Starting predictions for test regions...")

    num_bands, height, width = eq_data.shape
    half_window = window_size // 2
    test_mask = masks['test']

    # 只对测试区域的像素进行预测
    test_indices = np.where(test_mask)
    num_test_pixels = len(test_indices[0])
    print(f"Test regions: {num_test_pixels} pixels")

    # 使用自定义全球填充，正确处理经纬度边界
    eq_data_padded = global_pad(eq_data, half_window)

    # 初始化预测数组
    prediction_map = np.zeros((height, width), dtype=np.float32)

    # 创建测试区域的窗口化数据
    X_test_windowed = np.zeros((num_test_pixels, window_size, window_size, num_bands))

    # 收集测试区域像素的窗口化数据
    for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1])):
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i}/{num_test_pixels} test pixels")

        padded_y, padded_x = y + half_window, x + half_window
        for b in range(num_bands):
            X_test_windowed[i, :, :, b] = eq_data_padded[b,
                                         padded_y - half_window:padded_y + half_window + 1,
                                         padded_x - half_window:padded_x + half_window + 1]

    # 使用批处理进行预测
    print("Performing batch predictions for test regions...")
    predictions = []
    total_batches = math.ceil(num_test_pixels / batch_size)
    
    for i in range(0, num_test_pixels, batch_size):
        batch_end = min(i + batch_size, num_test_pixels)
        batch_predictions = model.predict(X_test_windowed[i:batch_end], verbose=0)
        predictions.append(batch_predictions)
        if (i // batch_size) % 20 == 0:
            print(f"  Completed {i // batch_size}/{total_batches} batches")
    
    # 合并所有预测结果
    all_predictions = np.concatenate(predictions).flatten()
    
    # 将预测结果填充到预测图中
    prediction_map[test_indices] = all_predictions
    
    return prediction_map


def visualize_test_predictions(prediction_map, boundary_data, masks, splits, eq_meta,
                               threshold=0.5, output_path_prefix=None):
    """Visualize the test region predictions"""
    print("Visualizing test region predictions...")
    
    if output_path_prefix is None:
        output_path_prefix = os.path.join(OUTPUT_DIR, "test_prediction")
    
    # Get test mask and region information
    test_mask = masks['test']
    test_regions = splits['test_regions']
    region_mask = masks['region']
    region_stats = splits['region_stats']

    # Binarize prediction
    binary_prediction = (prediction_map >= threshold).astype(np.uint8)

    # Post-processing: apply morphological operations to clean noise
    binary_prediction = ndimage.binary_opening(binary_prediction, structure=np.ones((3, 3)))
    binary_prediction = ndimage.binary_closing(binary_prediction, structure=np.ones((3, 3)))

    # Extract contours
    contours = measure.find_contours(prediction_map, threshold)

    # Calculate performance metrics for test regions
    test_indices = np.where(test_mask)
    test_true = (boundary_data[0][test_indices] > 0).astype(np.int32)
    test_pred = (prediction_map[test_indices] >= threshold).astype(np.int32)

    # Calculate precision, recall and F1
    true_positives = np.sum((test_true == 1) & (test_pred == 1))
    false_positives = np.sum((test_true == 0) & (test_pred == 1))
    false_negatives = np.sum((test_true == 1) & (test_pred == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Create evaluation report for test regions
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

    # Create overall test region prediction comparison figure
    plt.figure(figsize=(16, 12))
    
    # Show only test regions
    test_prediction = np.ma.masked_where(~test_mask, prediction_map)
    test_original = np.ma.masked_where(~test_mask, boundary_data[0])
    
    # Create background - test regions shown as light gray
    background = np.zeros_like(test_mask, dtype=np.float32)
    background[test_mask] = 0.1
    plt.imshow(background, cmap='gray', alpha=0.3)

    # Show original boundaries - red
    plt.imshow(test_original, cmap=boundary_cmap, alpha=0.7, vmin=0, vmax=1)

    # Draw prediction boundary contours - blue
    for contour in contours:
        # Check if contour is in test region
        contour_points = [(int(p[0]), int(p[1])) for p in contour 
                          if 0 <= int(p[0]) < test_mask.shape[0] and 0 <= int(p[1]) < test_mask.shape[1]]
        if any(test_mask[y, x] for y, x in contour_points):
            plt.plot(contour[:, 1], contour[:, 0], 'b-', linewidth=1)

    # Draw boundaries for each test region
    for region_id in test_regions:
        stats = region_stats[region_id]
        y_start, y_end = stats['h_start'], stats['h_end']
        x_start, x_end = stats['w_start'], stats['w_end']
        plt.plot([x_start, x_end, x_end, x_start, x_start],
                 [y_start, y_start, y_end, y_end, y_start],
                 'y-', linewidth=2, alpha=0.7)

    plt.title(f'Test Region Evaluation - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}', 
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
    
    plt.title('Test Region Prediction Heatmap', fontsize=14)
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
            print(f"  Region {region_id} is not in the test set, skipping visualization")
            continue
            
        # Calculate region performance metrics
        region_indices = np.where(region_test_mask)
        region_true = (boundary_data[0][region_indices] > 0).astype(np.int32)
        region_pred = (prediction_map[region_indices] >= threshold).astype(np.int32)
        
        region_tp = np.sum((region_true == 1) & (region_pred == 1))
        region_fp = np.sum((region_true == 0) & (region_pred == 1))
        region_fn = np.sum((region_true == 1) & (region_pred == 0))
        
        region_precision = region_tp / (region_tp + region_fp) if (region_tp + region_fp) > 0 else 0
        region_recall = region_tp / (region_tp + region_fn) if (region_tp + region_fn) > 0 else 0
        region_f1 = 2 * region_precision * region_recall / (region_precision + region_recall) if (region_precision + region_recall) > 0 else 0
        
        # Create region visualization - original boundary vs prediction
        plt.figure(figsize=(10, 8))
        
        # Crop region data
        region_data = boundary_data[0][y_start:y_end, x_start:x_end]
        region_pred_data = prediction_map[y_start:y_end, x_start:x_end]
        region_test_area = test_mask[y_start:y_end, x_start:x_end]
        
        # Create masked data
        region_original = np.ma.masked_where(~region_test_area, region_data)
        
        # Show region original boundaries
        plt.imshow(region_original, cmap=boundary_cmap, alpha=0.7, vmin=0, vmax=1)
        
        # Get contours in the region
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
                
        plt.title(f'Region {region_id} - Boundary: {stats["boundary_percentage"]:.1f}%\n'
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
                
        plt.title(f'Region {region_id} - Prediction Heatmap')
        plt.colorbar(label='Prediction Probability')
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_region_{region_id}_heatmap.png", dpi=300)
        plt.close()
    
    # Create metrics dictionary
    metrics = {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(true_positives),
        'fp': int(false_positives),
        'fn': int(false_negatives)
    }
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{output_path_prefix}_metrics.csv", index=False)
    
    print(f"Test prediction visualizations saved to: {output_path_prefix}_*.png")
    
    return metrics


def main():
    """Main function: Complete model training and evaluation workflow"""
    print("Starting ResNet-style plate boundary prediction process...")
    
    # Step 1: Load raster data
    eq_data, boundary_data, eq_meta, eq_transform, eq_crs = load_raster_data()
    
    # Step 2: Define geological regions
    region_mask, region_stats = define_geological_regions(eq_data, boundary_data, 
                                                         n_regions_h=4, n_regions_w=6)
    
    # Step 3: Create train-validation-test split
    masks, splits = create_improved_train_test_split(region_mask, region_stats, 
                                                   test_size=0.25, val_size=0.15,
                                                   min_boundary_percentage=0.5)
    
    # Create a timestamp subfolder to save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Results will be saved to: {run_output_dir}")
    
    # Visualize data split
    split_viz_path = os.path.join(run_output_dir, "data_split_visualization.png")
    visualize_data_split(masks, splits, split_viz_path)
    
    # Step 4: Prepare windowed data
    window_size = 7  # 7x7 window
    X_train, X_val, y_train, y_val, pos_weight = prepare_windowed_data(
        eq_data, boundary_data, masks, window_size=window_size
    )
    
    # Step 5: Train and evaluate model
    model, history, model_id, run_output_dir = train_and_evaluate(
        X_train, X_val, y_train, y_val, 
        pos_weight=pos_weight,
        epochs=50, 
        batch_size=64, 
        patience=10
    )
    
    # Step 6: Make predictions on test regions
    prediction_map = predict_for_test_regions(
        model, eq_data, masks, splits, 
        window_size=window_size,
        batch_size=256
    )
    
    # Step 7: Visualize test predictions
    output_path_prefix = os.path.join(run_output_dir, "test_prediction")
    test_metrics = visualize_test_predictions(
        prediction_map, boundary_data, masks, splits, eq_meta,
        threshold=0.5, 
        output_path_prefix=output_path_prefix
    )
    
    print("ResNet-style plate boundary prediction process completed!")
    
    return model, history, test_metrics


if __name__ == "__main__":
    main()
