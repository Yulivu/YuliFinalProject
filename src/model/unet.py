"""
改进版地震断层运动方式识别及板块边界预测模型
- 使用UNet架构进行分割
- 改进区域划分策略
- 优化可视化效果
- 增强测试集评估
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

# 设置路径 - 根据您的环境进行调整
DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2processed"
RAW_DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2raw"
MODEL_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\src\model"
OUTPUT_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\result\unet"
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
    """加载栅格数据，返回地震特征和板块边界栅格"""
    print("加载栅格数据...")

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

    print(f"地震特征栅格形状: {eq_data.shape}, 板块边界栅格形状: {boundary_data.shape}")

    return eq_data, boundary_data, eq_meta, eq_transform, eq_crs


def define_geological_regions(eq_data, boundary_data, n_regions_h=4, n_regions_w=4, random_state=42):
    """定义基于地质特征的区域划分"""
    print(f"定义 {n_regions_h}x{n_regions_w} 地质区域划分...")

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
    """创建改进的训练-验证-测试划分，确保测试区域包含足够的边界信息"""
    print("创建改进的训练-验证-测试划分...")
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

    print(f"训练/验证/测试区域: {len(train_regions)}/{len(val_regions)}/{len(test_regions)}个区域")

    return masks, splits


def visualize_data_split(masks, splits, output_path):
    """可视化数据划分，展示训练、验证和测试区域"""
    print("Visualizing data split...")

    region_mask = masks['region']
    train_mask = masks['train']
    val_mask = masks['val']
    test_mask = masks['test']
    region_stats = splits['region_stats']

    # 创建分割掩码的可视化
    height, width = region_mask.shape
    split_viz = np.zeros((height, width, 3), dtype=np.uint8)

    # 填充训练、验证和测试区域的颜色
    split_viz[train_mask] = [50, 150, 50]  # 训练区域为绿色
    split_viz[val_mask] = [150, 50, 50]    # 验证区域为红色
    split_viz[test_mask] = [50, 50, 150]   # 测试区域为蓝色

    # 绘制区域边界
    plt.figure(figsize=(15, 10))
    plt.imshow(split_viz)
    
    # 添加区域ID注释
    for region_id, stats in region_stats.items():
        h_center = (stats['h_start'] + stats['h_end']) // 2
        w_center = (stats['w_start'] + stats['w_end']) // 2
        boundary_perc = stats['boundary_percentage']
        
        # 根据边界百分比调整文本颜色
        if boundary_perc >= 1.0:
            text_color = 'yellow'
        else:
            text_color = 'white'
            
        plt.text(w_center, h_center, f"{region_id}\n{boundary_perc:.1f}%", 
                 ha='center', va='center', color=text_color, fontsize=8)
    
    # 添加图例
    legend_elements = [
        Patch(facecolor=(50/255, 150/255, 50/255), label='Training Regions'),
        Patch(facecolor=(150/255, 50/255, 50/255), label='Validation Regions'),
        Patch(facecolor=(50/255, 50/255, 150/255), label='Testing Regions')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Data Split: Training/Validation/Testing Regions', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def prepare_data_for_unet(eq_data, boundary_data, masks, patch_size=64, stride=32):
    """准备用于UNet模型的数据，提取图像块"""
    print("Preparing data for UNet model...")
    
    # 获取数据形状
    num_bands, height, width = eq_data.shape
    train_mask = masks['train']
    val_mask = masks['val']
    
    print(f"Original data shape: eq_data={eq_data.shape}, boundary_data={boundary_data.shape}")
    print(f"Train mask shape: {train_mask.shape}, Validation mask shape: {val_mask.shape}")
    
    # 确定填充大小以确保完整覆盖
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    
    # 填充数据
    eq_data_padded = np.pad(eq_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    boundary_data_padded = np.pad(boundary_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    train_mask_padded = np.pad(train_mask, ((0, pad_h), (0, pad_w)), mode='constant')
    val_mask_padded = np.pad(val_mask, ((0, pad_h), (0, pad_w)), mode='constant')
    
    # 调整填充后的形状
    _, padded_height, padded_width = eq_data_padded.shape
    print(f"Padded data shape: eq_data_padded={eq_data_padded.shape}")
    
    # 创建存储数据的列表
    X_train_patches = []
    y_train_patches = []
    X_val_patches = []
    y_val_patches = []
    
    patch_count = 0
    # 提取图像块
    for y in range(0, padded_height - patch_size + 1, stride):
        for x in range(0, padded_width - patch_size + 1, stride):
            # 获取当前图像块的掩码区域
            mask_patch = train_mask_padded[y:y+patch_size, x:x+patch_size]
            val_mask_patch = val_mask_padded[y:y+patch_size, x:x+patch_size]
            
            # 计算训练和验证掩码覆盖的像素百分比
            train_coverage = np.mean(mask_patch)
            val_coverage = np.mean(val_mask_patch)
            
            # 如果图像块至少有25%的像素属于训练/验证区域，则使用它
            if train_coverage >= 0.25:
                # 提取特征和标签图像块
                feature_patch = np.zeros((patch_size, patch_size, num_bands), dtype=np.float32)
                for b in range(num_bands):
                    feature_patch[:, :, b] = eq_data_padded[b, y:y+patch_size, x:x+patch_size]
                
                label_patch = boundary_data_padded[0, y:y+patch_size, x:x+patch_size]
                label_patch = label_patch.reshape(patch_size, patch_size, 1)  # [H, W, 1]
                
                X_train_patches.append(feature_patch)
                y_train_patches.append(label_patch)
                
                patch_count += 1
                if patch_count % 100 == 0:
                    print(f"Processed {patch_count} patches")
            
            elif val_coverage >= 0.25:
                # 提取特征和标签图像块
                feature_patch = np.zeros((patch_size, patch_size, num_bands), dtype=np.float32)
                for b in range(num_bands):
                    feature_patch[:, :, b] = eq_data_padded[b, y:y+patch_size, x:x+patch_size]
                
                label_patch = boundary_data_padded[0, y:y+patch_size, x:x+patch_size]
                label_patch = label_patch.reshape(patch_size, patch_size, 1)  # [H, W, 1]
                
                X_val_patches.append(feature_patch)
                y_val_patches.append(label_patch)
                
                patch_count += 1
                if patch_count % 100 == 0:
                    print(f"Processed {patch_count} patches")
    
    # 转换为NumPy数组
    if len(X_train_patches) > 0:
        X_train = np.array(X_train_patches, dtype=np.float32)
        y_train = np.array(y_train_patches, dtype=np.float32)
    else:
        raise ValueError("No training data found. Try reducing patch_size or stride.")
        
    if len(X_val_patches) > 0:
        X_val = np.array(X_val_patches, dtype=np.float32)
        y_val = np.array(y_val_patches, dtype=np.float32)
    else:
        raise ValueError("No validation data found. Try reducing patch_size or stride.")
    
    # 对标签进行二值化处理
    y_train = (y_train > 0).astype(np.float32)
    y_val = (y_val > 0).astype(np.float32)
    
    # 检查数据形状
    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}")
    
    # 确保输入数据形状正确
    if len(X_train.shape) != 4:
        raise ValueError(f"Incorrect training data shape: {X_train.shape}, should be (samples, height, width, channels)")
    
    if len(y_train.shape) != 4:
        raise ValueError(f"Incorrect training label shape: {y_train.shape}, should be (samples, height, width, 1)")
    
    return X_train, y_train, X_val, y_val


def build_unet_model(input_shape, dropout_rate=0.3):
    """构建UNet模型用于板块边界分割"""
    print("Building UNet model...")
    print(f"Model input shape: {input_shape}")
    
    # 确保输入形状正确
    if len(input_shape) != 3:
        raise ValueError(f"Incorrect input shape: {input_shape}, should be (height, width, channels)")
    
    # 设置L2正则化系数
    l2_reg = 1e-4
    reg = tf.keras.regularizers.l2(l2_reg)
    
    # 输入层
    inputs = layers.Input(shape=input_shape, name='input_layer')
    
    # 编码器部分
    # 第一个下采样块
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=reg)(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = layers.Dropout(dropout_rate)(pool1)
    
    # 第二个下采样块
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=reg)(drop1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=reg)(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = layers.Dropout(dropout_rate)(pool2)
    
    # 第三个下采样块
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=reg)(drop2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=reg)(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = layers.Dropout(dropout_rate)(pool3)
    
    # 桥接块
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=reg)(drop3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=reg)(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    drop4 = layers.Dropout(dropout_rate)(conv4)
    
    # 解码器部分
    # 第一个上采样块
    up5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(drop4)
    concat5 = layers.Concatenate()([up5, conv3])
    drop5 = layers.Dropout(dropout_rate)(concat5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=reg)(drop5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=reg)(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    
    # 第二个上采样块
    up6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    concat6 = layers.Concatenate()([up6, conv2])
    drop6 = layers.Dropout(dropout_rate)(concat6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=reg)(drop6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=reg)(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    
    # 第三个上采样块
    up7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    concat7 = layers.Concatenate()([up7, conv1])
    drop7 = layers.Dropout(dropout_rate)(concat7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=reg)(drop7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=reg)(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    
    # 输出层
    outputs = layers.Conv2D(1, 1, activation='sigmoid', name='output_layer')(conv7)
    
    # 创建模型
    model = models.Model(inputs=inputs, outputs=outputs, name='unet_model')
    
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


def train_and_evaluate_unet(X_train, y_train, X_val, y_val, pos_weight=1.0,
                           epochs=50, batch_size=16, patience=10):
    """训练和评估UNet模型，使用类别权重处理不平衡问题"""
    print("Starting UNet model training...")
    print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
    
    # 检查输入数据形状是否正确
    if len(X_train.shape) != 4:
        raise ValueError(f"Incorrect training data shape: {X_train.shape}, should be (samples, height, width, channels)")
    
    if len(y_train.shape) != 4:
        raise ValueError(f"Incorrect training label shape: {y_train.shape}, should be (samples, height, width, 1)")

    # 构建模型
    input_shape = X_train.shape[1:]  # (高度, 宽度, 通道数)
    model = build_unet_model(input_shape)
    model.summary()

    # 使用类别权重
    class_weights = {0: 1.0, 1: pos_weight}
    print(f"Using class weights: {class_weights}")

    # 生成时间戳作为模型ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"unet_boundary_model_{timestamp}"
    model_path = os.path.join(MODEL_DIR, f"{model_id}.h5")

    # 设置回调函数
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=patience),
        callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience // 2, 
                                    min_lr=1e-6, verbose=1)
    ]

    # 训练模型
    start_time = time.time()
    
    # 选择一个自定义的损失函数，处理类别不平衡
    custom_loss = weighted_binary_crossentropy(pos_weight)
    model.compile(
        optimizer='adam',
        loss=custom_loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    # 确保批量大小不大于样本数
    batch_size = min(batch_size, len(X_train))
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # 加载最佳模型
    model = tf.keras.models.load_model(model_path, custom_objects={'loss': custom_loss})

    # 评估模型
    val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(X_val, y_val, verbose=0)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
    
    print(f"Validation metrics: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}, F1={val_f1:.4f}")

    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    # 获取历史记录中的准确指标名称
    metric_names = list(history.history.keys())
    print(f"Available metrics: {metric_names}")
    
    # 找到对应的指标名称
    loss_key = 'loss'
    val_loss_key = 'val_loss'
    acc_key = [k for k in metric_names if 'accuracy' in k and not 'val' in k][0]
    val_acc_key = [k for k in metric_names if 'accuracy' in k and 'val' in k][0]
    prec_key = [k for k in metric_names if 'precision' in k and not 'val' in k][0] 
    val_prec_key = [k for k in metric_names if 'precision' in k and 'val' in k][0]
    rec_key = [k for k in metric_names if 'recall' in k and not 'val' in k][0]
    val_rec_key = [k for k in metric_names if 'recall' in k and 'val' in k][0]

    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history.history[loss_key], label='Train')
    plt.plot(history.history[val_loss_key], label='Val')
    plt.title('Loss', fontsize=12)
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history.history[acc_key], label='Train')
    plt.plot(history.history[val_acc_key], label='Val')
    plt.title('Accuracy', fontsize=12)
    plt.legend()

    # 绘制精确率和召回率
    plt.subplot(1, 3, 3)
    plt.plot(history.history[prec_key], label='Prec')
    plt.plot(history.history[val_prec_key], label='Val Prec')
    plt.plot(history.history[rec_key], label='Rec')
    plt.plot(history.history[val_rec_key], label='Val Rec')
    plt.title('Precision/Recall', fontsize=12)
    plt.legend()

    plt.tight_layout()
    history_plot_path = os.path.join(OUTPUT_DIR, f'{model_id}_training_history.png')
    plt.savefig(history_plot_path, dpi=300)
    plt.close()

    # 保存训练历史到CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, f'{model_id}_training_history.csv'), index=False)

    return model, history, model_id


def predict_for_test_regions_unet(model, eq_data, masks, patch_size=64, overlap=32, batch_size=4):
    """使用UNet模型对测试区域进行预测"""
    print("Starting prediction for test regions...")

    num_bands, height, width = eq_data.shape
    test_mask = masks['test']
    
    print(f"Original data shape: eq_data={eq_data.shape}")
    print(f"Test mask shape: {test_mask.shape}")
    
    # 创建预测结果地图
    prediction_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int32)
    
    # 为了处理边界，对原始数据进行填充
    pad = patch_size // 2
    eq_data_padded = global_pad(eq_data, pad)
    _, padded_height, padded_width = eq_data_padded.shape
    
    print(f"Padded data shape: eq_data_padded={eq_data_padded.shape}")
    
    # 移动窗口进行预测
    stride = patch_size - overlap
    
    # 收集测试区域内的窗口
    patches = []
    positions = []
    
    window_count = 0
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # 检查此窗口是否包含测试区域
            window_mask = test_mask[y:y+patch_size, x:x+patch_size]
            if np.any(window_mask):
                # 获取对应的填充数据窗口
                padded_y, padded_x = y + pad, x + pad
                patch = np.zeros((patch_size, patch_size, num_bands), dtype=np.float32)
                
                for b in range(num_bands):
                    patch[:, :, b] = eq_data_padded[b, 
                                                    padded_y:padded_y+patch_size, 
                                                    padded_x:padded_x+patch_size]
                
                patches.append(patch)
                positions.append((y, x))
                
                window_count += 1
                if window_count % 100 == 0:
                    print(f"Collected {window_count} test windows")
    
    # 如果没有测试窗口，提前返回
    if not patches:
        print("No valid windows found in test region")
        return prediction_map
        
    # 转换为数组
    X_test = np.array(patches, dtype=np.float32)
    print(f"Test data shape: {X_test.shape}, total {len(positions)} windows")
    
    # 检查数据形状是否正确
    if len(X_test.shape) != 4:
        raise ValueError(f"Incorrect test data shape: {X_test.shape}, should be (samples, height, width, channels)")
    
    # 确保批量大小不大于样本数
    batch_size = min(batch_size, len(X_test))
    
    # 批量预测
    total_batches = (len(X_test) + batch_size - 1) // batch_size
    all_predictions = []
    
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch = X_test[i:batch_end]
        
        # 打印第一个批次的形状以进行调试
        if i == 0:
            print(f"  First batch shape: {batch.shape}")
            
        batch_preds = model.predict(batch, verbose=0)
        all_predictions.append(batch_preds)
        
        if (i // batch_size) % 10 == 0 or i + batch_size >= len(X_test):
            print(f"  Processed {i//batch_size + 1}/{total_batches} batches")
    
    # 合并预测结果
    predictions = np.vstack(all_predictions)
    print(f"  Prediction results shape: {predictions.shape}")
    
    # 将预测结果映射回原始图像
    for i, (y, x) in enumerate(positions):
        pred = predictions[i].squeeze()  # 去除通道维度
        prediction_map[y:y+patch_size, x:x+patch_size] += pred
        count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # 平均重叠区域
    valid_indices = count_map > 0
    prediction_map[valid_indices] /= count_map[valid_indices]
    
    return prediction_map


def visualize_test_predictions(prediction_map, boundary_data, masks, splits, eq_meta,
                               threshold=0.5, output_path_prefix=None):
    """可视化测试区域的预测结果和真实边界，并与shapefile对比"""
    print("Visualizing test predictions...")
    
    test_mask = masks['test']
    region_mask = masks['region']
    test_regions = splits['test_regions']
    region_stats = splits['region_stats']
    
    # 创建二值预测图
    binary_prediction = prediction_map > threshold
    
    # 仅考虑测试区域
    binary_prediction_test = np.zeros_like(binary_prediction)
    binary_prediction_test[test_mask] = binary_prediction[test_mask]
    
    # 获取真实边界（仅测试区域）
    boundary_data_test = np.zeros_like(boundary_data[0])
    boundary_data_test[test_mask] = boundary_data[0, test_mask]
    
    # 创建可视化图像
    # 1. 全局预测可视化
    plt.figure(figsize=(20, 10))
    
    # 左图：原始边界
    plt.subplot(1, 3, 1)
    masked_boundary = np.ma.masked_where(boundary_data_test == 0, boundary_data_test)
    plt.imshow(np.zeros_like(boundary_data_test), cmap='gray', alpha=0.5)
    plt.imshow(masked_boundary, cmap='hot', alpha=0.8, vmin=0, vmax=1)
    plt.title("Ground Truth Plate Boundaries (Test Regions)", fontsize=14)
    plt.axis('off')
    
    # 中图：预测概率图
    plt.subplot(1, 3, 2)
    prediction_display = np.zeros_like(prediction_map)
    prediction_display[test_mask] = prediction_map[test_mask]
    plt.imshow(prediction_display, cmap='plasma', vmin=0, vmax=1)
    plt.colorbar(label='Boundary Probability')
    plt.title("Prediction Probability Map (Test Regions)", fontsize=14)
    plt.axis('off')
    
    # 右图：二值预测结果
    plt.subplot(1, 3, 3)
    masked_prediction = np.ma.masked_where(binary_prediction_test == 0, binary_prediction_test)
    plt.imshow(np.zeros_like(binary_prediction_test), cmap='gray', alpha=0.5)
    plt.imshow(masked_prediction, cmap='cool', alpha=0.8, vmin=0, vmax=1)
    plt.title(f"Predicted Plate Boundaries (Threshold={threshold})", fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    if output_path_prefix:
        plt.savefig(f"{output_path_prefix}_global_prediction.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 分区域对比可视化
    n_test_regions = len(test_regions)
    n_cols = min(3, n_test_regions)
    n_rows = (n_test_regions + n_cols - 1) // n_cols
    
    plt.figure(figsize=(6*n_cols, 5*n_rows))
    
    for i, region_id in enumerate(test_regions):
        region_info = region_stats[region_id]
        h_start, h_end = region_info['h_start'], region_info['h_end']
        w_start, w_end = region_info['w_start'], region_info['w_end']
        
        # 提取该区域的数据
        region_boundary = boundary_data[0, h_start:h_end, w_start:w_end]
        region_prediction = prediction_map[h_start:h_end, w_start:w_end]
        
        # 计算该区域的性能指标
        region_mask = (region_boundary > 0).astype(int)
        region_pred = (region_prediction > threshold).astype(int)
        
        # 计算准确率、精确率和召回率
        true_pos = np.sum((region_mask == 1) & (region_pred == 1))
        false_pos = np.sum((region_mask == 0) & (region_pred == 1))
        false_neg = np.sum((region_mask == 1) & (region_pred == 0))
        true_neg = np.sum((region_mask == 0) & (region_pred == 0))
        
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg) if (true_pos + true_neg + false_pos + false_neg) > 0 else 0
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 绘制该区域的真实边界和预测结果
        plt.subplot(n_rows, n_cols, i+1)
        
        # 背景为灰色
        plt.imshow(np.zeros_like(region_boundary), cmap='gray', alpha=0.3)
        
        # 叠加真实边界（红色）
        masked_true = np.ma.masked_where(region_boundary == 0, region_boundary)
        plt.imshow(masked_true, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
        
        # 叠加预测边界（蓝色）
        masked_pred = np.ma.masked_where(region_pred == 0, region_pred)
        plt.imshow(masked_pred, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
        
        # 添加区域信息和性能指标
        plt.title(f"Region {region_id}: F1={f1:.3f}, Prec={precision:.3f}, Rec={recall:.3f}", fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    if output_path_prefix:
        plt.savefig(f"{output_path_prefix}_region_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 评估指标可视化 - 为每个测试区域计算评估指标并绘制
    region_metrics = []
    
    for region_id in test_regions:
        region_info = region_stats[region_id]
        h_start, h_end = region_info['h_start'], region_info['h_end']
        w_start, w_end = region_info['w_start'], region_info['w_end']
        
        # 提取该区域的数据
        region_boundary = boundary_data[0, h_start:h_end, w_start:w_end]
        region_prediction = prediction_map[h_start:h_end, w_start:w_end]
        
        # 计算该区域的性能指标
        region_mask = (region_boundary > 0).astype(int)
        region_pred = (region_prediction > threshold).astype(int)
        
        # 计算准确率、精确率和召回率
        true_pos = np.sum((region_mask == 1) & (region_pred == 1))
        false_pos = np.sum((region_mask == 0) & (region_pred == 1))
        false_neg = np.sum((region_mask == 1) & (region_pred == 0))
        true_neg = np.sum((region_mask == 0) & (region_pred == 0))
        
        total = true_pos + true_neg + false_pos + false_neg
        accuracy = (true_pos + true_neg) / total if total > 0 else 0
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算边界像素百分比
        boundary_percentage = region_info['boundary_percentage']
        
        region_metrics.append({
            'region_id': region_id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'boundary_percentage': boundary_percentage
        })
    
    # 转换为DataFrame
    metrics_df = pd.DataFrame(region_metrics)
    
    # 计算全局平均指标
    global_metrics = {
        'accuracy': metrics_df['accuracy'].mean(),
        'precision': metrics_df['precision'].mean(),
        'recall': metrics_df['recall'].mean(),
        'f1': metrics_df['f1'].mean()
    }
    
    # 打印全局指标
    print(f"Global test metrics: Accuracy={global_metrics['accuracy']:.4f}, F1={global_metrics['f1']:.4f}")
    print(f"                     Precision={global_metrics['precision']:.4f}, Recall={global_metrics['recall']:.4f}")
    
    # 绘制指标条形图
    plt.figure(figsize=(10, 6))
    metrics_df.sort_values('f1', ascending=False, inplace=True)
    
    x = np.arange(len(metrics_df))
    width = 0.2
    
    plt.bar(x - 1.5*width, metrics_df['accuracy'], width, label='Accuracy')
    plt.bar(x - 0.5*width, metrics_df['precision'], width, label='Precision')
    plt.bar(x + 0.5*width, metrics_df['recall'], width, label='Recall')
    plt.bar(x + 1.5*width, metrics_df['f1'], width, label='F1 Score')
    
    plt.ylabel('Score')
    plt.title('Performance Metrics by Test Region')
    plt.xticks(x, [f"Region {r}" for r in metrics_df['region_id']])
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if output_path_prefix:
        plt.savefig(f"{output_path_prefix}_metrics.png", dpi=300)
        
        # 保存指标到CSV
        metrics_df.to_csv(f"{output_path_prefix}_metrics.csv", index=False)
    
    # 4. 与shapefile对比可视化
    try:
        # 加载shapefile
        plate_boundaries = gpd.read_file(PLATE_BOUNDARIES_SHP)
        
        # 创建栅格数据的坐标参考
        height, width = prediction_map.shape
        x_coords = np.linspace(eq_meta['transform'][2], 
                              eq_meta['transform'][2] + width * eq_meta['transform'][0], 
                              width)
        y_coords = np.linspace(eq_meta['transform'][5], 
                              eq_meta['transform'][5] + height * eq_meta['transform'][4], 
                              height)
        
        # 创建带坐标的可视化图
        plt.figure(figsize=(15, 12))
        
        # 底图为灰色
        plt.imshow(np.zeros_like(prediction_map), cmap='gray', 
                   extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
                   alpha=0.3)
        
        # 叠加预测边界
        binary_pred_display = np.ma.masked_where(binary_prediction == 0, binary_prediction)
        plt.imshow(binary_pred_display, cmap='plasma', 
                   extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
                   alpha=0.7, vmin=0, vmax=1)
        
        # 添加shapefile边界线
        plate_boundaries.plot(ax=plt.gca(), color='white', linewidth=1, alpha=0.8)
        
        plt.title('Predicted Boundaries vs. Actual Plate Boundaries (Shapefile)', fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # 添加图例
        legend_elements = [
            Line2D([0], [0], color='white', lw=2, label='Plate Boundaries (Shapefile)'),
            Patch(facecolor='purple', alpha=0.7, label='Predicted Boundaries')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        if output_path_prefix:
            plt.savefig(f"{output_path_prefix}_shapefile_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Shapefile comparison visualization completed successfully")
        
    except Exception as e:
        print(f"Error creating shapefile comparison: {str(e)}")
    
    return global_metrics, metrics_df


def main():
    """主函数：加载数据、训练模型并进行预测"""
    start_time = time.time()
    
    try:
        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 1. 加载栅格数据
        print("Loading raster data...")
        eq_data, boundary_data, eq_meta, eq_transform, eq_crs = load_raster_data()
        
        # 2. 定义地质区域
        print("Defining geological regions...")
        region_mask, region_stats = define_geological_regions(
            eq_data, boundary_data, n_regions_h=4, n_regions_w=4
        )
        
        # 3. 创建改进的训练-验证-测试划分
        print("Creating train-validation-test split...")
        masks, splits = create_improved_train_test_split(
            region_mask, region_stats, 
            test_size=0.25, val_size=0.15,
            min_boundary_percentage=0.5
        )
        
        # 可视化数据划分
        split_viz_path = os.path.join(OUTPUT_DIR, "data_split_visualization.png")
        visualize_data_split(masks, splits, split_viz_path)
        
        # 4. 为UNet模型准备数据
        # 减小patch_size和stride，以便获取更多训练样本
        X_train, y_train, X_val, y_val = prepare_data_for_unet(
            eq_data, boundary_data, masks, patch_size=64, stride=32
        )
        
        # 5. 计算正样本权重，用于处理类别不平衡
        pos_ratio = np.mean(y_train)
        pos_weight = 1.0 / pos_ratio if pos_ratio > 0 else 10.0
        print(f"Positive sample ratio: {pos_ratio:.5f}, using positive weight: {pos_weight:.2f}")
        
        # 6. 训练和评估UNet模型
        # 减小batch_size，增加patience
        model, history, model_id = train_and_evaluate_unet(
            X_train=X_train, 
            y_train=y_train, 
            X_val=X_val, 
            y_val=y_val,
            pos_weight=pos_weight,
            epochs=50,
            batch_size=8,  # 减小批量大小
            patience=15    # 增加耐心值
        )
        
        # 7. 对测试区域进行预测
        # 使用与训练相同的patch_size
        prediction_map = predict_for_test_regions_unet(
            model=model, 
            eq_data=eq_data, 
            masks=masks, 
            patch_size=64, 
            overlap=32, 
            batch_size=4
        )
        
        # 8. 可视化测试预测结果
        output_path_prefix = os.path.join(OUTPUT_DIR, f"{model_id}")
        global_metrics, region_metrics = visualize_test_predictions(
            prediction_map=prediction_map, 
            boundary_data=boundary_data, 
            masks=masks, 
            splits=splits, 
            eq_meta=eq_meta,
            threshold=0.5, 
            output_path_prefix=output_path_prefix
        )
        
        # 保存预测图
        prediction_output_path = os.path.join(OUTPUT_DIR, f"{model_id}_prediction.npy")
        np.save(prediction_output_path, prediction_map)
        
        # 打印总运行时间
        total_time = time.time() - start_time
        print(f"Total runtime: {total_time / 60:.2f} minutes")
        
        print(f"Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        total_time = time.time() - start_time
        print(f"Runtime: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    main() 