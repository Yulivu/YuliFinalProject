"""
改进版地震断层运动方式识别及板块边界预测模型
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
OUTPUT_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\result\v2c1"
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

    # 应用变换：顺时针旋转180度并水平翻转
    transformed_eq_data = np.zeros_like(eq_data)
    transformed_boundary_data = np.zeros_like(boundary_data)
    
    for i in range(eq_data.shape[0]):
        transformed_eq_data[i] = np.fliplr(np.rot90(eq_data[i], k=2))
    
    for i in range(boundary_data.shape[0]):
        transformed_boundary_data[i] = np.fliplr(np.rot90(boundary_data[i], k=2))

    print(f"地震特征栅格形状: {transformed_eq_data.shape}, 板块边界栅格形状: {transformed_boundary_data.shape}")

    return transformed_eq_data, transformed_boundary_data, eq_meta, eq_transform, eq_crs


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
    print("可视化数据划分...")

    region_mask = masks['region']
    train_regions = splits['train_regions']
    val_regions = splits['val_regions']
    test_regions = splits['test_regions']
    region_stats = splits['region_stats']

    # 创建区域类型掩码（0=训练，1=验证，2=测试）
    split_type_mask = np.zeros_like(region_mask)
    for r in val_regions:
        split_type_mask[region_mask == r] = 1
    for r in test_regions:
        split_type_mask[region_mask == r] = 2

    # 创建自定义色彩图
    colors = ['#4363d8', '#42d4f4', '#f58231']  # 蓝色=训练, 青色=验证, 橙色=测试
    cmap = LinearSegmentedColormap.from_list('split_cmap', colors, N=3)

    # 旋转180度并水平翻转图像
    split_type_mask_transformed = np.rot90(split_type_mask, k=2)  # 旋转180度
    split_type_mask_transformed = np.fliplr(split_type_mask_transformed)  # 水平翻转

    # 创建图像
    plt.figure(figsize=(12, 9))
    plt.imshow(split_type_mask_transformed, cmap=cmap, interpolation='nearest')

    # 添加区域边界和标签
    for region_id, stats in region_stats.items():
        y_start, y_end = stats['h_start'], stats['h_end']
        x_start, x_end = stats['w_start'], stats['w_end']
        
        # 变换区域坐标
        height, width = split_type_mask.shape
        y_start_transformed = height - y_end
        y_end_transformed = height - y_start
        x_start_transformed = width - x_end
        x_end_transformed = width - x_start
        
        # 绘制边界
        plt.plot([x_start_transformed, x_end_transformed, x_end_transformed, x_start_transformed, x_start_transformed],
                 [y_start_transformed, y_start_transformed, y_end_transformed, y_end_transformed, y_start_transformed],
                 'k-', linewidth=0.8, alpha=0.6)
        
        # 确定区域类型和文本颜色
        if region_id in test_regions:
            region_type, text_color = "Test", 'black'
        elif region_id in val_regions:
            region_type, text_color = "Val", 'black'
        else:
            region_type, text_color = "Train", 'white'
            
        # 添加标签 - 计算变换后的中心点
        center_y_transformed = (y_start_transformed + y_end_transformed) // 2
        center_x_transformed = (x_start_transformed + x_end_transformed) // 2
        plt.text(center_x_transformed, center_y_transformed, f"R{region_id}\n{region_type}\n{stats['boundary_percentage']:.1f}%",
                 color=text_color, ha='center', va='center', fontweight='bold', fontsize=9)

    # 添加图例
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

    print(f"数据划分可视化已保存至 {output_path}")


def prepare_windowed_data(eq_data, boundary_data, masks, window_size=7):
    """准备基于窗口的数据，利用每个像素周围的空间信息"""
    print(f"准备窗口大小为{window_size}x{window_size}的数据...")

    num_bands, height, width = eq_data.shape
    half_window = window_size // 2

    # 确定每个集合的样本数量
    train_count = np.sum(masks['train'])
    val_count = np.sum(masks['val'])
    test_count = np.sum(masks['test'])
    print(f"训练/验证/测试样本数量: {train_count}/{val_count}/{test_count}")

    # 初始化窗口化数据
    X_train_windowed = np.zeros((train_count, window_size, window_size, num_bands))
    X_val_windowed = np.zeros((val_count, window_size, window_size, num_bands))
    X_test_windowed = np.zeros((test_count, window_size, window_size, num_bands))

    # 获取训练/验证/测试像素的索引
    train_indices = np.where(masks['train'])
    val_indices = np.where(masks['val'])
    test_indices = np.where(masks['test'])

    # 创建索引映射以便稍后恢复空间位置
    train_index_map = {(y, x): i for i, (y, x) in enumerate(zip(train_indices[0], train_indices[1]))}
    val_index_map = {(y, x): i for i, (y, x) in enumerate(zip(val_indices[0], val_indices[1]))}
    test_index_map = {(y, x): i for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1]))}

    # 使用自定义全球填充来扩展数据，正确处理经纬度边界
    eq_data_padded = global_pad(eq_data, half_window)

    # 处理函数，用于填充窗口
    def process_windows(indices, windowed_data):
        count = len(indices[0])
        for i, (y, x) in enumerate(zip(indices[0], indices[1])):
            if i % 50000 == 0 and i > 0:
                print(f"  已处理 {i}/{count} 个样本")
                
            padded_y, padded_x = y + half_window, x + half_window
            for b in range(num_bands):
                windowed_data[i, :, :, b] = eq_data_padded[b,
                                           padded_y - half_window:padded_y + half_window + 1,
                                           padded_x - half_window:padded_x + half_window + 1]

    # 填充训练、验证和测试窗口
    print("准备训练样本...")
    process_windows(train_indices, X_train_windowed)
    
    print("准备验证样本...")
    process_windows(val_indices, X_val_windowed)
    
    print("准备测试样本...")
    process_windows(test_indices, X_test_windowed)

    # 获取目标值
    y_train = boundary_data[0][train_indices]
    y_val = boundary_data[0][val_indices]
    y_test = boundary_data[0][test_indices]

    # 二值化目标值（将板块边界视为二分类问题）
    y_train_binary = (y_train > 0).astype(np.float32)
    y_val_binary = (y_val > 0).astype(np.float32)
    y_test_binary = (y_test > 0).astype(np.float32)

    # 计算类别权重，用于处理类别不平衡
    pos_weight = np.sum(y_train_binary == 0) / np.sum(y_train_binary == 1)
    print(f"正样本权重: {pos_weight:.2f} (用于处理类别不平衡)")

    # 创建索引与空间位置的映射字典
    indices_mapping = {
        'train': {'indices': train_indices, 'index_map': train_index_map},
        'val': {'indices': val_indices, 'index_map': val_index_map},
        'test': {'indices': test_indices, 'index_map': test_index_map}
    }

    return (X_train_windowed, X_val_windowed, X_test_windowed,
            y_train_binary, y_val_binary, y_test_binary,
            pos_weight, indices_mapping)


def build_improved_model(input_shape, dropout_rate=0.3):
    """构建改进的CNN模型用于板块边界识别"""
    print("构建改进的CNN模型...")
    
    # 设置L2正则化系数
    l2_reg = 1e-4
    reg = tf.keras.regularizers.l2(l2_reg)

    # 模型构建
    inputs = layers.Input(shape=input_shape)
    
    # 第一个卷积块
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # 第二个卷积块
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # 第三个卷积块
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # 全局池化和全连接层
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # 输出层
    outputs = layers.Dense(1, activation='sigmoid')(x)

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
    print("开始训练模型...")
    print(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")

    # 构建模型
    model = build_improved_model(X_train.shape[1:])
    model.summary()

    # 使用类别权重
    class_weights = {0: 1.0, 1: pos_weight}
    print(f"使用类别权重: {class_weights}")

    # 生成时间戳作为模型ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"plate_boundary_model_{timestamp}"
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
    print(f"训练完成，耗时: {training_time:.2f} 秒")

    # 加载最佳模型
    model = tf.keras.models.load_model(model_path)

    # 评估模型
    val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(X_val, y_val, verbose=0)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
    
    print(f"验证集性能指标: 损失={val_loss:.4f}, 准确率={val_acc:.4f}, F1={val_f1:.4f}")

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
    history_plot_path = os.path.join(OUTPUT_DIR, f'{model_id}_training_history.png')
    plt.savefig(history_plot_path, dpi=300)
    plt.close()

    # 保存训练历史到CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, f'{model_id}_training_history.csv'), index=False)

    return model, history, model_id


def predict_for_test_regions(model, eq_data, masks, splits, window_size=7, batch_size=256):
    """仅对测试区域进行预测"""
    print("开始对测试区域进行预测...")

    num_bands, height, width = eq_data.shape
    half_window = window_size // 2
    test_mask = masks['test']

    # 只对测试区域的像素进行预测
    test_indices = np.where(test_mask)
    num_test_pixels = len(test_indices[0])
    print(f"测试区域: {num_test_pixels}个像素")

    # 使用自定义全球填充，正确处理经纬度边界
    eq_data_padded = global_pad(eq_data, half_window)

    # 初始化预测数组
    prediction_map = np.zeros((height, width), dtype=np.float32)

    # 创建测试区域的窗口化数据
    X_test_windowed = np.zeros((num_test_pixels, window_size, window_size, num_bands))

    # 收集测试区域像素的窗口化数据
    for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1])):
        if i % 50000 == 0 and i > 0:
            print(f"  已处理 {i}/{num_test_pixels} 个测试像素")

        padded_y, padded_x = y + half_window, x + half_window
        for b in range(num_bands):
            X_test_windowed[i, :, :, b] = eq_data_padded[b,
                                         padded_y - half_window:padded_y + half_window + 1,
                                         padded_x - half_window:padded_x + half_window + 1]

    # 使用批处理进行预测
    print("对测试区域进行批处理预测...")
    predictions = []
    total_batches = math.ceil(num_test_pixels / batch_size)
    
    for i in range(0, num_test_pixels, batch_size):
        batch_end = min(i + batch_size, num_test_pixels)
        batch_predictions = model.predict(X_test_windowed[i:batch_end], verbose=0)
        predictions.append(batch_predictions)
        if (i // batch_size) % 20 == 0:
            print(f"  已完成 {i // batch_size}/{total_batches} 批次")

    # 合并所有批次的预测并填入预测图
    all_predictions = np.vstack(predictions)
    for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1])):
        prediction_map[y, x] = all_predictions[i, 0]

    # 注意：不再需要旋转和翻转，因为输入数据已经完成了变换

    print("测试区域预测完成")
    return prediction_map


def visualize_test_predictions(prediction_map, boundary_data, masks, splits, eq_meta,
                               threshold=0.5, output_path_prefix=None):
    """可视化测试区域的预测结果"""
    print("可视化测试区域预测结果...")

    if output_path_prefix is None:
        output_path_prefix = os.path.join(OUTPUT_DIR, "test_prediction")

    # 获取测试掩码和区域信息
    test_mask = masks['test']
    test_regions = splits['test_regions']
    region_mask = masks['region']
    region_stats = splits['region_stats']

    # 二值化预测
    binary_prediction = (prediction_map >= threshold).astype(np.uint8)

    # 后处理：应用形态学操作清理噪声
    binary_prediction = ndimage.binary_opening(binary_prediction, structure=np.ones((3, 3)))
    binary_prediction = ndimage.binary_closing(binary_prediction, structure=np.ones((3, 3)))

    # 提取轮廓
    contours = measure.find_contours(prediction_map, threshold)

    # 计算测试区域的性能指标
    test_indices = np.where(test_mask)
    test_true = (boundary_data[0][test_indices] > 0).astype(np.int32)
    test_pred = (prediction_map[test_indices] >= threshold).astype(np.int32)

    # 计算精度、召回率和F1
    true_positives = np.sum((test_true == 1) & (test_pred == 1))
    false_positives = np.sum((test_true == 0) & (test_pred == 1))
    false_negatives = np.sum((test_true == 1) & (test_pred == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 为测试区域创建评估报告
    report = f"""测试区域评估报告:
    -----------------
    测试区域数量: {len(test_regions)}
    测试像素总数: {np.sum(test_mask)}
    边界像素占比: {np.sum(test_true) / np.sum(test_mask) * 100:.2f}%
    
    性能指标:
    - 精确率: {precision:.4f}
    - 召回率: {recall:.4f}
    - F1分数: {f1:.4f}
    """

    # 将报告保存到文件
    with open(f"{output_path_prefix}_report.txt", 'w') as f:
        f.write(report)

    print(report)

    # 创建自定义颜色映射
    boundary_cmap = LinearSegmentedColormap.from_list('boundary_cmap', ['white', 'crimson'], N=256)
    prediction_cmap = LinearSegmentedColormap.from_list('prediction_cmap', 
                                                        ['navy', 'deepskyblue', 'gold'], N=256)

    # 获取形状尺寸 - 注意：由于数据已经在load_raster_data中变换，这里不需要再做变换
    height, width = test_mask.shape

    # 创建整体测试区域预测对比图
    plt.figure(figsize=(16, 12))
    
    # 只显示测试区域
    test_prediction = np.ma.masked_where(~test_mask, prediction_map)
    test_original = np.ma.masked_where(~test_mask, boundary_data[0])
    
    # 创建背景 - 测试区域显示为浅灰色
    background = np.zeros_like(test_mask, dtype=np.float32)
    background[test_mask] = 0.1
    plt.imshow(background, cmap='gray', alpha=0.3)

    # 显示原始边界 - 红色
    plt.imshow(test_original, cmap=boundary_cmap, alpha=0.7, vmin=0, vmax=1)

    # 绘制预测边界轮廓 - 蓝色
    for contour in contours:
        # 检查轮廓是否在测试区域内
        contour_points = [(int(p[0]), int(p[1])) for p in contour 
                          if 0 <= int(p[0]) < test_mask.shape[0] and 0 <= int(p[1]) < test_mask.shape[1]]
        if any(test_mask[y, x] for y, x in contour_points):
            plt.plot(contour[:, 1], contour[:, 0], 'b-', linewidth=1)

    # 绘制每个测试区域的边界
    for region_id in test_regions:
        stats = region_stats[region_id]
        y_start, y_end = stats['h_start'], stats['h_end']
        x_start, x_end = stats['w_start'], stats['w_end']
        
        plt.plot([x_start, x_end, x_end, x_start, x_start],
                 [y_start, y_start, y_end, y_end, y_start],
                 'y-', linewidth=2, alpha=0.7)

    plt.title(f'Test Region Evaluation - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}', 
              fontsize=14)

    # 添加图例
    legend_elements = [
        Patch(facecolor='crimson', alpha=0.7, label='Original Boundary'),
        Line2D([0], [0], color='blue', lw=2, label='Predicted Boundary'),
        Patch(facecolor='yellow', alpha=0.3, label='Test Region'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_overview.png", dpi=300)
    plt.close()

    # 创建预测热图
    plt.figure(figsize=(16, 12))
    plt.imshow(background, cmap='gray', alpha=0.3)
    plt.imshow(test_prediction, cmap=prediction_cmap, alpha=0.7, vmin=0, vmax=1)
    
    # 显示原始边界轮廓
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
    
    # 为每个测试区域创建单独的可视化
    print(f"创建 {len(test_regions)} 个测试区域的单独可视化...")
    for region_id in test_regions:
        stats = region_stats[region_id]
        y_start, y_end = stats['h_start'], stats['h_end']
        x_start, x_end = stats['w_start'], stats['w_end']
        
        # 创建区域掩码
        region_mask_local = np.zeros_like(test_mask, dtype=bool)
        region_mask_local[y_start:y_end, x_start:x_end] = True
        region_test_mask = region_mask_local & test_mask
        
        if np.sum(region_test_mask) == 0:
            print(f"  区域 {region_id} 不在测试集中，跳过可视化")
            continue
            
        # 计算区域性能指标
        region_indices = np.where(region_test_mask)
        region_true = (boundary_data[0][region_indices] > 0).astype(np.int32)
        region_pred = (prediction_map[region_indices] >= threshold).astype(np.int32)
        
        region_tp = np.sum((region_true == 1) & (region_pred == 1))
        region_fp = np.sum((region_true == 0) & (region_pred == 1))
        region_fn = np.sum((region_true == 1) & (region_pred == 0))
        
        region_precision = region_tp / (region_tp + region_fp) if (region_tp + region_fp) > 0 else 0
        region_recall = region_tp / (region_tp + region_fn) if (region_tp + region_fn) > 0 else 0
        region_f1 = 2 * region_precision * region_recall / (region_precision + region_recall) if (region_precision + region_recall) > 0 else 0
        
        # 创建区域可视化 - 原始边界与预测对比
        plt.figure(figsize=(10, 8))
        
        # 裁剪区域数据
        region_data = boundary_data[0][y_start:y_end, x_start:x_end]
        region_pred_data = prediction_map[y_start:y_end, x_start:x_end]
        region_test_area = test_mask[y_start:y_end, x_start:x_end]
        
        # 创建掩码数据
        region_original = np.ma.masked_where(~region_test_area, region_data)
        
        # 显示区域原始边界
        plt.imshow(region_original, cmap=boundary_cmap, alpha=0.7, vmin=0, vmax=1)
        
        # 获取区域内的轮廓
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
        
        # 创建区域热图
        plt.figure(figsize=(10, 8))
        region_pred_masked = np.ma.masked_where(~region_test_area, region_pred_data)
        plt.imshow(region_pred_masked, cmap=prediction_cmap, vmin=0, vmax=1)
        
        # 显示原始边界轮廓
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
        plt.title(f'Region {region_id} - Prediction Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_region_{region_id}_heatmap.png", dpi=300)
        plt.close()

    # 保存预测结果为GeoTIFF
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
    """主函数"""
    print("开始改进版板块边界预测...")

    # 创建具有时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # 加载数据
    eq_data, boundary_data, eq_meta, eq_transform, eq_crs = load_raster_data()

    # 定义地质区域
    region_mask, region_stats = define_geological_regions(eq_data, boundary_data)

    # 划分数据集
    masks, splits = create_improved_train_test_split(region_mask, region_stats)

    # 可视化数据划分
    visualize_data_split(masks, splits, os.path.join(run_output_dir, "data_split.png"))

    # 准备窗口化数据
    window_size = 9  # 使用更大的窗口捕获更多空间上下文
    X_train, X_val, X_test, y_train, y_val, y_test, pos_weight, indices_mapping = prepare_windowed_data(
        eq_data, boundary_data, masks, window_size)

    # 训练模型
    model, history, model_id = train_and_evaluate(
        X_train, X_val, y_train, y_val,
        pos_weight=pos_weight,
        epochs=100,  # 增加轮次，依赖早停
        batch_size=128,  # 适当增加批次大小
        patience=15  # 更长的耐心值
    )

    # 仅对测试区域进行预测
    prediction_map = predict_for_test_regions(
        model, eq_data, masks, splits, window_size=window_size)

    # 可视化测试区域的预测结果
    output_prefix = os.path.join(run_output_dir, f"test_prediction_{model_id}")
    metrics = visualize_test_predictions(
        prediction_map, boundary_data, masks, splits, eq_meta,
        threshold=0.5, output_path_prefix=output_prefix
    )

    # 保存配置信息
    config = {
        'timestamp': timestamp,
        'window_size': window_size,
        'train_regions': len(splits['train_regions']),
        'val_regions': len(splits['val_regions']),
        'test_regions': len(splits['test_regions']),
        'metrics': metrics,
        'model_id': model_id
    }

    with open(os.path.join(run_output_dir, 'config.txt'), 'w') as f:
        f.write(str(config))

    print(f"板块边界预测完成！F1分数: {metrics['f1']:.4f}")
    return model, prediction_map, metrics


if __name__ == "__main__":
    main()