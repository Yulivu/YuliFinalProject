"""
地震断层运动方式识别及板块边界预测模型
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import LineString, Point
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from skimage import measure
from scipy import ndimage

# 设置路径
DATA_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2processed"
MODEL_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\src\model"
OUTPUT_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\result\v2c"
PLATE_BOUNDARIES_SHP = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2raw\plate\plate_boundaries.shp"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_raster_data():
    """加载栅格数据"""
    print("加载栅格数据...")

    # 加载地震特征栅格
    earthquake_raster_path = os.path.join(DATA_DIR, "earthquake_features_raster.tif")
    with rasterio.open(earthquake_raster_path) as eq_src:
        eq_data = eq_src.read()
        eq_meta = eq_src.meta
        eq_transform = eq_src.transform

    # 加载板块边界栅格
    boundaries_raster_path = os.path.join(DATA_DIR, "plate_boundaries_raster.tif")
    with rasterio.open(boundaries_raster_path) as bound_src:
        boundary_data = bound_src.read()
        boundary_meta = bound_src.meta

    print(f"地震特征栅格形状: {eq_data.shape}")
    print(f"板块边界栅格形状: {boundary_data.shape}")

    # 调整尺寸以确保两个栅格具有相同的尺寸
    min_height = min(eq_data.shape[1], boundary_data.shape[1])
    min_width = min(eq_data.shape[2], boundary_data.shape[2])

    eq_data = eq_data[:, :min_height, :min_width]
    boundary_data = boundary_data[:, :min_height, :min_width]

    print(f"调整后的地震特征栅格形状: {eq_data.shape}")
    print(f"调整后的板块边界栅格形状: {boundary_data.shape}")

    return eq_data, boundary_data, eq_meta, eq_transform

def create_geographic_train_test_split(eq_data, boundary_data, test_size=0.2, val_size=0.1, random_state=42):
    """
    创建具有地理意义的训练-测试划分
    采用区域块划分而非随机像素划分，确保地理连续性
    """
    print("执行具有地理意义的数据集划分...")

    # 获取数据形状
    num_bands, height, width = eq_data.shape

    # 创建地理掩码 - 将地图分成几个大区域
    # 我们使用分块而不是完全随机的方式，以保持地理连贯性

    # 创建一个均匀的网格，但加入随机偏移
    np.random.seed(random_state)

    # 创建4x4的区域网格(可根据需要调整)
    n_regions_h, n_regions_w = 4, 4
    region_height = height // n_regions_h
    region_width = width // n_regions_w

    # 创建区域掩码
    region_mask = np.zeros((height, width), dtype=np.int32)
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
            region_id += 1

    # 总区域数
    total_regions = region_id

    # 随机选择一些区域作为测试集
    test_regions = np.random.choice(
        range(total_regions),
        size=int(total_regions * test_size),
        replace=False
    )

    # 从剩余区域中选择验证集
    remaining_regions = np.array([r for r in range(total_regions) if r not in test_regions])
    val_regions = np.random.choice(
        remaining_regions,
        size=int(total_regions * val_size),
        replace=False
    )

    # 创建训练/验证/测试掩码
    train_mask = np.ones((height, width), dtype=bool)
    val_mask = np.zeros((height, width), dtype=bool)
    test_mask = np.zeros((height, width), dtype=bool)

    for r in test_regions:
        test_mask[region_mask == r] = True
        train_mask[region_mask == r] = False

    for r in val_regions:
        val_mask[region_mask == r] = True
        train_mask[region_mask == r] = False

    # 创建训练/验证/测试数据集
    X_train = np.zeros((np.sum(train_mask), num_bands))
    X_val = np.zeros((np.sum(val_mask), num_bands))
    X_test = np.zeros((np.sum(test_mask), num_bands))

    y_train = np.zeros((np.sum(train_mask),))
    y_val = np.zeros((np.sum(val_mask),))
    y_test = np.zeros((np.sum(test_mask),))

    # 提取有效像素
    train_indices = np.where(train_mask)
    val_indices = np.where(val_mask)
    test_indices = np.where(test_mask)

    # 填充数据
    for b in range(num_bands):
        X_train[:, b] = eq_data[b][train_indices]
        X_val[:, b] = eq_data[b][val_indices]
        X_test[:, b] = eq_data[b][test_indices]

    y_train = boundary_data[0][train_indices]
    y_val = boundary_data[0][val_indices]
    y_test = boundary_data[0][test_indices]

    # 保持掩码供后续可视化使用
    masks = {
        'train': train_mask,
        'val': val_mask,
        'test': test_mask,
        'region': region_mask
    }

    # 保存一份区域划分的可视化图
    plt.figure(figsize=(12, 8))
    cmap = plt.cm.get_cmap('viridis', total_regions)
    plt.imshow(region_mask, cmap=cmap)
    plt.colorbar(label='Region ID')
    plt.title('Geographic Region Division')

    # 标记测试区域
    for r in test_regions:
        y, x = np.where(region_mask == r)
        center_y, center_x = int(np.mean(y)), int(np.mean(x))
        plt.text(center_x, center_y, 'Test', color='white',
                 ha='center', va='center', fontweight='bold')

    # 标记验证区域
    for r in val_regions:
        y, x = np.where(region_mask == r)
        center_y, center_x = int(np.mean(y)), int(np.mean(x))
        plt.text(center_x, center_y, 'Val', color='yellow',
                 ha='center', va='center', fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, 'geographic_split.png'))
    plt.close()

    # 打印数据集大小
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    # 检查每个集合中包含的边界像素百分比
    train_boundary_pct = np.mean(y_train > 0) * 100
    val_boundary_pct = np.mean(y_val > 0) * 100
    test_boundary_pct = np.mean(y_test > 0) * 100

    print(f"训练集中板块边界像素占比: {train_boundary_pct:.2f}%")
    print(f"验证集中板块边界像素占比: {val_boundary_pct:.2f}%")
    print(f"测试集中板块边界像素占比: {test_boundary_pct:.2f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test, masks

def prepare_windowed_data(eq_data, boundary_data, masks, window_size=7):
    """
    准备基于窗口的数据，利用每个像素周围的空间信息
    """
    print(f"准备窗口大小为{window_size}x{window_size}的数据...")

    num_bands, height, width = eq_data.shape
    half_window = window_size // 2

    # 确定每个集合的样本数量
    train_count = np.sum(masks['train'])
    val_count = np.sum(masks['val'])
    test_count = np.sum(masks['test'])

    # 初始化窗口化数据
    X_train_windowed = np.zeros((train_count, window_size, window_size, num_bands))
    X_val_windowed = np.zeros((val_count, window_size, window_size, num_bands))
    X_test_windowed = np.zeros((test_count, window_size, window_size, num_bands))

    # 获取训练/验证/测试像素的索引
    train_indices = np.where(masks['train'])
    val_indices = np.where(masks['val'])
    test_indices = np.where(masks['test'])

    # 填充训练窗口
    for i, (y, x) in enumerate(zip(train_indices[0], train_indices[1])):
        for b in range(num_bands):
            # 提取窗口，处理边界
            window = np.zeros((window_size, window_size))
            for wy in range(window_size):
                for wx in range(window_size):
                    # 计算原始图像中的位置
                    img_y = y + (wy - half_window)
                    img_x = x + (wx - half_window)

                    # 检查边界
                    if 0 <= img_y < height and 0 <= img_x < width:
                        window[wy, wx] = eq_data[b, img_y, img_x]

            X_train_windowed[i, :, :, b] = window

    # 填充验证窗口
    for i, (y, x) in enumerate(zip(val_indices[0], val_indices[1])):
        for b in range(num_bands):
            window = np.zeros((window_size, window_size))
            for wy in range(window_size):
                for wx in range(window_size):
                    img_y = y + (wy - half_window)
                    img_x = x + (wx - half_window)

                    if 0 <= img_y < height and 0 <= img_x < width:
                        window[wy, wx] = eq_data[b, img_y, img_x]

            X_val_windowed[i, :, :, b] = window

    # 填充测试窗口
    for i, (y, x) in enumerate(zip(test_indices[0], test_indices[1])):
        for b in range(num_bands):
            window = np.zeros((window_size, window_size))
            for wy in range(window_size):
                for wx in range(window_size):
                    img_y = y + (wy - half_window)
                    img_x = x + (wx - half_window)

                    if 0 <= img_y < height and 0 <= img_x < width:
                        window[wy, wx] = eq_data[b, img_y, img_x]

            X_test_windowed[i, :, :, b] = window

    # 获取目标值
    y_train = boundary_data[0][train_indices]
    y_val = boundary_data[0][val_indices]
    y_test = boundary_data[0][test_indices]

    # 二值化目标值（将板块边界视为二分类问题）
    y_train_binary = (y_train > 0).astype(np.float32)
    y_val_binary = (y_val > 0).astype(np.float32)
    y_test_binary = (y_test > 0).astype(np.float32)

    print(f"窗口化训练集: {X_train_windowed.shape}")
    print(f"窗口化验证集: {X_val_windowed.shape}")
    print(f"窗口化测试集: {X_test_windowed.shape}")

    return (X_train_windowed, X_val_windowed, X_test_windowed,
            y_train_binary, y_val_binary, y_test_binary)

def build_model(input_shape, dropout_rate=0.3):
    """构建CNN模型用于板块边界识别"""

    inputs = layers.Input(shape=input_shape)

    # 第一个卷积块
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # 第二个卷积块
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # 全局池化和全连接层
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # 输出层 - 二分类
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model

def train_and_evaluate(X_train, X_val, y_train, y_val, epochs=30, batch_size=64):
    """训练和评估模型"""
    print("开始训练模型...")

    # 检查数据
    print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"验证集形状: {X_val.shape}, 标签形状: {y_val.shape}")

    # 构建模型
    model = build_model(X_train.shape[1:])
    model.summary()

    # 保存模型结构图
    try:
        plot_model(model, to_file=os.path.join(OUTPUT_DIR, 'model_architecture.png'), show_shapes=True)
        print(f"模型结构图已保存至 {os.path.join(OUTPUT_DIR, 'model_architecture.png')}")
    except Exception as e:
        print(f"无法保存模型结构图: {e}")

    # 设置回调函数
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        )
    ]

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )

    # 绘制训练历史
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()

    # 加载最佳模型
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.h5'))

    return model

def predict_and_visualize(model, eq_data, boundary_data, masks, eq_meta, eq_transform, window_size=7):
    """使用模型进行预测并可视化结果"""
    print("开始预测并可视化...")

    num_bands, height, width = eq_data.shape
    half_window = window_size // 2

    # 创建空的预测图
    prediction_map = np.zeros((height, width))

    # 设置批处理大小
    batch_size = 1024

    # 为所有像素创建滑动窗口
    all_windows = []
    all_indices = []

    for y in range(height):
        for x in range(width):
            window = np.zeros((window_size, window_size, num_bands))
            for b in range(num_bands):
                for wy in range(window_size):
                    for wx in range(window_size):
                        img_y = y + (wy - half_window)
                        img_x = x + (wx - half_window)

                        if 0 <= img_y < height and 0 <= img_x < width:
                            window[wy, wx, b] = eq_data[b, img_y, img_x]

            all_windows.append(window)
            all_indices.append((y, x))

            # 当积累足够的窗口，进行批处理预测
            if len(all_windows) >= batch_size:
                batch_windows = np.array(all_windows)
                batch_preds = model.predict(batch_windows, verbose=0)

                # 填充预测图
                for (y_idx, x_idx), pred in zip(all_indices, batch_preds):
                    prediction_map[y_idx, x_idx] = pred[0]

                # 清空批次
                all_windows = []
                all_indices = []
                print(f"已处理 {y*width + x + 1}/{height*width} 像素", end='\r')

    # 处理剩余的窗口
    if all_windows:
        batch_windows = np.array(all_windows)
        batch_preds = model.predict(batch_windows, verbose=0)

        for (y_idx, x_idx), pred in zip(all_indices, batch_preds):
            prediction_map[y_idx, x_idx] = pred[0]

    print("\n预测完成")

    # 应用阈值获取二值预测
    threshold = 0.5
    binary_prediction = (prediction_map >= threshold).astype(np.uint8)

    # 后处理：应用形态学操作清理噪声
    binary_prediction = ndimage.binary_opening(binary_prediction, structure=np.ones((3, 3)))
    binary_prediction = ndimage.binary_closing(binary_prediction, structure=np.ones((3, 3)))

    # 提取轮廓作为线条
    contours = measure.find_contours(prediction_map, threshold)

    # 加载原始板块边界shapefile进行对比
    gdf_boundaries = gpd.read_file(PLATE_BOUNDARIES_SHP)

    # 创建可视化图
    plt.figure(figsize=(18, 12))

    # 显示原始板块边界栅格
    plt.subplot(2, 2, 1)
    plt.imshow(boundary_data[0], cmap='Reds', vmin=0, vmax=1)
    plt.title('Original Plate Boundaries')
    plt.colorbar(label='Boundary Value')

    # 显示模型预测概率
    plt.subplot(2, 2, 2)
    plt.imshow(prediction_map, cmap='plasma', vmin=0, vmax=1)
    plt.title('Predicted Boundary Probability')
    plt.colorbar(label='Probability')

    # 显示模型二值预测
    plt.subplot(2, 2, 3)
    plt.imshow(binary_prediction, cmap='binary')
    plt.title('Binary Prediction (Threshold = 0.5)')

    # 对比可视化：原始边界和预测
    plt.subplot(2, 2, 4)

    # 创建底图
    plt.imshow(np.zeros_like(binary_prediction), cmap='binary', alpha=0.1)

    # 显示训练、验证和测试区域
    train_mask_vis = np.ma.masked_where(~masks['train'], np.ones_like(masks['train']))
    val_mask_vis = np.ma.masked_where(~masks['val'], np.ones_like(masks['val']))
    test_mask_vis = np.ma.masked_where(~masks['test'], np.ones_like(masks['test']))

    plt.imshow(train_mask_vis, cmap='Blues', alpha=0.2)
    plt.imshow(val_mask_vis, cmap='Greens', alpha=0.2)
    plt.imshow(test_mask_vis, cmap='Oranges', alpha=0.2)

    # 绘制原始边界
    original_boundary = np.ma.masked_where(boundary_data[0] == 0, boundary_data[0])
    plt.imshow(original_boundary, cmap='Reds', alpha=0.7)

    # 绘制预测边界轮廓
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], 'b-', linewidth=1)

    plt.title('Comparison: Original (Red) vs Predicted (Blue)')

    # 添加图例
    legend_elements = [
        Patch(facecolor='blue', alpha=0.2, label='Training Area'),
        Patch(facecolor='green', alpha=0.2, label='Validation Area'),
        Patch(facecolor='orange', alpha=0.2, label='Testing Area'),
        Patch(facecolor='red', alpha=0.7, label='Original Boundaries'),
        plt.Line2D([0], [0], color='blue', lw=1, label='Predicted Boundaries')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_comparison.png'), dpi=300)
    plt.close()

    # 保存预测结果为GeoTIFF
    pred_meta = eq_meta.copy()
    pred_meta.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': None,
        'width': width,
        'height': height
    })

    with rasterio.open(os.path.join(OUTPUT_DIR, 'predicted_boundaries.tif'), 'w', **pred_meta) as dst:
        dst.write(prediction_map.astype(np.float32), 1)

    # 为测试区域创建单独的评估图
    plt.figure(figsize=(12, 10))

    # 只显示测试区域
    test_prediction = np.ma.masked_where(~masks['test'], prediction_map)
    test_original = np.ma.masked_where(~masks['test'], boundary_data[0])

    # 计算测试区域的性能指标
    test_indices = np.where(masks['test'])
    test_true = (boundary_data[0][test_indices] > 0).astype(np.int32)
    test_pred = (prediction_map[test_indices] >= threshold).astype(np.int32)

    # 计算精度、召回率和F1
    true_positives = np.sum((test_true == 1) & (test_pred == 1))
    false_positives = np.sum((test_true == 0) & (test_pred == 1))
    false_negatives = np.sum((test_true == 1) & (test_pred == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 显示测试区域对比
    plt.imshow(np.zeros_like(binary_prediction), cmap='binary', alpha=0.1)
    plt.imshow(test_mask_vis, cmap='Oranges', alpha=0.3)
    plt.imshow(test_original, cmap='Reds', alpha=0.7)

    # 在测试区域绘制预测边界
    for contour in contours:
        # 检查轮廓是否在测试区域内
        contour_in_test = [masks['test'][int(p[0]), int(p[1])]
                           for p in contour if 0 <= int(p[0]) < height and 0 <= int(p[1]) < width]
        if any(contour_in_test):  # 如果至少部分在测试区域
            plt.plot(contour[:, 1], contour[:, 0], 'b-', linewidth=1)

    plt.title(f'Test Area Evaluation\nPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')

    # 添加图例
    legend_elements = [
        Patch(facecolor='orange', alpha=0.3, label='Test Area'),
        Patch(facecolor='red', alpha=0.7, label='Original Boundaries'),
        plt.Line2D([0], [0], color='blue', lw=1, label='Predicted Boundaries')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_area_evaluation.png'), dpi=300)
    plt.close()

    # 将结果打印到控制台和文件
    results = (
        f"模型评估结果 (测试集):\n"
        f"精确率 (Precision): {precision:.4f}\n"
        f"召回率 (Recall): {recall:.4f}\n"
        f"F1分数: {f1:.4f}\n"
    )

    print(results)

    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.txt'), 'w') as f:
        f.write(results)

    return prediction_map, binary_prediction


def main():
    """主函数"""
    print("开始板块边界预测...")

    # 加载数据
    eq_data, boundary_data, eq_meta, eq_transform = load_raster_data()

    # 创建具有地理意义的训练-测试划分
    X_train, X_val, X_test, y_train, y_val, y_test, masks = create_geographic_train_test_split(
        eq_data, boundary_data)

    # 准备窗口化数据
    window_size = 7  # 可根据需要调整
    X_train_win, X_val_win, X_test_win, y_train_bin, y_val_bin, y_test_bin = prepare_windowed_data(
        eq_data, boundary_data, masks, window_size)

    # 训练模型
    model = train_and_evaluate(X_train_win, X_val_win, y_train_bin, y_val_bin)

    # 预测和可视化
    prediction_map, binary_prediction = predict_and_visualize(
        model, eq_data, boundary_data, masks, eq_meta, eq_transform, window_size)

    print("板块边界预测完成！结果已保存至输出目录。")


if __name__ == "__main__":
    # 设置随机种子以确保结果可重复
    np.random.seed(42)
    tf.random.set_seed(42)

    # 设置TensorFlow内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"找到 {len(gpus)} 个 GPU，已设置内存增长")
        except RuntimeError as e:
            print(f"GPU设置错误: {e}")

    # 执行主函数
    main()