import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from affine import Affine

# ---------------------- 参数设置 ----------------------
# 输入预处理后的 CSV 数据
csv_path = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2_processing\v2better\earthquake_features_enhanced.csv"
# 输出多通道 tif 文件，与 plate_boundaries 栅格在同一文件夹
output_tif = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2processed\earthquake_features_raster.tif"
# plate_boundaries shapefile 路径（用于获取相同的空间范围）
plate_shp = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2raw\plate\plate_boundaries.shp"

# ---------------------- 读取 plate_boundaries 获取相同范围 ----------------------
print("读取 plate_boundaries.shp 以获取空间范围...")
gdf = gpd.read_file(plate_shp)
data_bounds = gdf.total_bounds
minx, miny, maxx, maxy = data_bounds
print(f"数据范围: 经度 {minx} 到 {maxx}, 纬度 {miny} 到 {maxy}")

# ---------------------- 全局范围与分辨率 ----------------------
resolution = 1.5
width = int((maxx - minx) / resolution)
height = int((maxy - miny) / resolution)

print(f"网格范围：经度({minx}, {maxx}), 纬度({miny}, {maxy})")
print(f"生成栅格的尺寸：宽度 = {width}, 高度 = {height}")

# 构造仿射变换矩阵
transform = Affine.translation(minx, miny) * Affine.scale(resolution, resolution)

# ---------------------- 加载 CSV 数据 ----------------------
print("加载预处理后的 CSV 数据 ...")
df = pd.read_csv(csv_path)
print(f"CSV 中记录数：{len(df)}")

# 检查定位字段
if "centroid_lon" not in df.columns or "centroid_lat" not in df.columns:
    raise ValueError("数据中缺少 centroid_lon 或 centroid_lat 列，用于确定空间位置。")

# ---------------------- 选择用于聚合的特征 ----------------------
# 这里，我们选择 CSV 中所有数值型列，但排除定位用的经纬度
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in numeric_cols if col not in ["centroid_lon", "centroid_lat"]]

print("将聚合以下特征（每个将作为一个波段）：")
print(features)
n_channels = len(features)
print(f"总共 {n_channels} 个特征波段。")

# ---------------------- 初始化聚合数组 ----------------------
channels_sum = np.zeros((n_channels, height, width), dtype=np.float32)
cell_count = np.zeros((height, width), dtype=np.int32)

# ---------------------- 聚合过程 ----------------------
print("开始对每个记录进行网格聚合...")
for idx, row in df.iterrows():
    try:
        lon = float(row["centroid_lon"])
        lat = float(row["centroid_lat"])
    except Exception as e:
        continue

    col_index = int((lon - minx) / resolution)
    row_index = int((lat - miny) / resolution)
    if col_index < 0 or col_index >= width or row_index < 0 or row_index >= height:
        continue

    cell_count[row_index, col_index] += 1
    for i, feat in enumerate(features):
        try:
            val = float(row[feat])
            if np.isnan(val):
                continue
            channels_sum[i, row_index, col_index] += val
        except Exception as e:
            continue

# 计算平均值
for i in range(n_channels):
    mask = cell_count > 0
    channels_sum[i, mask] = channels_sum[i, mask] / cell_count[mask]

print("空间聚合完成。")

# ---------------------- 保存为多通道 GeoTIFF ----------------------
print(f"将聚合数据保存为 {output_tif} ...")
with rasterio.open(
    output_tif,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=n_channels,
    dtype=channels_sum.dtype,
    crs=gdf.crs,  # 使用与 plate_boundaries 相同的坐标系统
    transform=transform,
) as dst:
    for i in range(n_channels):
        dst.write(channels_sum[i, :, :], i+1)

print("多通道栅格数据生成完成。")
