"""
v2_3_1.py - 板块边界类型编码与数据优化脚本

功能：
1. 读取原始v2_3处理后的earthquake_features_processed.csv文件
2. 添加边界类型编码(0-5)，保留原始nearest_boundary_type文本字段
3. 添加边界三大类分组(convergent/divergent/transform)
4. 处理eigenvector字段，转换为数值格式以便栅格化，并进行归一化
5. 删除冗余字段，优化数据结构
6. 保存处理后的文件，使其能更好地用于边界类型分析

边界类型编码方案：
- 0 = subduction zone (汇聚型)
- 1 = collision zone (汇聚型)
- 2 = spreading center (发散型)
- 3 = extension zone (发散型)
- 4 = dextral transform (转换型)
- 5 = sinistral transform (转换型)
- -1 = inferred 或其他未识别类型
"""

import os
import numpy as np
import pandas as pd
import rasterio

# ---------------------- 参数设置 ----------------------
input_csv = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2_processing\v2better\earthquake_features_processed.csv"
output_csv = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2_processing\v2better\earthquake_features_enhanced.csv"

# ---------------------- 读取数据 ----------------------
print("读取v2_3处理后的数据...")
df = pd.read_csv(input_csv)
print(f"记录数：{len(df)}")

# ---------------------- 处理eigenvector字段 ----------------------
# 定义归一化函数
def min_max(series):
    if series.max() == series.min():
        return series
    return (series - series.min()) / (series.max() - series.min())

# 将eigenvector字段转换为数值格式以便栅格化
vector_cols = ["eigenvector1", "eigenvector2", "eigenvector3"]
for col in vector_cols:
    if col in df.columns:
        try:
            print(f"处理{col}字段...")
            # 尝试解析元组字符串
            df[f"{col}_val"] = df[col].apply(
                lambda x: float(str(x).replace("(", "").replace(")", "").split(",")[0])
                if pd.notnull(x) else np.nan)
            df[f"{col}_plunge"] = df[col].apply(
                lambda x: float(str(x).replace("(", "").replace(")", "").split(",")[1])
                if pd.notnull(x) else np.nan)
            df[f"{col}_azimuth"] = df[col].apply(
                lambda x: float(str(x).replace("(", "").replace(")", "").split(",")[2])
                if pd.notnull(x) else np.nan)

            # 归一化新字段
            df[f"{col}_val_norm"] = min_max(df[f"{col}_val"])
            df[f"{col}_plunge_norm"] = min_max(df[f"{col}_plunge"])
            df[f"{col}_azimuth_norm"] = min_max(df[f"{col}_azimuth"])

            # 删除原始字符串列和未归一化的数值列
            df.drop(columns=[col, f"{col}_val", f"{col}_plunge", f"{col}_azimuth"], inplace=True)
            print(f"{col}已成功转换为3个归一化数值字段")
        except Exception as e:
            print(f"处理向量列 {col} 时出错: {e}")

# ---------------------- 添加边界类型编码 ----------------------
# 将nearest_boundary_type转换为详细编码
if "nearest_boundary_type" in df.columns:
    print("添加边界类型编码...")
    # 定义边界类型映射
    boundary_mapping = {
        "subduction zone": 0,    # 汇聚型
        "collision zone": 1,     # 汇聚型
        "spreading center": 2,   # 发散型
        "extension zone": 3,     # 发散型
        "dextral transform": 4,  # 转换型
        "sinistral transform": 5, # 转换型
        "inferred": -1,          # 未确定类型
    }

    # 创建新的编码列
    df["boundary_type_code"] = df["nearest_boundary_type"].apply(
        lambda x: boundary_mapping.get(str(x).lower(), -1) if pd.notnull(x) else -1)

    unique_types = df["nearest_boundary_type"].unique()
    mapped_types = list(boundary_mapping.keys())
    unmapped = [t for t in unique_types if str(t).lower() not in mapped_types and pd.notnull(t)]

    if unmapped:
        print(f"警告：发现未映射的边界类型：{unmapped}，已编码为-1")

    # 打印映射信息
    print("边界类型编码映射：")
    for code, type_name in sorted({v: k for k, v in boundary_mapping.items()}.items()):
        print(f"  {code} = {type_name}")

    # 添加三大类分组字段
    df["boundary_category"] = df["boundary_type_code"].apply(
        lambda x: "convergent" if x in [0, 1] else
                 "divergent" if x in [2, 3] else
                 "transform" if x in [4, 5] else np.nan)
else:
    print("错误：数据中找不到nearest_boundary_type字段！")

# ---------------------- 删除冗余字段 ----------------------
print("检查并删除冗余字段...")

# 确认要保留的关键字段
keep_fields = [
    # 坐标字段
    "centroid_lat", "centroid_lon", "hypocenter_lat", "hypocenter_lon",
    # 板块和边界信息
    "nearest_boundary_type", "boundary_type_code", "boundary_category",
    "plate", "plate_type_code", "crust_type_code",
    # 所有归一化字段(_norm结尾)
]

# 获取所有已归一化的字段（保留）
norm_fields = [col for col in df.columns if col.endswith('_norm')]
keep_fields.extend(norm_fields)

# 获取要删除的字段
all_fields = df.columns.tolist()
redundant_fields = [col for col in all_fields if col not in keep_fields]

if redundant_fields:
    print(f"删除以下冗余字段: {redundant_fields}")
    df.drop(columns=redundant_fields, inplace=True)

# ---------------------- 查看处理后数据 ----------------------
print("\n处理后的数据列：")
print(df.columns.tolist())

# 统计字段类型信息
num_fields = len(df.columns)
numerical_fields = len(df.select_dtypes(include=[np.number]).columns)
categorical_fields = len(df.select_dtypes(include=['object']).columns)

print(f"\n处理后共有 {num_fields} 个字段:")
print(f"  - {numerical_fields} 个数值字段")
print(f"  - {categorical_fields} 个分类字段")

# 统计边界类型分布
if "boundary_category" in df.columns:
    boundary_counts = df["boundary_category"].value_counts()
    print("\n边界类型分布:")
    for category, count in boundary_counts.items():
        print(f"  {category}: {count} 条记录 ({count/len(df)*100:.1f}%)")

# ---------------------- 保存处理后的数据 ----------------------
print(f"\n保存处理后的数据至：{output_csv}")
df.to_csv(output_csv, index=False)
print("数据处理完成！")

# ---------------------- 读取tif文件并打印波段信息 ----------------------
tif_path = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2processed\earthquake_features_raster.tif"
with rasterio.open(tif_path) as src:
    for i in range(1, src.count + 1):
        band = src.read(i)
        print(f"波段{i}: min={band.min()}, max={band.max()}, unique={np.unique(band)[:10]}")