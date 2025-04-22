import os
import math
import numpy as np
import pandas as pd

# ---------------------- 参数设置 ----------------------
input_csv = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2_processing\v2better\earthquake_features.csv"
output_csv = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2_processing\v2better\earthquake_features_processed.csv"
EPS = 1e-6  # 用于防止除零

# ---------------------- 读取数据 ----------------------
print("读取原始 earthquake_features.csv ...")
df = pd.read_csv(input_csv)
print(f"原始记录数：{len(df)}")

# ---------------------- 删除无关列 ----------------------
# 删除不需要的文本信息
drop_cols = ["origin_date", "origin_time", "location_info", "cmt_event_name", "cmt_type", "depth_type", "nodal_tokens", "data_used"]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# 删除 version 和 ms_norm 列（如果存在）
for col in ["version", "ms_norm"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# ---------------------- 处理震源矩误差 ----------------------
# 针对每个震源矩分量，生成加权特征 weighted_FIELD = FIELD / (FIELD_err + EPS)
tensor_fields = ["Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp"]
for field in tensor_fields:
    err_field = field + "_err"
    new_field = "weighted_" + field
    if field in df.columns and err_field in df.columns:
        df[new_field] = df[field] / (df[err_field].fillna(0) + EPS)
        # 删除原始值与误差字段
        df.drop(columns=[field, err_field], inplace=True)
    else:
        print(f"字段 {field} 或 {err_field} 不存在。")

# ---------------------- 板块信息简化 ----------------------
# 保留 "plate"，删除 "plate_code", "plate_id"
for col in ["plate_code", "plate_id"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# ---------------------- 文本字段编码 ----------------------
# 将 plate_type 和 crust_type 转换为数字代码，并删除原字段
def encode_series(series):
    unique_vals = sorted(series.dropna().unique())
    mapping = {val: i for i, val in enumerate(unique_vals)}
    return series.map(mapping), mapping

if "plate_type" in df.columns:
    df["plate_type_code"], plate_type_map = encode_series(df["plate_type"])
    print("plate_type映射：", plate_type_map)
    df.drop(columns=["plate_type"], inplace=True)

if "crust_type" in df.columns:
    df["crust_type_code"], crust_type_map = encode_series(df["crust_type"])
    print("crust_type映射：", crust_type_map)
    df.drop(columns=["crust_type"], inplace=True)

# fault_type1 和 fault_type2 转换为数字代码
fault_mapping = {"Reverse": 0, "Normal": 1, "Strike-slip": 2, "Mixed": 3}
for col in ["fault_type1", "fault_type2"]:
    if col in df.columns:
        df[col + "_code"] = df[col].apply(lambda x: fault_mapping.get(str(x).strip(), np.nan))
        df.drop(columns=[col], inplace=True)

# ---------------------- 断层面信息处理 ----------------------
# 如果存在 nodal_plane1，则解析出 nodal_strike, nodal_dip, nodal_rake
def parse_nodal_plane(plane):
    try:
        plane = str(plane).replace("(", "").replace(")", "").strip()
        parts = plane.split(",")
        if len(parts) >= 3:
            return float(parts[0].strip()), float(parts[1].strip()), float(parts[2].strip())
    except:
        return np.nan, np.nan, np.nan
    return np.nan, np.nan, np.nan

if "nodal_plane1" in df.columns:
    print("解析 nodal_plane1 ...")
    nodal_df = df["nodal_plane1"].apply(
        lambda x: pd.Series(parse_nodal_plane(x), index=["nodal_strike", "nodal_dip", "nodal_rake"]))
    df = pd.concat([df, nodal_df], axis=1)
    # 删除冗余字段：删除 nodal_plane1、nodal_plane2 以及现有的其他断层参数
    drop_nodal = ["nodal_plane1", "nodal_plane2", "strike1", "dip1", "rake1", "strike2", "dip2", "rake2"]
    df.drop(columns=[col for col in drop_nodal if col in df.columns], inplace=True)

# ---------------------- 衍生特征：机制与边界一致性得分 ----------------------
def mech_boundary_score(nodal_strike, boundary_azimuth):
    if pd.isnull(nodal_strike) or pd.isnull(boundary_azimuth):
        return np.nan
    diff = abs(nodal_strike - boundary_azimuth) % 180
    return math.cos(math.radians(diff))

if "nodal_strike" in df.columns and "boundary_azimuth" in df.columns:
    df["mechanism_boundary_score"] = df.apply(lambda r: mech_boundary_score(r["nodal_strike"], r["boundary_azimuth"]), axis=1)
else:
    print("缺少 nodal_strike 或 boundary_azimuth，无法生成 mechanism_boundary_score.")

# ---------------------- 归一化/正则化 ----------------------
# 对需要归一化的数值字段生成新列，然后删除原始列，但注意经纬度字段保持原始不归一化
def min_max(series):
    if series.max() == series.min():
        return series
    return (series - series.min()) / (series.max() - series.min())

# 列表中排除经纬度相关字段
norm_cols = [
    # 归一化深度、震级、震源矩、距离和角度等字段
    "hypocenter_depth", "centroid_depth", "depth_normalized",
    "mb", "ms",
    "moment_exponent", "scalar_moment",
    "distance_to_boundary", "distance_to_oc_boundary",
    "weighted_Mrr", "weighted_Mtt", "weighted_Mpp", "weighted_Mrt", "weighted_Mrp", "weighted_Mtp",
    "nodal_strike", "nodal_dip", "nodal_rake", "boundary_azimuth", "azimuth_to_boundary",
    "strike1_boundary_angle", "strike2_boundary_angle",
    "mechanism_boundary_score"
]

# 对于上述列表中的每个列，生成归一化结果，并删除原始列；但不对 "hypocenter_lat", "hypocenter_lon", "centroid_lat", "centroid_lon" 做归一化
for col in norm_cols:
    if col in df.columns:
        new_name = col + "_norm"
        df[new_name] = min_max(df[col])
        df.drop(columns=[col], inplace=True)

# 注意：保留原始经纬度数据，不进行归一化处理
# 这里确保 "hypocenter_lat", "hypocenter_lon", "centroid_lat", "centroid_lon" 仍存在
for col in ["hypocenter_lat", "hypocenter_lon", "centroid_lat", "centroid_lon"]:
    if col not in df.columns:
        print(f"警告：数据中缺少原始坐标列 {col}，需要保证后续空间聚合使用。")

# ---------------------- 查看处理后数据 ----------------------
print("处理后数据列：")
print(df.columns.tolist())

# ---------------------- 保存处理后的数据 ----------------------
print("保存处理后的数据至：", output_csv)
df.to_csv(output_csv, index=False)
print("数据处理完成。")
