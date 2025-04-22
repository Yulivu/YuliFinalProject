"""
数据处理脚本：整合GCMT地震数据与Hasterok板块构造数据
生成用于机器学习的特征集
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import math

# 定义路径
INPUT_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2raw\plate"
GCMT_PATH = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2_processing\cmt2.csv"
OUTPUT_DIR = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2_processing\v2better"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_gcmt_data():
    """加载GCMT地震目录数据"""
    print("加载GCMT地震目录数据...")
    df = pd.read_csv(GCMT_PATH)

    # 检查并处理NaN或无效数据
    # 确保经纬度列中没有NaN值
    invalid_coords = df['centroid_lon'].isna() | df['centroid_lat'].isna() | \
                    np.isinf(df['centroid_lon']) | np.isinf(df['centroid_lat'])

    if invalid_coords.any():
        print(f"警告: 发现 {invalid_coords.sum()} 条无效坐标记录，将被移除")
        df = df[~invalid_coords]

    # 将地震事件转换为GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['centroid_lon'], df['centroid_lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    print(f"加载了 {len(gdf)} 条地震记录")
    return gdf

def load_plate_data():
    """加载Hasterok板块数据"""
    print("加载Hasterok板块数据...")

    # 加载板块边界数据
    plate_boundaries = gpd.read_file(os.path.join(INPUT_DIR, "plate_boundaries.shp"))
    print(f"加载了 {len(plate_boundaries)} 条板块边界")

    # 加载板块多边形数据
    plates = gpd.read_file(os.path.join(INPUT_DIR, "plates.shp"))
    print(f"加载了 {len(plates)} 个板块多边形")

    # 加载大陆-海洋边界数据
    oc_boundaries = gpd.read_file(os.path.join(INPUT_DIR, "oc_boundaries.shp"))
    print(f"加载了 {len(oc_boundaries)} 条大陆-海洋边界")

    # 加载地质省区数据
    geoprovinces = gpd.read_file(os.path.join(INPUT_DIR, "global_gprv.shp"))
    print(f"加载了 {len(geoprovinces)} 个地质省区")

    # 打印列名以便调试
    print("Plates columns:", plates.columns.tolist())
    print("Geoprovinces columns:", geoprovinces.columns.tolist())

    return {
        'plate_boundaries': plate_boundaries,
        'plates': plates,
        'oc_boundaries': oc_boundaries,
        'geoprovinces': geoprovinces
    }

def compute_distance_azimuth(point, line_geometry):
    """计算点到线的最短距离和方位角"""
    try:
        min_dist = point.distance(line_geometry)

        # 找到线上最近的点
        if isinstance(line_geometry, LineString):
            nearest_point = line_geometry.interpolate(line_geometry.project(point))
            # 计算方位角 (0度为北，顺时针增加)
            dx = nearest_point.x - point.x
            dy = nearest_point.y - point.y
            azimuth = (90 - math.degrees(math.atan2(dy, dx))) % 360

            # 计算线段在最近点处的方向
            for i in range(len(line_geometry.coords) - 1):
                p1 = Point(line_geometry.coords[i])
                p2 = Point(line_geometry.coords[i+1])
                segment = LineString([p1, p2])
                # 如果最近点在这个线段上
                if nearest_point.distance(segment) < 1e-8:
                    segment_azimuth = (90 - math.degrees(math.atan2(
                        p2.y - p1.y, p2.x - p1.x))) % 360
                    return min_dist, azimuth, segment_azimuth

            # 如果找不到最近的线段，返回None作为线段方向
            return min_dist, azimuth, None
        else:
            # 对于非LineString几何体，只返回距离
            return min_dist, None, None
    except Exception as e:
        print(f"计算距离时出错: {e}")
        return None, None, None

def extract_nodal_plane_info(row):
    """从地震事件提取断层面信息"""
    if pd.isna(row['nodal_plane1']) or pd.isna(row['nodal_plane2']):
        return None, None, None, None, None, None

    try:
        # 尝试解析nodal_plane字符串
        np1 = str(row['nodal_plane1']).replace('(', '').replace(')', '').split(',')
        np2 = str(row['nodal_plane2']).replace('(', '').replace(')', '').split(',')

        if len(np1) >= 3 and len(np2) >= 3:
            strike1 = float(np1[0])
            dip1 = float(np1[1])
            rake1 = float(np1[2])

            strike2 = float(np2[0])
            dip2 = float(np2[1])
            rake2 = float(np2[2])

            return strike1, dip1, rake1, strike2, dip2, rake2
    except (ValueError, IndexError, TypeError) as e:
        pass

    return None, None, None, None, None, None

def compute_fault_type(strike, dip, rake):
    """根据断层面参数计算断层类型"""
    if pd.isna(strike) or pd.isna(dip) or pd.isna(rake):
        return "Unknown"

    # 根据Aki & Richards分类标准
    if -45 <= rake <= 45 or 135 <= rake <= 225:
        return "Strike-slip"
    elif 45 < rake < 135:
        return "Reverse"
    elif -135 < rake < -45:
        return "Normal"
    else:
        return "Mixed"

def create_feature_dataset(earthquakes_gdf, tectonic_data):
    """创建特征数据集"""
    print("生成特征数据集...")

    # 提取数据
    plate_boundaries = tectonic_data['plate_boundaries']
    plates = tectonic_data['plates']
    oc_boundaries = tectonic_data['oc_boundaries']
    geoprovinces = tectonic_data['geoprovinces']

    # 创建结果DataFrame
    features_df = earthquakes_gdf.copy()

    # 1. 提取断层面信息
    print("提取断层面信息...")
    nodal_info = features_df.apply(extract_nodal_plane_info, axis=1, result_type='expand')
    if not nodal_info.empty:
        features_df[['strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2']] = nodal_info

        # 计算断层类型
        features_df['fault_type1'] = features_df.apply(
            lambda row: compute_fault_type(row['strike1'], row['dip1'], row['rake1']), axis=1)
        features_df['fault_type2'] = features_df.apply(
            lambda row: compute_fault_type(row['strike2'], row['dip2'], row['rake2']), axis=1)

    # 2. 板块边界关系特征
    print("计算与板块边界的关系特征...")

    # 创建板块边界的空间索引用于快速距离查询
    boundary_points = []
    boundary_indices = []

    for idx, boundary in plate_boundaries.iterrows():
        if isinstance(boundary.geometry, LineString):
            for coord in boundary.geometry.coords:
                boundary_points.append((coord[0], coord[1]))
                boundary_indices.append(idx)

    if boundary_points:
        # 创建KD树
        boundary_tree = cKDTree(boundary_points)

        # 对每个地震，找到最近的边界点
        earthquake_points = []
        valid_indices = []

        for i, point in enumerate(features_df.geometry):
            try:
                x, y = point.x, point.y
                if np.isfinite(x) and np.isfinite(y):
                    earthquake_points.append((x, y))
                    valid_indices.append(i)
            except Exception as e:
                print(f"处理点 {i} 时出错: {e}")

        if earthquake_points:
            # 确保earthquake_points不为空
            print(f"使用 {len(earthquake_points)} 个有效点进行KD树查询")
            earthquake_points = np.array(earthquake_points)

            # 进行KD树查询
            distances, indices = boundary_tree.query(earthquake_points)

            # 创建与原始DataFrame相同长度的结果列
            distance_to_boundary = pd.Series([np.nan] * len(features_df))
            nearest_boundary_type = pd.Series([None] * len(features_df))

            # 只更新有效索引位置的数据
            for i, (valid_idx, dist, idx) in enumerate(zip(valid_indices, distances, indices)):
                distance_to_boundary.iloc[valid_idx] = dist

                # 获取最近边界的类型
                nearest_boundary_idx = boundary_indices[idx]
                nearest_boundary_type.iloc[valid_idx] = plate_boundaries.iloc[nearest_boundary_idx]['type']

            # 添加到特征DataFrame
            features_df['distance_to_boundary'] = distance_to_boundary
            features_df['nearest_boundary_type'] = nearest_boundary_type

            # 计算方位角和边界走向
            print("计算方位角和边界走向...")
            azimuth_to_boundary = pd.Series([np.nan] * len(features_df))
            boundary_azimuth = pd.Series([np.nan] * len(features_df))

            for i, valid_idx in enumerate(valid_indices):
                try:
                    boundary_idx = boundary_indices[indices[i]]
                    boundary_geom = plate_boundaries.iloc[boundary_idx].geometry
                    point = features_df.iloc[valid_idx].geometry

                    dist, azimuth, bound_azimuth = compute_distance_azimuth(point, boundary_geom)

                    if azimuth is not None:
                        azimuth_to_boundary.iloc[valid_idx] = azimuth
                    if bound_azimuth is not None:
                        boundary_azimuth.iloc[valid_idx] = bound_azimuth
                except Exception as e:
                    print(f"计算点 {valid_idx} 的方位角时出错: {e}")

            features_df['azimuth_to_boundary'] = azimuth_to_boundary
            features_df['boundary_azimuth'] = boundary_azimuth

            # 计算断层面走向与边界走向的夹角
            features_df['strike1_boundary_angle'] = features_df.apply(
                lambda row: abs(row['strike1'] - row['boundary_azimuth']) % 180
                if not pd.isna(row['strike1']) and not pd.isna(row['boundary_azimuth']) else None, axis=1)

            features_df['strike2_boundary_angle'] = features_df.apply(
                lambda row: abs(row['strike2'] - row['boundary_azimuth']) % 180
                if not pd.isna(row['strike2']) and not pd.isna(row['boundary_azimuth']) else None, axis=1)

    # 3. 地质环境特征
    print("提取地质环境特征...")
    try:
        # 空间连接获取每个地震所在的板块
        earthquakes_with_plates = gpd.sjoin(features_df, plates, how="left", predicate="within")

        # 确认连接后的列名
        print("空间连接后的板块列名:", [col for col in earthquakes_with_plates.columns if 'id' in col.lower() or 'plate' in col.lower() or 'crust' in col.lower()])

        # 使用正确的列名
        if 'id_right' in earthquakes_with_plates.columns:
            features_df['plate_id'] = earthquakes_with_plates['id_right']
        elif 'plate_id' in earthquakes_with_plates.columns:
            features_df['plate_id'] = earthquakes_with_plates['plate_id']

        # 其他列
        columns_map = {
            'plate': 'plate',
            'plate_code': 'plate_code',
            'plate_type': 'plate_type',
            'crust_type': 'crust_type',
            'poly_name': 'plate_name'
        }

        for src_col, dest_col in columns_map.items():
            if src_col in earthquakes_with_plates.columns:
                features_df[dest_col] = earthquakes_with_plates[src_col]
            elif src_col + '_right' in earthquakes_with_plates.columns:
                features_df[dest_col] = earthquakes_with_plates[src_col + '_right']

    except Exception as e:
        print(f"空间连接板块数据时出错: {e}")
        print("可用的列名:", earthquakes_with_plates.columns.tolist() if 'earthquakes_with_plates' in locals() else "无法获取列名")

    try:
        # 空间连接获取每个地震所在的地质省区
        earthquakes_with_provinces = gpd.sjoin(features_df, geoprovinces, how="left", predicate="within")

        # 确认连接后的列名
        print("地质省区连接后的列名:", [col for col in earthquakes_with_provinces.columns if 'id' in col.lower() or 'prov' in col.lower() or 'orogen' in col.lower()])

        # 使用正确的列名
        if 'id_right' in earthquakes_with_provinces.columns:
            features_df['province_id'] = earthquakes_with_provinces['id_right']
        elif 'id' in earthquakes_with_provinces.columns and 'id_right' not in earthquakes_with_provinces.columns:
            features_df['province_id'] = earthquakes_with_provinces['id']

        # 其他列
        columns_map = {
            'prov_name': 'province_name',
            'prov_type': 'province_type',
            'lastorogen': 'last_orogen'
        }

        for src_col, dest_col in columns_map.items():
            if src_col in earthquakes_with_provinces.columns:
                features_df[dest_col] = earthquakes_with_provinces[src_col]
            elif src_col + '_right' in earthquakes_with_provinces.columns:
                features_df[dest_col] = earthquakes_with_provinces[src_col + '_right']

    except Exception as e:
        print(f"空间连接地质省区数据时出错: {e}")
        print("可用的列名:", earthquakes_with_provinces.columns.tolist() if 'earthquakes_with_provinces' in locals() else "无法获取列名")

    # 计算到最近大陆-海洋边界的距离
    if not oc_boundaries.empty:
        print("计算到大陆-海洋边界的距离...")
        try:
            oc_boundary_points = []
            for idx, boundary in oc_boundaries.iterrows():
                if isinstance(boundary.geometry, LineString):
                    for coord in boundary.geometry.coords:
                        oc_boundary_points.append((coord[0], coord[1]))

            if oc_boundary_points:
                oc_tree = cKDTree(oc_boundary_points)

                # 使用之前收集的有效点
                if earthquake_points.size > 0:
                    oc_distances, _ = oc_tree.query(earthquake_points)

                    # 创建与原始DataFrame相同长度的结果列
                    distance_to_oc = pd.Series([np.nan] * len(features_df))

                    # 只更新有效索引位置的数据
                    for i, valid_idx in enumerate(valid_indices):
                        distance_to_oc.iloc[valid_idx] = oc_distances[i]

                    features_df['distance_to_oc_boundary'] = distance_to_oc
        except Exception as e:
            print(f"计算到大陆-海洋边界的距离时出错: {e}")

    # 4. 深度归一化（假设地壳厚度平均为35km）
    try:
        if 'centroid_depth' in features_df.columns:
            features_df['depth_normalized'] = features_df['centroid_depth'].apply(
                lambda x: x / 35.0 if pd.notnull(x) and np.isfinite(x) else np.nan)
    except Exception as e:
        print(f"计算深度归一化时出错: {e}")

    # 删除了步骤5：板块动力学特征计算

    print("特征生成完成")
    return features_df

def main():
    """主函数：处理数据并生成特征"""
    print("开始数据处理...")

    # 加载数据
    earthquakes_gdf = load_gcmt_data()
    tectonic_data = load_plate_data()

    # 创建特征
    features_df = create_feature_dataset(earthquakes_gdf, tectonic_data)

    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "earthquake_features.csv")
    features_df.drop(columns=['geometry'], errors='ignore').to_csv(output_path, index=False)

    # 保存地理数据
    geo_output_path = os.path.join(OUTPUT_DIR, "earthquake_features.gpkg")
    try:
        features_df.to_file(geo_output_path, driver="GPKG")
    except Exception as e:
        print(f"保存地理数据时出错: {e}")
        # 尝试只保存有效列
        valid_cols = [col for col in features_df.columns if col != 'geometry'] + ['geometry']
        features_df[valid_cols].to_file(geo_output_path, driver="GPKG")

    print(f"处理完成。结果保存至: {output_path}")

    # 生成可视化图表
    print("生成可视化图表...")

    try:
        # 1. 断层类型分布饼图
        if 'fault_type1' in features_df.columns:
            plt.figure(figsize=(10, 6))
            fault_counts = features_df['fault_type1'].dropna().value_counts()
            if not fault_counts.empty:
                plt.pie(fault_counts, labels=fault_counts.index, autopct='%1.1f%%')
                plt.title('Distribution of Earthquake Fault Types')
                plt.savefig(os.path.join(OUTPUT_DIR, 'fault_type_distribution.png'))
                plt.close()

        # 2. 地震深度直方图
        if 'centroid_depth' in features_df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(features_df['centroid_depth'].dropna(), bins=50)
            plt.xlabel('Depth (km)')
            plt.ylabel('Frequency')
            plt.title('Earthquake Depth Distribution')
            plt.savefig(os.path.join(OUTPUT_DIR, 'depth_distribution.png'))
            plt.close()

        # 3. 地震与板块边界距离关系
        if 'distance_to_boundary' in features_df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(features_df['distance_to_boundary'].dropna(), bins=50)
            plt.xlabel('Distance to Plate Boundary (degrees)')
            plt.ylabel('Frequency')
            plt.title('Earthquake Distance to Plate Boundaries')
            plt.savefig(os.path.join(OUTPUT_DIR, 'boundary_distance_distribution.png'))
            plt.close()

        # 4. 断层走向与板块边界走向夹角
        if 'strike1_boundary_angle' in features_df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(features_df['strike1_boundary_angle'].dropna(), bins=36)
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Frequency')
            plt.title('Angle Between Fault Strike and Plate Boundary')
            plt.savefig(os.path.join(OUTPUT_DIR, 'strike_boundary_angle.png'))
            plt.close()

        # 5. 板块边界类型分布
        if 'nearest_boundary_type' in features_df.columns:
            plt.figure(figsize=(12, 7))
            boundary_counts = features_df['nearest_boundary_type'].dropna().value_counts()
            if not boundary_counts.empty:
                boundary_counts.plot(kind='bar')
                plt.xlabel('Boundary Type')
                plt.ylabel('Count')
                plt.title('Distribution of Nearest Plate Boundary Types')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'boundary_type_distribution.png'))
                plt.close()

        # 6. 不同断层类型与板块边界类型的关系
        if 'fault_type1' in features_df.columns and 'nearest_boundary_type' in features_df.columns:
            # 创建一个交叉表
            cross_tab = pd.crosstab(
                features_df['fault_type1'].dropna(),
                features_df['nearest_boundary_type'].dropna()
            )

            # 绘制堆叠条形图
            plt.figure(figsize=(14, 8))
            cross_tab.plot(kind='bar', stacked=True)
            plt.xlabel('Fault Type')
            plt.ylabel('Count')
            plt.title('Relationship Between Fault Type and Plate Boundary Type')
            plt.xticks(rotation=0)
            plt.legend(title='Boundary Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'fault_boundary_relationship.png'))
            plt.close()

            # 保存交叉表
            cross_tab.to_csv(os.path.join(OUTPUT_DIR, 'fault_boundary_crosstab.csv'))
    except Exception as e:
        print(f"生成可视化图表时出错: {e}")

    print("数据处理与可视化完成")

if __name__ == "__main__":
    main()