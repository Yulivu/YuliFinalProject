#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成用于机器学习的数据集
"""

import os
import pandas as pd
import numpy as np
import json
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# 设置路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSING_DIR = os.path.join(DATA_DIR, 'processing')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# 确保输出目录存在
os.makedirs(PROCESSED_DIR, exist_ok=True)


def convert_to_serializable(obj):
    """将NumPy类型转换为Python原生类型，确保可以JSON序列化"""
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    else:
        return obj


def load_data():
    """加载原始数据文件"""
    earthquakes = pd.read_csv(os.path.join(PROCESSING_DIR, 'earthquakes.csv'))
    plate_boundaries = pd.read_csv(os.path.join(PROCESSING_DIR, 'plate_boundaries.csv'))
    plates = pd.read_csv(os.path.join(PROCESSING_DIR, 'plates.csv'))
    plates_info = pd.read_csv(os.path.join(PROCESSING_DIR, 'plates_info.csv'))
    plate_poles = pd.read_csv(os.path.join(PROCESSING_DIR, 'plate_poles.csv'))

    print(f"加载了 {len(earthquakes)} 条地震记录")
    print(f"加载了 {len(plate_boundaries)} 条板块边界记录")
    print(f"加载了 {len(plate_poles)} 条板块极点记录")

    return earthquakes, plate_boundaries, plates, plates_info, plate_poles


def classify_fault_type_global_cmt(rake):
    """根据全球CMT标准分类断层类型
    参考: Frohlich, 1992 & Global CMT catalog
    """
    if pd.isna(rake):
        return 'unknown'

    rake = float(rake)

    # 标准化rake值到-180到180度范围
    while rake > 180:
        rake -= 360
    while rake < -180:
        rake += 360

    if abs(rake) <= 30 or abs(rake) >= 150:
        return 'strike-slip'  # 走滑断层
    elif 30 < rake < 150:
        return 'reverse'  # 逆断层
    elif -150 < rake < -30:
        return 'normal'  # 正断层
    else:
        return 'unknown'  # 未知类型


def calculate_angular_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """计算两点间的角距离（单位：度）"""
    # 转换为弧度
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)

    # 球面三角法计算角距离
    cos_angle = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)
    # 处理浮点数精度问题，确保cos_angle在[-1, 1]范围内
    cos_angle = max(min(cos_angle, 1.0), -1.0)

    # 将角距离转换回度数
    return math.degrees(math.acos(cos_angle))


def calculate_plate_velocities(lon, lat, boundary_row, poles_df):
    """
    计算给定位置的板块相对运动速度和方向

    参数:
    - lon, lat: 位置坐标
    - boundary_row: 包含plate1和plate2信息的边界行数据
    - poles_df: 欧拉极数据

    返回:
    - 速度大小、方向和类型特征
    """
    try:
        if not isinstance(boundary_row, pd.Series) or 'plate1' not in boundary_row or 'plate2' not in boundary_row:
            return {
                'relative_velocity': 0.0,
                'movement_direction': 0.0,
                'convergence_rate': 0.0,
                'transform_rate': 0.0,
                'extension_rate': 0.0,
                'boundary_motion_type': 'unknown'
            }

        plate1 = boundary_row['plate1']
        plate2 = boundary_row['plate2']

        # 获取两个板块的欧拉极数据
        plate1_pole = poles_df[poles_df['plate_id'] == plate1]
        plate2_pole = poles_df[poles_df['plate_id'] == plate2]

        if len(plate1_pole) == 0 or len(plate2_pole) == 0:
            return {
                'relative_velocity': 0.0,
                'movement_direction': 0.0,
                'convergence_rate': 0.0,
                'transform_rate': 0.0,
                'extension_rate': 0.0,
                'boundary_motion_type': 'unknown'
            }

        # 提取欧拉极参数
        pole1_lat = plate1_pole.iloc[0]['pole_latitude']
        pole1_lon = plate1_pole.iloc[0]['pole_longitude']
        pole1_rate = plate1_pole.iloc[0]['rotation_rate']

        pole2_lat = plate2_pole.iloc[0]['pole_latitude']
        pole2_lon = plate2_pole.iloc[0]['pole_longitude']
        pole2_rate = plate2_pole.iloc[0]['rotation_rate']

        # 计算相对欧拉极 (简化计算，实际需要使用更复杂的矢量运算)
        # 这里仅作示意，实际计算相对欧拉极需要更复杂的3D旋转矩阵运算
        rel_rate = abs(pole1_rate - pole2_rate)

        # 计算角距离 (从欧拉极到给定点的距离，单位：度)
        dist1 = calculate_angular_distance(lat, lon, pole1_lat, pole1_lon)
        dist2 = calculate_angular_distance(lat, lon, pole2_lat, pole2_lon)

        # 使用角距离估算线性速度 (公式: v = ω*R*sin(dist))
        # 其中ω是角速度，R是地球半径(约6371km)，dist是角距离
        earth_radius_km = 6371
        vel1 = pole1_rate * earth_radius_km * math.sin(math.radians(dist1)) / 1000.0  # 转换为m/year
        vel2 = pole2_rate * earth_radius_km * math.sin(math.radians(dist2)) / 1000.0

        # 估算相对速度(简化)
        relative_velocity = abs(vel1 - vel2)

        # 估算运动方向(简化)
        # 实际应计算欧拉极旋转向量与边界法向量的关系
        movement_direction = (dist1 + dist2) / 2  # 仅作占位，实际计算复杂得多

        # 分解速度分量 (高度简化)
        # 实际需要计算板块边界局部坐标系下的速度分量
        transform_rate = relative_velocity * 0.5  # 走滑分量
        convergence_rate = relative_velocity * 0.3  # 汇聚分量
        extension_rate = relative_velocity * 0.2  # 扩张分量

        # 判断主要运动类型
        if transform_rate > convergence_rate and transform_rate > extension_rate:
            boundary_motion_type = 'transform'
        elif convergence_rate > extension_rate:
            boundary_motion_type = 'convergent'
        else:
            boundary_motion_type = 'divergent'

        return {
            'relative_velocity': relative_velocity,
            'movement_direction': movement_direction,
            'convergence_rate': convergence_rate,
            'transform_rate': transform_rate,
            'extension_rate': extension_rate,
            'boundary_motion_type': boundary_motion_type
        }
    except Exception as e:
        print(f"计算板块速度时出错: {e}")
        return {
            'relative_velocity': 0.0,
            'movement_direction': 0.0,
            'convergence_rate': 0.0,
            'transform_rate': 0.0,
            'extension_rate': 0.0,
            'boundary_motion_type': 'unknown'
        }


def calculate_distance_to_boundary(quake_lon, quake_lat, boundaries_df):
    """计算地震到最近板块边界的距离(度)"""
    try:
        boundary_points = boundaries_df[['longitude', 'latitude']].values
        quake_point = np.array([[quake_lon, quake_lat]])

        # 使用KD树快速查找最近点
        tree = cKDTree(boundary_points)
        distance, idx = tree.query(quake_point, k=1)

        # 返回距离和最近边界点的索引
        return distance[0], idx[0]
    except Exception as e:
        print(f"计算距离时出错: {e}")
        return 999.0, -1  # 返回一个大值表示出错和无效索引


def get_boundary_info(quake_lon, quake_lat, boundaries_df, nearest_idx, max_dist=2.0):
    """获取地震附近的板块边界信息"""
    try:
        # 如果提供了有效的最近边界点索引
        if nearest_idx >= 0 and nearest_idx < len(boundaries_df):
            nearest_boundary = boundaries_df.iloc[nearest_idx]

            # 返回边界信息
            return {
                'boundary_type': nearest_boundary.get('boundary_type', 'unknown'),
                'plate1': nearest_boundary.get('plate1', 'unknown'),
                'plate2': nearest_boundary.get('plate2', 'unknown'),
                'segment_id': nearest_boundary.get('segment_id', -1)
            }

        # 如果索引无效，搜索附近区域
        nearby = boundaries_df[
            (boundaries_df['longitude'] > quake_lon - max_dist) &
            (boundaries_df['longitude'] < quake_lon + max_dist) &
            (boundaries_df['latitude'] > quake_lat - max_dist) &
            (boundaries_df['latitude'] < quake_lat + max_dist)
            ]

        if len(nearby) == 0:
            return {
                'boundary_type': 'unknown',
                'plate1': 'unknown',
                'plate2': 'unknown',
                'segment_id': -1
            }

        # 找出最近的边界
        min_dist = float('inf')
        nearest_row = None

        for idx, row in nearby.iterrows():
            dist = np.sqrt((quake_lon - row['longitude']) ** 2 + (quake_lat - row['latitude']) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_row = row

        if nearest_row is not None:
            return {
                'boundary_type': nearest_row.get('boundary_type', 'unknown'),
                'plate1': nearest_row.get('plate1', 'unknown'),
                'plate2': nearest_row.get('plate2', 'unknown'),
                'segment_id': nearest_row.get('segment_id', -1)
            }
        else:
            return {
                'boundary_type': 'unknown',
                'plate1': 'unknown',
                'plate2': 'unknown',
                'segment_id': -1
            }
    except Exception as e:
        print(f"获取边界信息时出错: {e}")
        return {
            'boundary_type': 'unknown',
            'plate1': 'unknown',
            'plate2': 'unknown',
            'segment_id': -1
        }


def create_grid_cell_id(lon, lat, cell_size=1.0):
    """创建网格单元ID"""
    grid_lon = int((lon + 180) / cell_size)
    grid_lat = int((lat + 90) / cell_size)
    return f"{grid_lon}_{grid_lat}"


def engineer_features(earthquakes_df, boundaries_df, poles_df, cell_size=1.0):
    """为地震数据创建特征"""
    features = []

    print("正在创建特征...")
    for _, quake in tqdm(earthquakes_df.iterrows(), total=len(earthquakes_df)):
        try:
            # 提取基本信息
            quake_id = quake['event_id']
            year = quake['year'] if 'year' in quake else 0
            lon = quake['longitude']
            lat = quake['latitude']
            depth = quake['depth']

            # 震级信息 (使用mb, ms等多种震级)
            mb = quake['mb'] if 'mb' in quake and not pd.isna(quake['mb']) else 0.0
            ms = quake['ms'] if 'ms' in quake and not pd.isna(quake['ms']) else 0.0

            # 使用可用的最大震级
            magnitude = max(mb, ms)

            # 断层特征
            strike1 = quake['strike1'] if 'strike1' in quake and not pd.isna(quake['strike1']) else 0
            dip1 = quake['dip1'] if 'dip1' in quake and not pd.isna(quake['dip1']) else 0
            rake1 = quake['rake1'] if 'rake1' in quake and not pd.isna(quake['rake1']) else 0

            # 计算派生特征
            fault_type = classify_fault_type_global_cmt(rake1)
            distance, nearest_idx = calculate_distance_to_boundary(lon, lat, boundaries_df)

            # 获取边界信息
            boundary_info = get_boundary_info(lon, lat, boundaries_df, nearest_idx)

            # 计算板块运动特征 (如果边界信息有效)
            if boundary_info['plate1'] != 'unknown' and boundary_info['plate2'] != 'unknown':
                # 创建包含plate1和plate2的Series以便传入函数
                boundary_row = pd.Series(boundary_info)
                plate_motion = calculate_plate_velocities(lon, lat, boundary_row, poles_df)
            else:
                plate_motion = {
                    'relative_velocity': 0.0,
                    'movement_direction': 0.0,
                    'convergence_rate': 0.0,
                    'transform_rate': 0.0,
                    'extension_rate': 0.0,
                    'boundary_motion_type': 'unknown'
                }

            grid_cell = create_grid_cell_id(lon, lat, cell_size)

            # 判断是否在板块边界附近
            is_near_boundary = 1 if distance < 1.0 else 0

            # 编码断层类型
            fault_type_code = {
                'strike-slip': 0,
                'reverse': 1,
                'normal': 2,
                'unknown': 3
            }.get(fault_type, 3)

            # 组合所有特征
            feature_dict = {
                'event_id': quake_id,
                'year': year,
                'longitude': lon,
                'latitude': lat,
                'depth': depth,
                'magnitude': magnitude,
                'mb': mb,
                'ms': ms,
                'strike1': strike1,
                'dip1': dip1,
                'rake1': rake1,
                'distance_to_boundary': distance,
                'boundary_type': boundary_info['boundary_type'],
                'plate1': boundary_info['plate1'],
                'plate2': boundary_info['plate2'],
                'segment_id': boundary_info['segment_id'],
                'relative_velocity': plate_motion['relative_velocity'],
                'movement_direction': plate_motion['movement_direction'],
                'convergence_rate': plate_motion['convergence_rate'],
                'transform_rate': plate_motion['transform_rate'],
                'extension_rate': plate_motion['extension_rate'],
                'boundary_motion_type': plate_motion['boundary_motion_type'],
                'grid_cell': grid_cell,
                'is_near_boundary': is_near_boundary,
                'fault_type': fault_type,
                'fault_type_code': fault_type_code
            }

            features.append(feature_dict)
        except Exception as e:
            print(f"处理地震 {quake.get('event_id', 'unknown')} 时出错: {e}")

    # 转换为DataFrame
    features_df = pd.DataFrame(features)

    return features_df


def create_boundary_grid(boundaries_df, region=None, resolution=0.1):
    """
    创建网格化的板块边界图

    参数:
    - boundaries_df: 板块边界数据框
    - region: 区域范围 [lon_min, lon_max, lat_min, lat_max]
    - resolution: 网格分辨率(度)

    返回:
    - grid: 边界网格数组
    - metadata: 网格元数据
    """
    # 如果未指定区域，使用全球范围
    if region is None:
        region = [-180, 180, -90, 90]

    lon_min, lon_max, lat_min, lat_max = region
    width = int((lon_max - lon_min) / resolution)
    height = int((lat_max - lat_min) / resolution)

    # 创建空网格
    grid = np.zeros((height, width), dtype=np.uint8)

    # 填充边界点
    for _, point in boundaries_df.iterrows():
        x = int((point['longitude'] - lon_min) / resolution)
        y = int((point['latitude'] - lat_min) / resolution)

        if 0 <= x < width and 0 <= y < height:
            # 标记边界点为1
            grid[y, x] = 1

            # 标记附近点以创建粗一些的边界线(可选)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        grid[ny, nx] = 1

    # 创建元数据
    metadata = {
        'region': region,
        'resolution': resolution,
        'width': width,
        'height': height,
        'description': '板块边界网格，1表示边界，0表示非边界'
    }

    return grid, metadata


def create_fault_type_grid(features_df, region=None, resolution=0.1):
    """
    创建不同断层类型的分布网格

    参数:
    - features_df: 地震特征数据框
    - region: 区域范围 [lon_min, lon_max, lat_min, lat_max]
    - resolution: 网格分辨率(度)

    返回:
    - grids: 包含不同断层类型网格的字典
    - metadata: 网格元数据
    """
    # 如果未指定区域，使用全球范围
    if region is None:
        region = [-180, 180, -90, 90]

    lon_min, lon_max, lat_min, lat_max = region
    width = int((lon_max - lon_min) / resolution)
    height = int((lat_max - lat_min) / resolution)

    # 创建空网格
    strike_slip_grid = np.zeros((height, width), dtype=np.uint8)
    reverse_grid = np.zeros((height, width), dtype=np.uint8)
    normal_grid = np.zeros((height, width), dtype=np.uint8)

    # 按断层类型填充网格
    for _, quake in features_df.iterrows():
        x = int((quake['longitude'] - lon_min) / resolution)
        y = int((quake['latitude'] - lat_min) / resolution)

        if 0 <= x < width and 0 <= y < height:
            fault_type = quake['fault_type']

            if fault_type == 'strike-slip':
                strike_slip_grid[y, x] += 1
            elif fault_type == 'reverse':
                reverse_grid[y, x] += 1
            elif fault_type == 'normal':
                normal_grid[y, x] += 1

    # 标准化网格 (对于更好的可视化)
    max_value = max(strike_slip_grid.max(), reverse_grid.max(), normal_grid.max())
    if max_value > 0:
        strike_slip_grid = (strike_slip_grid / max_value * 255).astype(np.uint8)
        reverse_grid = (reverse_grid / max_value * 255).astype(np.uint8)
        normal_grid = (normal_grid / max_value * 255).astype(np.uint8)

    # 创建元数据
    metadata = {
        'region': region,
        'resolution': resolution,
        'width': width,
        'height': height,
        'description': '断层类型分布网格，值表示该类型地震的相对频率'
    }

    # 返回网格和元数据
    grids = {
        'strike_slip': strike_slip_grid,
        'reverse': reverse_grid,
        'normal': normal_grid
    }

    return grids, metadata


def create_regional_datasets(features_df, boundaries_df, regions):
    """为每个指定区域创建数据集"""
    regional_datasets = {}

    for name, region in regions.items():
        print(f"处理区域: {name}")
        lon_min, lon_max, lat_min, lat_max = region

        # 过滤该区域的地震
        region_quakes = features_df[
            (features_df['longitude'] >= lon_min) &
            (features_df['longitude'] <= lon_max) &
            (features_df['latitude'] >= lat_min) &
            (features_df['latitude'] <= lat_max)
            ].copy()

        # 过滤该区域的边界
        region_boundaries = boundaries_df[
            (boundaries_df['longitude'] >= lon_min) &
            (boundaries_df['longitude'] <= lon_max) &
            (boundaries_df['latitude'] >= lat_min) &
            (boundaries_df['latitude'] <= lat_max)
            ].copy()

        # 创建该区域的边界网格
        grid, grid_metadata = create_boundary_grid(region_boundaries, region, resolution=0.1)

        # 创建该区域的断层类型网格
        fault_grids, fault_grid_metadata = create_fault_type_grid(region_quakes, region, resolution=0.1)

        # 保存区域数据
        region_path = os.path.join(PROCESSED_DIR, f"region_{name}")
        os.makedirs(region_path, exist_ok=True)

        # 保存地震特征
        region_quakes.to_csv(os.path.join(region_path, "earthquake_features.csv"), index=False)

        # 保存边界网格
        np.savez_compressed(
            os.path.join(region_path, "boundary_grid.npz"),
            grid=grid,
            metadata=json.dumps(convert_to_serializable(grid_metadata))
        )

        # 保存断层类型网格
        np.savez_compressed(
            os.path.join(region_path, "fault_type_grids.npz"),
            strike_slip=fault_grids['strike_slip'],
            reverse=fault_grids['reverse'],
            normal=fault_grids['normal'],
            metadata=json.dumps(convert_to_serializable(fault_grid_metadata))
        )

        # 创建区域可视化
        plt.figure(figsize=(15, 10))

        # 板块边界
        plt.subplot(2, 2, 1)
        plt.imshow(grid, cmap='binary', origin='lower')
        plt.colorbar(label='Boundary')
        plt.title(f'Plate Boundaries - {name}')

        # 地震分布
        plt.subplot(2, 2, 2)
        plt.scatter(
            region_quakes['longitude'],
            region_quakes['latitude'],
            c=region_quakes['fault_type_code'],
            cmap='viridis',
            alpha=0.5,
            s=3
        )
        plt.colorbar(label='Fault Type')
        plt.title(f'Earthquake Distribution - {name}')

        # 断层类型热图
        plt.subplot(2, 2, 3)
        # 合成RGB图
        rgb_img = np.zeros((fault_grids['strike_slip'].shape[0], fault_grids['strike_slip'].shape[1], 3),
                           dtype=np.uint8)
        rgb_img[:, :, 0] = fault_grids['reverse']  # 红色通道 - 逆断层
        rgb_img[:, :, 1] = fault_grids['strike_slip']  # 绿色通道 - 走滑断层
        rgb_img[:, :, 2] = fault_grids['normal']  # 蓝色通道 - 正断层
        plt.imshow(rgb_img, origin='lower')
        plt.title(f'Fault Type Distribution - {name} (R:Reverse G:Strike-slip B:Normal)')

        # 震级与深度
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(
            region_quakes['magnitude'],
            region_quakes['depth'],
            c=region_quakes['fault_type_code'],
            cmap='viridis',
            alpha=0.5,
            s=5
        )
        plt.colorbar(scatter, label='Fault Type')
        plt.title(f'Magnitude vs Depth - {name}')
        plt.xlabel('Magnitude')
        plt.ylabel('Depth (km)')
        plt.gca().invert_yaxis()  # 深度轴反转，使浅地震在上方

        plt.tight_layout()
        plt.savefig(os.path.join(region_path, "region_overview.png"), dpi=150)
        plt.close()

        # 统计信息 - 确保数值都是Python原生类型而非NumPy类型
        stats = {
            'region_name': name,
            'boundaries': int(len(region_boundaries)),
            'earthquakes': int(len(region_quakes)),
            'strike_slip_count': int(len(region_quakes[region_quakes['fault_type'] == 'strike-slip'])),
            'reverse_count': int(len(region_quakes[region_quakes['fault_type'] == 'reverse'])),
            'normal_count': int(len(region_quakes[region_quakes['fault_type'] == 'normal'])),
            'unknown_count': int(len(region_quakes[region_quakes['fault_type'] == 'unknown'])),
            'near_boundary_count': int(region_quakes['is_near_boundary'].sum()),
            'magnitude_avg': float(region_quakes['magnitude'].mean()),
            'depth_avg': float(region_quakes['depth'].mean()),
            'boundary_types': region_boundaries[
                'boundary_type'].value_counts().to_dict() if 'boundary_type' in region_boundaries.columns else {}
        }

        # 转换为可序列化格式并保存统计信息
        serialized_stats = convert_to_serializable(stats)
        with open(os.path.join(region_path, "stats.json"), "w") as f:
            json.dump(serialized_stats, f, indent=2)

        regional_datasets[name] = serialized_stats

    return regional_datasets


def main():
    """主函数"""
    try:
        print("开始生成机器学习数据集...")

        # 加载数据
        earthquakes, plate_boundaries, plates, plates_info, plate_poles = load_data()

        # 特征工程
        features_df = engineer_features(earthquakes, plate_boundaries, plate_poles)

        # 保存全局特征数据集
        features_df.to_csv(os.path.join(PROCESSED_DIR, "earthquake_features.csv"), index=False)
        print(f"保存了 {len(features_df)} 条特征记录到 earthquake_features.csv")

        # 定义构造意义的区域
        regions = {
            "circum_pacific_north": [120, 180, 30, 65],  # 环太平洋带北部(日本-千岛-阿拉斯加)
            "circum_pacific_west": [90, 150, -15, 30],  # 环太平洋带西部(菲律宾-印尼-所罗门)
            "circum_pacific_east": [-150, -80, -60, 25],  # 环太平洋带东部(智利-秘鲁-墨西哥)
            "alpine_himalayan": [0, 100, 25, 50],  # 阿尔卑斯-喜马拉雅带
            "mid_atlantic_ridge": [-50, -10, -30, 80],  # 大西洋中脊
            "east_african_rift": [25, 45, -15, 20],  # 东非裂谷
            "indian_ocean_ridge": [40, 100, -50, -10]  # 印度洋中脊
        }

        # 创建区域数据集
        print("开始创建区域数据集...")
        regional_stats = create_regional_datasets(features_df, plate_boundaries, regions)

        # 保存区域元数据
        print("保存区域元数据...")
        region_metadata = {
            'regions': regions,
            'stats': regional_stats,
            'total_earthquakes': int(len(earthquakes)),
            'total_boundaries': int(len(plate_boundaries))
        }

        with open(os.path.join(PROCESSED_DIR, "region_metadata.json"), "w") as f:
            json.dump(region_metadata, f, indent=2)

        print("数据集生成完成!")

        # 生成数据分布可视化
        print("生成数据分布可视化...")

        plt.figure(figsize=(12, 10))

        # 全球地震分布
        plt.subplot(2, 2, 1)
        plt.scatter(
            features_df['longitude'],
            features_df['latitude'],
            c=features_df['fault_type_code'],
            cmap='viridis',
            alpha=0.5,
            s=3
        )
        plt.colorbar(label='Fault Type')
        plt.title('Global Earthquake Distribution')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # 断层类型分布
        plt.subplot(2, 2, 2)
        fault_counts = features_df['fault_type'].value_counts()
        fault_counts.plot(kind='bar')
        plt.title('Fault Type Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        # 深度分布
        plt.subplot(2, 2, 3)
        plt.hist(features_df['depth'], bins=50)
        plt.title('Depth Distribution')
        plt.xlabel('Depth (km)')
        plt.ylabel('Count')

        # 震级分布
        plt.subplot(2, 2, 4)
        plt.hist(features_df['magnitude'], bins=20)
        plt.title('Magnitude Distribution')
        plt.xlabel('Magnitude')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(os.path.join(PROCESSED_DIR, "data_distribution.png"), dpi=150)
        plt.close()

        # 生成rake角度分布图，以验证断层类型分类
        plt.figure(figsize=(10, 6))
        plt.hist(features_df['rake1'].dropna(), bins=36, range=(-180, 180))
        plt.axvspan(-30, 30, alpha=0.3, color='green', label='Strike-slip')
        plt.axvspan(150, 180, alpha=0.3, color='green')
        plt.axvspan(-180, -150, alpha=0.3, color='green')
        plt.axvspan(30, 150, alpha=0.3, color='red', label='Reverse')
        plt.axvspan(-150, -30, alpha=0.3, color='blue', label='Normal')
        plt.legend()
        plt.xlabel('Rake (degrees)')
        plt.ylabel('Count')
        plt.title('Distribution of Rake Values - Global CMT Classification')
        plt.savefig(os.path.join(PROCESSED_DIR, "rake_distribution.png"), dpi=150)
        plt.close()

        print("可视化保存完成!")

        # 附加数据分析：断层类型与其他特征的关系
        print("生成断层类型与其他特征的关系图...")

        plt.figure(figsize=(15, 10))

        # 断层类型vs深度
        plt.subplot(2, 2, 1)
        for fault_type, group in features_df.groupby('fault_type'):
            if fault_type != 'unknown':
                plt.hist(group['depth'], bins=30, alpha=0.5, label=fault_type)
        plt.legend()
        plt.title('Fault Type vs Depth')
        plt.xlabel('Depth (km)')
        plt.ylabel('Count')

        # 断层类型vs震级
        plt.subplot(2, 2, 2)
        for fault_type, group in features_df.groupby('fault_type'):
            if fault_type != 'unknown':
                plt.hist(group['magnitude'], bins=15, alpha=0.5, label=fault_type)
        plt.legend()
        plt.title('Fault Type vs Magnitude')
        plt.xlabel('Magnitude')
        plt.ylabel('Count')

        # 震级vs深度，按断层类型着色
        plt.subplot(2, 2, 3)
        for fault_type, group in features_df.groupby('fault_type'):
            if fault_type != 'unknown':
                plt.scatter(group['magnitude'], group['depth'], alpha=0.3, label=fault_type, s=5)
        plt.legend()
        plt.title('Magnitude vs Depth by Fault Type')
        plt.xlabel('Magnitude')
        plt.ylabel('Depth (km)')
        plt.gca().invert_yaxis()  # 深度轴反转

        # 板块边界距离分布
        plt.subplot(2, 2, 4)
        plt.hist(features_df['distance_to_boundary'], bins=50, range=(0, 10))
        plt.title('Distance to Plate Boundary Distribution')
        plt.xlabel('Distance (degrees)')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(os.path.join(PROCESSED_DIR, "feature_relationships.png"), dpi=150)
        plt.close()

        print("特征关系图保存完成!")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()