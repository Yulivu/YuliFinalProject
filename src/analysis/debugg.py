# tectonic_data_analysis.py

import pandas as pd
import os

# 根目录（脚本所在路径的上两级）
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
data_dir = os.path.join(base_dir, 'dataProcess', 'processing')
output_dir = os.path.join(base_dir, 'dataProcess', 'explore')
os.makedirs(output_dir, exist_ok=True)

# 通用读取函数
def read_csv_with_fallback(filepath, column_names):
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        if list(df.columns) != column_names:
            # 假设没有 header，需要手动加上
            df = pd.read_csv(filepath, names=column_names, header=None, encoding='utf-8-sig')
    except Exception as e:
        print(f"❌ 读取失败: {filepath}\n{e}")
        df = pd.DataFrame(columns=column_names)
    return df

# 文件路径和对应列名
files_info = {
    'plate_boundaries.csv': [
        'segment_name', 'reference', 'segment_id', 'point_index',
        'longitude', 'latitude', 'plate1', 'plate2', 'boundary_type'
    ],
    'plates.csv': [
        'plate_id', 'segment_id', 'point_index', 'longitude', 'latitude'
    ],
    'plates_info.csv': [
        'plate_id', 'plate_name', 'pole_latitude', 'pole_longitude', 'rotation_rate'
    ],
    'plate_poles.csv': [
        'plate_id', 'pole_latitude', 'pole_longitude', 'rotation_rate'
    ]
}

# 读取所有文件
dataframes = {}
for fname, cols in files_info.items():
    path = os.path.join(data_dir, fname)
    print(f"📄 正在读取: {fname}")
    df = read_csv_with_fallback(path, cols)
    print(f"✅ {fname} 读取成功，行数: {len(df)}，列名: {df.columns.tolist()}")
    dataframes[fname] = df

# 示例分析：plate_boundaries 的边界段可视化保存（以后可以改成地图）
boundaries = dataframes['plate_boundaries.csv']
grouped = boundaries.groupby('segment_id')

# 输出每段边界的点数统计
segment_counts = grouped.size().reset_index(name='point_count')
segment_counts.to_csv(os.path.join(output_dir, 'boundary_segment_point_counts.csv'), index=False)
print("📊 已保存 boundary_segment_point_counts.csv")

# 保存所有 DataFrame 的预览信息（可用于后续理解）
for fname, df in dataframes.items():
    preview_path = os.path.join(output_dir, fname.replace('.csv', '_head.csv'))
    df.head(10).to_csv(preview_path, index=False)
    print(f"📝 预览已保存：{preview_path}")
