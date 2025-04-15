#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

def load_eq_data(project_root: Path) -> pd.DataFrame:
    """
    加载 earthquakes.csv，并返回 DataFrame
    """
    eq_file = project_root / 'dataProcess' / 'processing' / 'earthquakes.csv'
    if not eq_file.is_file():
        raise FileNotFoundError(f"Cannot find {eq_file}")
    return pd.read_csv(eq_file)

def classify_fault_type(rake: float) -> str:
    """
    简易分类：
      - Reverse: 45 <= rake <= 135
      - Normal: -135 <= rake <= -45
      - Strike-slip: 其余
    不在范围或无效的 => Unknown
    """
    # 规范到 [-180,180]
    while rake > 180:
        rake -= 360
    while rake < -180:
        rake += 360

    if 45 <= rake <= 135:
        return "Reverse"
    elif -135 <= rake <= -45:
        return "Normal"
    else:
        return "Strike-slip"

def add_fault_type(eq_df: pd.DataFrame) -> pd.DataFrame:
    """
    为 DataFrame 添加 fault_type 列，根据 rake1
    """
    def safe_classify(x):
        try:
            return classify_fault_type(float(x))
        except:
            return "Unknown"
    eq_df['fault_type'] = eq_df['rake1'].apply(safe_classify)
    return eq_df

def main():
    # 假设本脚本位于 <项目根目录>/src/analysis/，向上 1 级到 /src，向上 1 级到 根目录
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    # 输出目录
    explore_dir = project_root / 'dataProcess' / 'explore'
    explore_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载地震数据
    eq_df = load_eq_data(project_root)

    # 2) 只保留 mb >= 4.0
    eq_df = eq_df[eq_df['mb'] >= 4.0].copy()
    eq_df.reset_index(drop=True, inplace=True)
    print(f"Total earthquakes (mb >= 4.0): {len(eq_df)}")

    # 3) 添加 fault_type 列
    eq_df = add_fault_type(eq_df)

    # 4) 统计 Unknown 数量
    unknown_count = (eq_df['fault_type'] == 'Unknown').sum()
    print(f"Unknown fault_type count: {unknown_count}")

    # 5) 也可统计各类型数量
    fault_counts = eq_df['fault_type'].value_counts()
    print("\n--- Fault type counts ---")
    print(fault_counts)

    # 6) 保存到 CSV
    stats_path = explore_dir / 'fault_type_stats.csv'
    fault_counts.to_csv(stats_path, header=['count'])
    print(f"\nStatistics saved to {stats_path}")

if __name__ == "__main__":
    main()
