#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def ensure_dir(path: Path):
    """Ensure that a directory exists. Create if it does not exist."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def load_data(project_root: Path):
    """
    Load CSV dataProcess from dataProcess/processing:
      - earthquakes.csv
      - plate_boundaries.csv
    Return them as eq_df, boundaries_df.
    """
    processing_dir = project_root / "dataProcess" / "processing"
    eq_file = processing_dir / "earthquakes.csv"
    bd_file = processing_dir / "plate_boundaries.csv"

    if not eq_file.is_file():
        raise FileNotFoundError(f"Cannot find {eq_file}")
    if not bd_file.is_file():
        raise FileNotFoundError(f"Cannot find {bd_file}")

    eq_df = pd.read_csv(eq_file)
    boundaries_df = pd.read_csv(bd_file)
    return eq_df, boundaries_df

def filter_earthquakes(eq_df: pd.DataFrame, mb_min=4.0) -> pd.DataFrame:
    """
    Filter out earthquakes with mb < mb_min.
    """
    filtered = eq_df[eq_df["mb"] >= mb_min].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered

def classify_fault_type(rake: float) -> str:
    """
    Simple classification by rake angle:
      - Reverse: 45 <= rake <= 135
      - Normal: -135 <= rake <= -45
      - Strike-slip: everything else
    """
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
    Derive 'fault_type' column from rake1.
    """
    eq_df["fault_type"] = eq_df["rake1"].apply(lambda x: classify_fault_type(float(x)))
    return eq_df

def plot_plate_boundaries_scatter(boundaries_df: pd.DataFrame, output_path: Path):
    """
    Plot all PB2002 boundary points as black scatter (no line, no color by boundary_type),
    at high resolution, without shifting the longitude.
    """
    plt.figure(figsize=(16, 8), dpi=600)
    ax = plt.gca()

    ax.scatter(
        boundaries_df["longitude"], boundaries_df["latitude"],
        s=2, color='black', alpha=0.6, label='Plate boundaries'
    )

    ax.set_title("Plate Boundaries (Scatter, Black)")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", "box")
    ax.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_global_map_scatter(eq_df: pd.DataFrame, boundaries_df: pd.DataFrame, output_path: Path):
    """
    Plot global earthquakes (colored by fault_type) + plate boundaries as black scatter,
    high resolution, with original [-180,180] range.
    """
    fault_colors = {
        "Strike-slip": "green",
        "Reverse": "red",
        "Normal": "blue"
    }

    plt.figure(figsize=(16, 8), dpi=600)
    ax = plt.gca()

    # 1) Plate boundaries in black
    ax.scatter(
        boundaries_df["longitude"], boundaries_df["latitude"],
        s=2, color='black', alpha=0.6, label='Plate boundaries'
    )

    # 2) Earthquakes in color
    for ftype, color in fault_colors.items():
        sub = eq_df[eq_df["fault_type"] == ftype]
        ax.scatter(sub["longitude"], sub["latitude"], s=5, c=color, alpha=0.7, label=ftype)

    ax.set_title("Global Earthquakes (Mb>=4) + Plate Boundaries", fontsize=10)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", "box")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_region_map_scatter(eq_df: pd.DataFrame, boundaries_df: pd.DataFrame,
                            region_name: str, lon_min: float, lon_max: float,
                            lat_min: float, lat_max: float, output_path: Path):
    """
    Plot a regional map: black scatter for plate boundaries + colored earthquakes,
    no shift in longitude, high resolution.
    """
    fault_colors = {
        "Strike-slip": "green",
        "Reverse": "red",
        "Normal": "blue"
    }

    # 只绘制区域内的边界点 & 地震
    mask_bd = (
        (boundaries_df["longitude"] >= lon_min) & (boundaries_df["longitude"] <= lon_max) &
        (boundaries_df["latitude"] >= lat_min) & (boundaries_df["latitude"] <= lat_max)
    )
    region_bd = boundaries_df[mask_bd]

    mask_eq = (
        (eq_df["longitude"] >= lon_min) & (eq_df["longitude"] <= lon_max) &
        (eq_df["latitude"] >= lat_min) & (eq_df["latitude"] <= lat_max)
    )
    region_quakes = eq_df[mask_eq]

    plt.figure(figsize=(16, 8), dpi=600)
    ax = plt.gca()

    # Plot boundaries in black
    ax.scatter(
        region_bd["longitude"], region_bd["latitude"],
        s=3, color='black', alpha=0.7, label='Plate boundaries'
    )

    # Plot earthquakes in color
    for ftype, color in fault_colors.items():
        sub = region_quakes[region_quakes["fault_type"] == ftype]
        ax.scatter(sub["longitude"], sub["latitude"], s=8, c=color, alpha=0.6, label=ftype)

    ax.set_title(f"{region_name} Region (Mb>=4) - Plate Boundaries & Fault Type", fontsize=10)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", "box")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_rake_vs_depth(eq_df: pd.DataFrame, output_path: Path):
    """
    Plot Rake1 vs Depth, colored by fault_type, high resolution,
    doesn't involve longitude.
    """
    fault_colors = {
        "Strike-slip": "green",
        "Reverse": "red",
        "Normal": "blue"
    }

    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.gca()

    for ftype, color in fault_colors.items():
        sub = eq_df[eq_df["fault_type"] == ftype]
        ax.scatter(sub["rake1"], sub["depth"], s=10, c=color, alpha=0.5, label=ftype)

    ax.set_title("Depth vs. Rake1 (Mb>=4)", fontsize=10)
    ax.set_xlabel("Rake1 (degrees)")
    ax.set_ylabel("Depth (km)")
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # 定位到项目根目录 (若脚本位于 <root>/src/analysis/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    # 输出目录
    explore_dir = project_root / "dataProcess" / "explore"
    ensure_dir(explore_dir)

    # 1) 加载数据
    eq_df, boundaries_df = load_data(project_root)

    # 2) 过滤地震
    eq_df = filter_earthquakes(eq_df, 4.0)

    # 3) rake1->fault_type
    eq_df = add_fault_type(eq_df)

    # 4) 仅板块边界 (黑色散点) - 保留原坐标
    plate_scatter_path = explore_dir / "plate_boundaries_scatter_black.png"
    plot_plate_boundaries_scatter(boundaries_df, plate_scatter_path)

    # 5) 全球地震 + 边界 (黑色散点) - 原坐标
    global_map_path = explore_dir / "global_fault_scatter_black.png"
    plot_global_map_scatter(eq_df, boundaries_df, global_map_path)

    # 6) 太平洋区域 (示例：-160..-100, lat -60..60)
    pacific_path = explore_dir / "pacific_region_scatter_black.png"
    plot_region_map_scatter(eq_df, boundaries_df,
                            "Pacific", -160, -100, -60, 60,
                            pacific_path)

    # 7) 印度洋区域 (30..120, lat -60..30)
    indian_path = explore_dir / "indian_region_scatter_black.png"
    plot_region_map_scatter(eq_df, boundaries_df,
                            "Indian Ocean", 30, 120, -60, 30,
                            indian_path)

    # 8) Rake vs Depth
    rake_depth_path = explore_dir / "rake_vs_depth.png"
    plot_rake_vs_depth(eq_df, rake_depth_path)

    print("Done! All high-resolution figures saved in:", explore_dir)


if __name__ == "__main__":
    main()
