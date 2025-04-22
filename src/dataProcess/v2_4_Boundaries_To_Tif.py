import os
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from affine import Affine

# 定义路径
input_shp = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2raw\plate\plate_boundaries.shp"
output_dir = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2processed"
output_raster = os.path.join(output_dir, "plate_boundaries_raster.tif")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

print("读取 plate_boundaries.shp ...")
# 使用 GeoPandas 读取 shapefile
gdf = gpd.read_file(input_shp)
print(f"读取到 {len(gdf)} 条边界记录")

# 固定全局经纬度范围
minx, miny, maxx, maxy = -180.0, -90.0, 180.0, 90.0
print("手动设置范围：", minx, miny, maxx, maxy)

# 设置栅格分辨率
resolution = 1.5
width = int((maxx - minx) / resolution)    # 360
height = int((maxy - miny) / resolution)   # 180

print("生成栅格的尺寸：宽度 =", width, "高度 =", height)

# 定义仿射变换：将栅格坐标映射到地理坐标
transform = Affine.translation(minx, miny) * Affine.scale(resolution, resolution)

# 准备 shapes：每个 geometry 赋值为 1（代表边界区域）
shapes = ((geom, 1) for geom in gdf.geometry)

print("进行栅格化 ...")
# 利用 rasterize 将矢量数据转换为栅格（掩膜），all_touched=True 表示与任意像元相交就标记为1
raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    all_touched=True,  # 若希望更加保守，则可以设为 False
    dtype=np.uint8
)

# 将栅格结果写入 GeoTIFF 文件
print("将结果写入", output_raster)
with rasterio.open(
    output_raster, 'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=raster.dtype,
    crs=gdf.crs,  # 保持与输入相同的坐标参考系统
    transform=transform,
) as dst:
    dst.write(raster, 1)

print("栅格化完成，结果保存在：", output_raster)
