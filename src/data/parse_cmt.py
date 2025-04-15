#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理CMT地震目录的ndk格式数据
"""

import os
import pandas as pd
import re
from datetime import datetime

# 文件路径设置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSING_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processing')

# 确保输出目录存在
os.makedirs(PROCESSING_DATA_DIR, exist_ok=True)


def parse_cmt_catalog():
    """解析CMT地震目录的ndk格式文件并转换为CSV格式"""
    print("开始解析CMT地震目录...")

    cmt_file = os.path.join(RAW_DATA_DIR, 'cmt_catalog', 'jan76_dec20.ndk')

    # 检查文件是否存在
    if not os.path.exists(cmt_file):
        print(f"错误: 文件 {cmt_file} 不存在!")
        return None

    # 用于存储解析后的数据
    earthquakes = []

    try:
        with open(cmt_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

            # ndk格式中，每5行代表一个地震事件
            line_count = len(lines)
            event_count = 0
            error_count = 0

            for i in range(0, line_count, 5):
                if i + 4 >= line_count:
                    break

                try:
                    # 解析第一行 - 地震基本信息
                    line1 = lines[i].strip()

                    # 检查是否有PDEW前缀（新格式）
                    if line1.startswith("PDEW"):
                        # 去掉PDEW前缀
                        line1 = line1[4:].strip()

                    # 提取日期时间
                    date_match = re.search(r'(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2}\.?\d*)', line1)

                    if date_match:
                        date_str = date_match.group(1)
                        time_str = date_match.group(2)

                        try:
                            dt = datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S.%f")
                        except ValueError:
                            try:
                                dt = datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S")
                            except ValueError:
                                dt = None
                                error_count += 1

                        # 提取日期后面的内容作为位置信息
                        location_str = line1[date_match.end():].strip()
                        loc_parts = location_str.split()

                        # 提取位置和深度
                        if len(loc_parts) >= 3:
                            try:
                                latitude = float(loc_parts[0])
                                longitude = float(loc_parts[1])
                                depth = float(loc_parts[2])
                            except ValueError:
                                latitude = longitude = depth = float('nan')
                                error_count += 1
                        else:
                            latitude = longitude = depth = float('nan')

                        # 提取震级和地点
                        mb = ms = float('nan')
                        region = ""

                        if len(loc_parts) >= 5:
                            try:
                                mb = float(loc_parts[3])
                                ms = float(loc_parts[4])
                            except ValueError:
                                pass

                        if len(loc_parts) > 5:
                            region = " ".join(loc_parts[5:])
                    else:
                        # 尝试解析老格式
                        parts = line1.split()
                        date_time_str = " ".join(parts[:2]) if len(parts) >= 2 else ""

                        try:
                            dt = datetime.strptime(date_time_str, "%m/%d/%y %H:%M:%S.%f")
                        except ValueError:
                            try:
                                dt = datetime.strptime(date_time_str, "%m/%d/%y %H:%M:%S")
                            except ValueError:
                                dt = None
                                error_count += 1

                        # 提取位置和深度
                        if len(parts) >= 5:
                            try:
                                latitude = float(parts[2])
                                longitude = float(parts[3])
                                depth = float(parts[4])
                            except (ValueError, IndexError):
                                latitude = longitude = depth = float('nan')
                                error_count += 1
                        else:
                            latitude = longitude = depth = float('nan')

                        # 提取震级和地点
                        mb = ms = float('nan')
                        region = ""

                        if len(parts) >= 7:
                            try:
                                mb = float(parts[5])
                                ms = float(parts[6])
                            except (ValueError, IndexError):
                                pass

                        if len(parts) > 7:
                            region = " ".join(parts[7:])

                    # 解析第二行 - CMT标识符
                    line2 = lines[i + 1].strip()
                    event_id = line2.split()[0] if len(line2.split()) > 0 else ""

                    # 解析第五行 - 断层面解
                    line5 = lines[i + 4].strip()
                    parts5 = line5.split()

                    # 尝试提取断层面参数
                    strike1 = dip1 = rake1 = strike2 = dip2 = rake2 = float('nan')

                    # 根据行的格式不同，尝试不同的解析方法
                    if len(parts5) >= 11:
                        # 尝试多种可能的断层面参数位置
                        try:
                            # 格式1：V10 8.940 75 283 1.260 2 19 -10.190 15 110 9.560 202 30 93
                            if parts5[0].startswith('V') and len(parts5) >= 13:
                                strike1 = float(parts5[2])
                                dip1 = float(parts5[3])
                                if len(parts5) >= 10:
                                    rake1 = float(parts5[9])
                                if len(parts5) >= 12:
                                    strike2 = float(parts5[10])
                                    dip2 = float(parts5[11])
                                if len(parts5) >= 13:
                                    rake2 = float(parts5[12])

                            # 格式2：MLI ... strike1 dip1 strike2 dip2 rake2
                            elif parts5[0] == 'MLI' and len(parts5) >= 14:
                                strike1 = float(parts5[9])
                                dip1 = float(parts5[10])
                                strike2 = float(parts5[11])
                                dip2 = float(parts5[12])
                                rake2 = float(parts5[13])

                        except (ValueError, IndexError):
                            pass

                    # 构建地震事件数据
                    if dt:  # 只有当成功解析日期时才添加记录
                        event = {
                            'event_id': event_id,
                            'datetime': dt,
                            'year': dt.year,
                            'month': dt.month,
                            'day': dt.day,
                            'hour': dt.hour,
                            'minute': dt.minute,
                            'second': dt.second,
                            'latitude': latitude,
                            'longitude': longitude,
                            'depth': depth,
                            'mb': mb,
                            'ms': ms,
                            'region': region,
                            'strike1': strike1,
                            'dip1': dip1,
                            'rake1': rake1,
                            'strike2': strike2,
                            'dip2': dip2,
                            'rake2': rake2
                        }

                        earthquakes.append(event)
                        event_count += 1

                    # 每1000个事件打印一次进度
                    if event_count % 1000 == 0:
                        print(f"已处理 {event_count} 个地震事件...")

                except Exception as e:
                    error_count += 1
                    if error_count < 10:  # 仅显示前10个错误
                        print(f"解析错误（行 {i + 1}）: {e}")
                    elif error_count == 10:
                        print("发生更多错误，不再单独显示...")

    except Exception as e:
        print(f"读取文件时出错: {e}")

    # 转换为DataFrame
    df = pd.DataFrame(earthquakes)

    # 保存为CSV
    output_file = os.path.join(PROCESSING_DATA_DIR, 'earthquakes.csv')
    df.to_csv(output_file, index=False)
    print(f"已保存{len(df)}个地震事件到 {output_file}")
    print(f"总共处理 {event_count} 个事件，遇到 {error_count} 个解析错误")

    return df


if __name__ == "__main__":
    parse_cmt_catalog()