"""
把GCMT的ndk数据转成csv，没有其他功能
    ndk_filepath = r"/data/v2raw/cmt/jan76_dec20.ndk"
    output_csv = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2_processing\cmt2.csv"
"""

import os
import pandas as pd
import numpy as np
import json


def parse_line1(line):
    # 第一行: Hypocenter 基本信息
    ref_catalog = line[0:4].strip()
    date = line[5:15].strip()
    time = line[16:26].strip()
    lat = line[27:33].strip()
    lon = line[34:41].strip()
    depth = line[42:47].strip()
    mag_str = line[48:55].strip()
    location = line[56:80].strip()

    # 对震级字段，可以选择拆分为两个震级，如 mb 和 MS
    mags = mag_str.split()
    mb_val = float(mags[0]) if len(mags) >= 1 and mags[0] not in ["", "0.0"] else None
    ms_val = float(mags[1]) if len(mags) >= 2 and mags[1] not in ["", "0.0"] else None

    return {
        "ref_catalog": ref_catalog,
        "origin_date": date,
        "origin_time": time,
        "hypocenter_lat": lat,
        "hypocenter_lon": lon,
        "hypocenter_depth": depth,
        "mb": mb_val,
        "ms": ms_val,
        "location_info": location
    }


def parse_line2(line):
    # 第二行: CMT 基本信息
    cmt_event_name = line[0:16].strip()
    data_used = line[17:61].strip()
    cmt_type = line[62:68].strip()
    moment_rate_info = line[69:80].strip()
    return {
        "cmt_event_name": cmt_event_name,
        "data_used": data_used,
        "cmt_type": cmt_type,
        "moment_rate_info": moment_rate_info
    }


def parse_line3(line):
    # 第三行: Centroid 参数
    # 检查是否以 "CENTROID:" 开头
    if not line.startswith("CENTROID:"):
        return {}
    centroid_part = line[9:58].strip()  # 理论上包含8个数值
    tokens = centroid_part.split()
    # 如果不足8个值，则补充None
    if len(tokens) < 8:
        tokens += [None] * (8 - len(tokens))
    try:
        centroid_time = float(tokens[0]) if tokens[0] is not None else None
        centroid_time_err = float(tokens[1]) if tokens[1] is not None else None
        centroid_lat = float(tokens[2]) if tokens[2] is not None else None
        centroid_lat_err = float(tokens[3]) if tokens[3] is not None else None
        centroid_lon = float(tokens[4]) if tokens[4] is not None else None
        centroid_lon_err = float(tokens[5]) if tokens[5] is not None else None
        centroid_depth = float(tokens[6]) if tokens[6] is not None else None
        centroid_depth_err = float(tokens[7]) if tokens[7] is not None else None
    except Exception as e:
        centroid_time = centroid_time_err = centroid_lat = centroid_lat_err = None
        centroid_lon = centroid_lon_err = centroid_depth = centroid_depth_err = None

    depth_type = line[59:63].strip()
    centroid_timestamp = line[64:80].strip()
    return {
        "centroid_time_offset": centroid_time,
        "centroid_time_err": centroid_time_err,
        "centroid_lat": centroid_lat,
        "centroid_lat_err": centroid_lat_err,
        "centroid_lon": centroid_lon,
        "centroid_lon_err": centroid_lon_err,
        "centroid_depth": centroid_depth,
        "centroid_depth_err": centroid_depth_err,
        "depth_type": depth_type,
        "centroid_timestamp": centroid_timestamp
    }


def parse_line4(line):
    # 第四行: 矩张量分量
    tokens = line.strip().split()
    if len(tokens) < 13:
        return {}
    try:
        # 根据规范，tokens[0] 为 exponent（前 2字符），将其转换为整数
        moment_exponent = int(tokens[0])
    except:
        moment_exponent = None
    try:
        Mrr = float(tokens[1])
        Mrr_err = float(tokens[2])
        Mtt = float(tokens[3])
        Mtt_err = float(tokens[4])
        Mpp = float(tokens[5])
        Mpp_err = float(tokens[6])
        Mrt = float(tokens[7])
        Mrt_err = float(tokens[8])
        Mrp = float(tokens[9])
        Mrp_err = float(tokens[10])
        Mtp = float(tokens[11])
        Mtp_err = float(tokens[12])
    except Exception as e:
        Mrr = Mrr_err = Mtt = Mtt_err = Mpp = Mpp_err = Mrt = Mrt_err = Mrp = Mrp_err = Mtp = Mtp_err = None

    return {
        "moment_exponent": moment_exponent,
        "Mrr": Mrr,
        "Mrr_err": Mrr_err,
        "Mtt": Mtt,
        "Mtt_err": Mtt_err,
        "Mpp": Mpp,
        "Mpp_err": Mpp_err,
        "Mrt": Mrt,
        "Mrt_err": Mrt_err,
        "Mrp": Mrp,
        "Mrp_err": Mrp_err,
        "Mtp": Mtp,
        "Mtp_err": Mtp_err
    }


def parse_line5(line, moment_exponent=None):
    """
    第五行: 解析 nodal plane、主轴、标量矩等信息
    这里将行按空白拆分为 tokens，要求总共应有 17 个 token：
      token[0]: 版本代码 (Version)
      token[1:4]: 第一组 - 第一 eigenvector = (eigenvalue, plunge, azimuth)
      token[4:7]: 第二组 - 第二 eigenvector
      token[7:10]: 第三组 - 第三 eigenvector
      token[10]: 标量矩 (scalar moment)
      token[11:14]: 第一 nodal plane参数 = (strike, dip, rake)
      token[14:17]: 第二 nodal plane参数 = (strike, dip, rake)
    如果提供了 moment_exponent，则对 eigenvalues 和 scalar moment 乘以 10**(moment_exponent)
    """
    raw_line = line.strip()
    tokens = raw_line.split()
    result = {}
    result["nodal_info"] = raw_line
    result["nodal_tokens"] = json.dumps(tokens, ensure_ascii=False)

    if len(tokens) != 17:
        # 如果 token 数量不对，直接返回 raw tokens
        return result

    try:
        version = tokens[0]
        eigen1 = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
        eigen2 = (float(tokens[4]), float(tokens[5]), float(tokens[6]))
        eigen3 = (float(tokens[7]), float(tokens[8]), float(tokens[9]))
        scalar_moment = float(tokens[10])
        nodal_plane1 = (float(tokens[11]), float(tokens[12]), float(tokens[13]))
        nodal_plane2 = (float(tokens[14]), float(tokens[15]), float(tokens[16]))
    except Exception as e:
        version = None
        eigen1 = eigen2 = eigen3 = None
        scalar_moment = None
        nodal_plane1 = nodal_plane2 = None

    # 如果提供了 moment_exponent，则对 eigenvalue和 scalar moment 乘以10^(exponent)
    if moment_exponent is not None:
        scale = 10 ** moment_exponent
        if eigen1 is not None:
            eigen1 = (eigen1[0] * scale, eigen1[1], eigen1[2])
        if eigen2 is not None:
            eigen2 = (eigen2[0] * scale, eigen2[1], eigen2[2])
        if eigen3 is not None:
            eigen3 = (eigen3[0] * scale, eigen3[1], eigen3[2])
        if scalar_moment is not None:
            scalar_moment = scalar_moment * scale

    result.update({
        "version": version,
        "eigenvector1": eigen1,
        "eigenvector2": eigen2,
        "eigenvector3": eigen3,
        "scalar_moment": scalar_moment,
        "nodal_plane1": nodal_plane1,
        "nodal_plane2": nodal_plane2
    })
    return result


def process_ndk(ndk_filepath):
    """
    读取 NDK 文件，每 5 行为一组事件，解析各行后整合为字典列表，最后返回 DataFrame。
    """
    events = []
    with open(ndk_filepath, "r") as f:
        lines = f.readlines()

    total_lines = len(lines)
    if total_lines % 5 != 0:
        print("警告：NDK 文件行数不是 5 的倍数，可能存在格式问题。")

    num_events = total_lines // 5
    print(f"共检测到 {num_events} 个事件。")
    for i in range(num_events):
        event = {}
        block = lines[i * 5:(i + 1) * 5]
        if len(block) < 5:
            continue
        # 依次解析每一行
        event.update(parse_line1(block[0]))
        event.update(parse_line2(block[1]))
        event.update(parse_line3(block[2]))
        line4_dict = parse_line4(block[3])
        event.update(line4_dict)
        # 对第五行，将来自 line4 的 moment_exponent 传入以便缩放 eigenvalue 和 scalar moment
        event.update(parse_line5(block[4], moment_exponent=line4_dict.get("moment_exponent")))
        events.append(event)
    return pd.DataFrame(events)


def main():
    ndk_filepath = r"/data/v2raw/cmt/jan76_dec20.ndk"
    output_csv = r"C:\Users\debuf\Desktop\YuliFinalProject\data\v2_processing\cmt2.csv"

    print("开始处理 NDK 文件...")
    df = process_ndk(ndk_filepath)
    print(f"处理完成，共提取 {len(df)} 条事件记录。")

    # 打印前 5 条记录用于检查
    print(df.head())

    # 保存为 CSV 文件
    df.to_csv(output_csv, index=False)
    print(f"已将处理结果保存到：{output_csv}")


if __name__ == "__main__":
    main()
