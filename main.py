#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于机器学习的地震断层运动方式识别及其在板块边界判识中的应用
主程序入口
"""

import os
import sys
import subprocess


def main():
    """主程序入口"""
    print("========================================================")
    print("基于机器学习的地震断层运动方式识别及其在板块边界判识中的应用")
    print("========================================================")

    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 处理CMT地震目录数据
    cmt_script_path = os.path.join(project_root, 'src', 'dataProcess', 'parse_cmt.py')
    if os.path.exists(cmt_script_path):
        print(f"\n1. 执行CMT地震目录处理脚本:")
        subprocess.run([sys.executable, cmt_script_path])
    else:
        print(f"错误: 找不到CMT处理脚本: {cmt_script_path}")
        print("请先创建CMT处理脚本")

    # 处理板块边界数据
    plates_script_path = os.path.join(project_root, 'src', 'dataProcess', 'parse_plates.py')
    if os.path.exists(plates_script_path):
        print(f"\n2. 执行板块边界数据处理脚本:")
        subprocess.run([sys.executable, plates_script_path])
    else:
        print(f"错误: 找不到板块数据处理脚本: {plates_script_path}")
        print("请先创建板块数据处理脚本")

    print("\n数据处理完成！数据已保存到 dataProcess/processing/ 目录")
    print("可视化结果保存在 dataProcess/explore/ 目录")

    # 这里将来可以添加更多功能
    # 如特征工程、模型训练等


if __name__ == "__main__":
    main()