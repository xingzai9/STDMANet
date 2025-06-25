#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图片转视频工具 - 将一系列图片拼接为MP4视频
"""

import os
import argparse
import cv2
import glob
from tqdm import tqdm
import re

def natural_sort_key(s):
    """
    用于自然排序文件名的辅助函数
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def images_to_video(image_dir, output_file, framerate=30, img_pattern="*.*", resize=None):
    """
    将图片目录中的图片转换为视频
    
    参数:
        image_dir (str): 图片所在的目录
        output_file (str): 输出视频的路径
        framerate (int): 帧率，默认30fps
        img_pattern (str): 图片匹配模式，默认所有图片
        resize (tuple): 可选，调整尺寸 (width, height)
    """
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']
    
    # 获取所有图片文件
    image_files = []
    for ext in supported_formats:
        pattern = os.path.join(image_dir, img_pattern.replace("*.*", f"*{ext}"))
        image_files.extend(glob.glob(pattern))
    
    # 自然排序文件名
    image_files.sort(key=natural_sort_key)
    
    if not image_files:
        print(f"错误：在 {image_dir} 中没有找到支持的图片文件")
        return False
    
    # 读取第一张图片获取尺寸
    img = cv2.imread(image_files[0])
    if img is None:
        print(f"错误：无法读取图片 {image_files[0]}")
        return False
    
    height, width, _ = img.shape
    
    if resize:
        width, height = resize
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
    video_writer = cv2.VideoWriter(output_file, fourcc, framerate, (width, height))
    
    print(f"开始处理 {len(image_files)} 张图片...")
    
    # 处理每一张图片
    for img_path in tqdm(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：跳过无法读取的图片 {img_path}")
            continue
        
        if resize:
            img = cv2.resize(img, (width, height))
        
        video_writer.write(img)
    
    # 释放资源
    video_writer.release()
    print(f"视频已成功保存到: {output_file}")
    return True

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将图片序列转换为MP4视频')
    parser.add_argument('-i', '--input', default=rf'C:\baidunetdiskdownload\train\000023\img', help='输入图片所在目录')
    parser.add_argument('-o', '--output', default=rf'C:\baidunetdiskdownload\train\000023\output.mp4', help='输出视频文件路径')
    parser.add_argument('-f', '--fps', type=int, default=50, help='视频帧率 (默认: 30)')
    parser.add_argument('-p', '--pattern', default='*.*', help='图片匹配模式 (默认: *.*)')
    parser.add_argument('-w', '--width', type=int, help='输出视频宽度 (可选)')
    parser.add_argument('--height', type=int, help='输出视频高度 (可选)')
    
    args = parser.parse_args()
    
    resize = None
    if args.width and args.height:
        resize = (args.width, args.height)
    
    # 执行转换
    images_to_video(args.input, args.output, args.fps, args.pattern, resize)

if __name__ == "__main__":
    main()
