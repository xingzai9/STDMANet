import os
import cv2
import numpy as np
from PIL import Image
import argparse

# 用户可在此处自定义每个目标的颜色，顺序对应目标编号（1号、2号、3号...）
COLORS = [
    (255, 0, 0),    # 红色
    (0, 255, 0),    # 绿色
    (255, 255, 0),  # 黄色
    (0, 0, 255),    # 蓝色
    (255, 0, 255),  # 紫色
    (0, 255, 255),  # 青色
    # 可继续添加更多颜色
]

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))

# 在文件开头添加
SEQ_NAME = '000023'

def parse_args():
    parser = argparse.ArgumentParser(description='Draw bounding boxes for each target in masks and create a video.')
    base_path = rf'C:\baidunetdiskdownload\train\{SEQ_NAME}'
    parser.add_argument('--images', type=str, default=fr'{base_path}\img', help='Path to images folder')
    parser.add_argument('--masks', type=str, default=fr'{base_path}\mask', help='Path to masks folder')
    parser.add_argument('--output', type=str, default=fr'{base_path}\output_bbox.mp4', help='Output video file path')
    parser.add_argument('--fps', type=int, default=50, help='Frames per second for output video')
    parser.add_argument('--save_frames', type=str, default=fr'{base_path}\img_bbox', help='Directory to save images with bounding boxes (optional)')
    return parser.parse_args()


def main():
    args = parse_args()

    # 新增：如果指定了保存图片的文件夹，则创建
    save_frames_dir = args.save_frames
    if save_frames_dir is not None:
        os.makedirs(save_frames_dir, exist_ok=True)

    # 构建主文件名到完整路径的映射
    image_dict = {os.path.splitext(f)[0]: os.path.join(args.images, f)
                  for f in os.listdir(args.images) if os.path.isfile(os.path.join(args.images, f)) and is_image_file(f)}
    mask_dict = {os.path.splitext(f)[0]: os.path.join(args.masks, f)
                 for f in os.listdir(args.masks) if os.path.isfile(os.path.join(args.masks, f)) and is_image_file(f)}

    # 取交集，保证一一对应
    common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

    if not common_keys:
        print('No matching image-mask pairs found!')
        return

    frames = []
    for key in common_keys:
        img_path = image_dict[key]
        mask_path = mask_dict[key]
        img = cv2.imread(img_path)
        mask = np.array(Image.open(mask_path))
        for target_id in np.unique(mask):
            if target_id == 0:
                continue
            mask_bin = (mask == target_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = COLORS[(target_id-1) % len(COLORS)]
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w+1, y+h+1), color, 1)
        frames.append(img)
        # 新增：保存带框图片
        if save_frames_dir is not None:
            out_img_path = os.path.join(save_frames_dir, os.path.basename(img_path))
            cv2.imwrite(out_img_path, img)

    if len(frames) == 0:
        print('No frames to write!')
        return
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f'Video saved to {args.output}')


if __name__ == '__main__':
    main() 