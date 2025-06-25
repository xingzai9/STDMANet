import os
import argparse
import re
import shutil
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='重命名数据集图片文件为连续编号格式')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录，包含images和masks子目录')
    parser.add_argument('--backup', action='store_true', help='是否备份原始文件夹')
    parser.add_argument('--dry_run', action='store_true', help='只显示会进行的操作，不实际重命名')
    
    return parser.parse_args()


def natural_sort_key(s):
    """按照自然顺序对字符串排序的键函数"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def rename_files(folder_path, prefix="", is_mask=False, dry_run=False):
    """
    重命名给定文件夹中的所有图像文件
    
    参数:
        folder_path: 包含图像的文件夹路径
        prefix: 文件名前缀
        is_mask: 是否为掩码文件夹
        dry_run: 如果为True，只打印将要进行的操作，不实际重命名
    
    返回:
        重命名操作的映射字典 {原文件名: 新文件名}
    """
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return {}
    
    # 获取所有图像文件
    valid_extensions = ['.png', '.jpg', '.jpeg']
    files = [f for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    # 按照自然顺序排序文件
    files.sort(key=natural_sort_key)
    
    # 创建映射字典
    rename_map = {}
    
    # 重命名文件
    for i, filename in enumerate(files):
        file_ext = os.path.splitext(filename)[1]
        output_ext = '.png' if is_mask else file_ext  # 掩码统一使用PNG格式
        new_name = f"{prefix}{i+1:04d}{output_ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        rename_map[filename] = new_name
        
        if dry_run:
            print(f"将重命名: {os.path.join(folder_path, filename)} -> {os.path.join(folder_path, new_name)}")
        else:
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                print(f"重命名 {filename} 失败: {str(e)}")
    
    return rename_map


def process_dataset(data_dir, dry_run=False):
    """处理整个数据集的图像和掩码"""
    # 构建路径
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    
    # 确保目录存在
    if not os.path.exists(img_dir):
        print(f"图像目录不存在: {img_dir}")
        return
    
    if not os.path.exists(mask_dir):
        print(f"掩码目录不存在: {mask_dir}")
        os.makedirs(mask_dir, exist_ok=True)
    
    # 获取所有序列目录
    seq_folders = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    
    for seq in tqdm(seq_folders, desc="处理序列"):
        # 处理图像目录
        img_seq_path = os.path.join(img_dir, seq)
        print(f"\n处理图像目录: {img_seq_path}")
        
        # 重命名图像文件
        img_map = rename_files(img_seq_path, dry_run=dry_run)
        
        # 检查并创建对应的掩码目录
        mask_seq_path = os.path.join(mask_dir, seq)
        if not os.path.exists(mask_seq_path):
            print(f"创建掩码目录: {mask_seq_path}")
            if not dry_run:
                os.makedirs(mask_seq_path, exist_ok=True)
        
        # 如果掩码目录存在，处理掩码文件
        if os.path.exists(mask_seq_path):
            print(f"处理掩码目录: {mask_seq_path}")
            mask_files = os.listdir(mask_seq_path)
            
            # 检查掩码目录是否为空
            if not mask_files:
                print(f"掩码目录为空: {mask_seq_path}")
                continue
            
            # 如果掩码文件与图像文件有对应关系，则按照图像文件的重命名方式进行重命名
            if len(mask_files) == len(img_map):
                mask_files.sort(key=natural_sort_key)
                img_files = list(img_map.keys())
                img_files.sort(key=natural_sort_key)
                
                for i, (old_mask_name, old_img_name) in enumerate(zip(mask_files, img_files)):
                    new_mask_name = img_map[old_img_name].split('.')[0] + '.png'  # 使用对应图像的新名称，但扩展名为png
                    old_mask_path = os.path.join(mask_seq_path, old_mask_name)
                    new_mask_path = os.path.join(mask_seq_path, new_mask_name)
                    
                    if dry_run:
                        print(f"将重命名掩码: {old_mask_path} -> {new_mask_path}")
                    else:
                        try:
                            os.rename(old_mask_path, new_mask_path)
                        except Exception as e:
                            print(f"重命名掩码 {old_mask_name} 失败: {str(e)}")
            else:
                # 如果掩码与图像没有直接对应关系，则直接重命名掩码文件
                print(f"掩码文件数量({len(mask_files)})与图像文件数量({len(img_map)})不匹配，独立重命名掩码文件")
                rename_files(mask_seq_path, is_mask=True, dry_run=dry_run)


def backup_folder(folder_path):
    """备份文件夹"""
    if not os.path.exists(folder_path):
        print(f"文件夹不存在，无法备份: {folder_path}")
        return False
    
    backup_path = f"{folder_path}_backup"
    if os.path.exists(backup_path):
        print(f"备份文件夹已存在: {backup_path}")
        return False
    
    try:
        shutil.copytree(folder_path, backup_path)
        print(f"成功备份文件夹: {folder_path} -> {backup_path}")
        return True
    except Exception as e:
        print(f"备份文件夹失败: {str(e)}")
        return False


def main():
    args = parse_args()
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"数据目录不存在: {args.data_dir}")
        return
    
    # 备份原始文件夹
    if args.backup and not args.dry_run:
        print("正在备份数据文件夹...")
        backup_folder(args.data_dir)
    
    # 处理数据集
    print("正在处理数据集...")
    process_dataset(args.data_dir, dry_run=args.dry_run)
    
    if args.dry_run:
        print("\n这是一次演练运行，没有实际修改文件。")
        print("如果要执行实际重命名操作，请移除 --dry_run 参数。")
    else:
        print("\n文件重命名完成！")


if __name__ == "__main__":
    main()
