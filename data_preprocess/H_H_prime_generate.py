#!/home/junjianli/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2020-10-26 08:39
# Email: yps18@mails.tsinghua.edu.cn
# Filename: H_H_prime_generate.py
# Description:
#   Dataset preprocessing using glob for file listing and saving results in a specified directory.
# ******************************************************

import os
import cv2
import random
import numpy as np
import threadpool
import glob  # Added import for glob
from csco_vahadane import vahadane, read_image

# 定义数据集路径
dataset_path = '/home/junjianli/data/Kather_Multi_Class'

# 修改为新的保存目录
mid_fix = 'NCT-CRC-HE-100K-Stain-Separation'
PATCH_PATH = os.path.join(dataset_path, 'NCT-CRC-HE-100K')  # 输入图像目录
OUTPUT_BASE_PATH = os.path.join(dataset_path, mid_fix)  # 输出图像基目录

# 定义输出子目录
H_PATH = os.path.join(OUTPUT_BASE_PATH, 'H')
E_PATH = os.path.join(OUTPUT_BASE_PATH, 'E')
H_prime_PATH = os.path.join(OUTPUT_BASE_PATH, 'H_prime')
E_prime_PATH = os.path.join(OUTPUT_BASE_PATH, 'E_prime')

IMAGE_SIZE = 224  # 图像大小


def get_HorE(concentration):
    """
    根据浓度信息生成H或E图像。

    参数:
    - concentration (np.ndarray): 浓度矩阵

    返回:
    - np.ndarray: 生成的图像
    """
    return np.clip(255 * np.exp(-1 * concentration), 0, 255).reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)


def save_img(concen, concen_prime, name):
    """
    保存生成的H、E、H_prime和E_prime图像。

    参数:
    - concen (np.ndarray): 原始浓度矩阵
    - concen_prime (np.ndarray): 扰动后的浓度矩阵
    - name (str): 图像名称
    """
    # 生成并保存H图像
    H = get_HorE(concen[0, :])
    cv2.imwrite(os.path.join(H_PATH, f'{name}_H.png'), H)

    # 生成并保存E图像
    E = get_HorE(concen[1, :])
    cv2.imwrite(os.path.join(E_PATH, f'{name}_E.png'), E)

    # 生成并保存H_prime图像
    H_prime = get_HorE(concen_prime[0, :])
    cv2.imwrite(os.path.join(H_prime_PATH, f'{name}_H_prime.png'), H_prime)

    # 生成并保存E_prime图像
    E_prime = get_HorE(concen_prime[1, :])
    cv2.imwrite(os.path.join(E_prime_PATH, f'{name}_E_prime.png'), E_prime)


def get_img_list(path, k=None):
    """
    使用glob递归获取所有.png图像的相对路径，并随机打乱顺序。

    参数:
    - path (str): PATCH_PATH，即原始图像的根目录
    - k (int, optional): 要返回的图像数量，如果为None，则返回所有图像

    返回:
    - list: 图像相对路径列表
    """
    # 使用glob递归查找所有.png文件
    img_list = glob.glob(os.path.join(path, '**', '*.tif'), recursive=True)

    # 获取相对于PATCH_PATH的相对路径
    img_list = [os.path.relpath(img, path) for img in img_list]

    # 固定随机种子并打乱图像顺序
    random.seed(10)
    random.shuffle(img_list)

    if k is not None:
        return img_list[:k]
    else:
        return img_list


def main(img):
    """
    处理单个图像：读取图像，进行染色分离，扰动，保存结果。

    参数:
    - img (str): 图像的相对路径
    """
    try:
        # 获取图像名称（不包括扩展名）
        name = os.path.splitext(os.path.basename(img))[0]
        print(f'Processing: {name}')

        # 构建完整的图像路径并读取图像
        img_path = os.path.join(PATCH_PATH, img)

        if os.path.exists(os.path.join(H_PATH, f'{name}_H.png')):
            print(f"Skip {img} as it has been processed.")
            return
        img = read_image(img_path)

        # 进行染色分离
        stain, concen = vhd.stain_separate(img)

        # 添加扰动并重新进行染色分离
        perturb_stain = stain + np.random.randn(3, 2) * 0.05
        perturb_stain, perturb_concen = vhd.stain_separate(img, perturb_stain)

        # 保存生成的图像
        save_img(concen, perturb_concen, name)

    except Exception as e:
        print(f"Error processing {img}: {e}")


if __name__ == '__main__':
    # 定义处理线程数
    num_threads = 20

    # 初始化vahadane对象
    vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=1, ITER=50)
    vhd.show_config()

    # 确保所有目标输出目录存在
    for path in [H_PATH, E_PATH, H_prime_PATH, E_prime_PATH]:
        os.makedirs(path, exist_ok=True)

    # 获取所有图像列表
    img_list = get_img_list(PATCH_PATH, k=None)  # 若需限制数量，可设置k为特定值

    # not_processed_lst = ['BACK-LIMCTSWT', 'BACK-FTFIFESV']
    not_processed_lst = ['BACK']
    for img in img_list:
        for not_processed_img in not_processed_lst:
            if not_processed_img in img:
                img_list.remove(img)
    img_list = [img for img in img_list if 'BACK-LIMCTSWT' not in img]
    print('len of image_list:', len(img_list))

    # 创建线程池并分配任务

    # pool = threadpool.ThreadPool(num_threads)
    # requests = threadpool.makeRequests(main, img_list)
    # [pool.putRequest(req) for req in requests]
    # pool.wait()

    from tqdm import tqdm
    for img in tqdm(img_list):
        main(img)

    print("All images have been processed and saved successfully.")