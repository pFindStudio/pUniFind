# import matplotlib.pyplot as plt
from os.path import join

import torch


def plot_spectra(
    a_x,
    a_h,
    b_x,
    b_h,
    save_path="/mnt/vepfs/fs_users/zhaojiale/Contrast_MS_Pep/results/check_tmp",
    save_name=None,
):
    plt.figure(figsize=(10, 5))

    # 绘制上方的柱状图 (a_x, a_h)
    plt.bar(a_x, a_h, color="red", width=0.5, label="a-ion")

    # 绘制下方的柱状图 (b_x, b_h)
    plt.bar(b_x, -1 * b_h, color="blue", width=0.5, label="b-ion")

    # 添加水平线
    plt.axhline(0, color="black", linewidth=0.5)

    # 添加图例
    plt.legend()

    # 设置标题和标签
    plt.title("Mass Spectrum")
    plt.xlabel("m/z")
    plt.ylabel("Relative Intensity (%)")
    if save_path is not None:
        plt.savefig(join(save_path, save_name + ".png"))  # 保存图片
    plt.show()
