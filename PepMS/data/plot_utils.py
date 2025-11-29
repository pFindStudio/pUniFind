# import matplotlib.pyplot as plt
from os.path import join
from typing import Tuple

import torch

plot_base = r"/mnt/vepfs/fs_users/zhaojiale/Contrast_MS_Pep/scripts/plots"


def color_array(arr):
    categories = []
    for num in arr:
        if num == -1:
            categories.append("gray")
        elif num < 4:
            categories.append("blue")
        else:
            categories.append("red")
    return categories


def plot_spectrum(mz, intensity, labels=None, colors=None, plot_path=None):
    # mz = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    # intensity = [25, 30, 15, 50, 35, 45, 20, 60, 55, 70, 65, 75]
    # labels = ['y1-H3PO4', 'b2', 'y2', 'y3-H3PO4', 'y4', 'b4-H3PO4', 'b5', 'y7', 'y8', 'y10', 'y11-H3PO4', 'b12-H3PO4']
    # colors = ['red', 'blue', 'red', 'red', 'red', 'blue', 'blue', 'red', 'red', 'red', 'red', 'blue']
    # blue red gray
    assert len(mz) == len(intensity), (len(mz), len(intensity))
    if labels is None:
        labels = ["" for _ in range(len(mz))]
    if colors is None:
        colors = ["gray" for _ in range(len(mz))]
    fig, ax = plt.subplots(figsize=(12, 6))
    # 绘制条形图，去除边框并使柱子更细
    for mz, intensity, label, color in zip(mz, intensity, labels, colors):
        ax.bar(
            mz, intensity, width=2, color=color, edgecolor=None
        )  # width 参数调整柱子的宽度
        ax.text(mz, intensity, label, ha="center", va="bottom", fontsize=8, color=color)

    # 添加余弦相似度文本
    # cosine_similarity = 0.95
    # ax.text(0.95, 0.95, f'Cosine Similarity = {cosine_similarity}', transform=ax.transAxes, ha='right', va='top', fontsize=10)

    # 设置标签和标题
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative Intensity (%)")
    # ax.set_title('Annotated on LUAD data')

    # 显示图例
    handles = [
        plt.Line2D([0], [0], color="red", lw=4),
        plt.Line2D([0], [0], color="blue", lw=4),
        plt.Line2D([0], [0], color="gray", lw=4),
    ]
    labels = ["y-ion", "b-ion", "noise"]
    ax.legend(handles, labels, loc="upper right")

    # 保存图片到指定路径
    if plot_path is None:
        output_path = join(plot_base, "eg.png")
    else:
        output_path = plot_path

    plt.savefig(output_path)


def sqrt_and_norm(
    mz_array: torch.Tensor,
    int_array: torch.Tensor,
    precursor_mz: float,
    precursor_charge: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Square root the intensities and scale to unit norm.

    Parameters
    ----------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    precursor_mz : float
        The precursor m/z.
    precursor_charge : int
        The precursor charge.

    Returns
    -------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    """
    int_array = torch.sqrt(int_array)
    int_array /= torch.linalg.norm(int_array)
    return mz_array, int_array


def remove_precursor_peak(
    mz_array: torch.Tensor,
    int_array: torch.Tensor,
    precursor_mz: float,
    tol: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Square root the intensities and scale to unit norm.

    Parameters
    ----------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    precursor_mz : float
        The precursor m/z.
    precursor_charge : int
        The precursor charge.
    tol : float, optional
        The tolerance to remove the precursor peak in m/z.

    Returns
    -------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    """
    bounds = (precursor_mz - tol, precursor_mz + tol)
    outside = (mz_array >= bounds[0]) | (mz_array <= bounds[1])
    return mz_array[outside], int_array[outside]


def append_mgf(
    filename, title, charge, rtinseconds, pepmass, mz, intensity, pepseq=None
):
    with open(filename, "a") as f:  # 使用追加模式 'a'
        f.write("BEGIN IONS\r\n")
        f.write(f"TITLE={title}\r\n")
        f.write(f"CHARGE={charge}+\r\n")
        f.write(f"RTINSECONDS={rtinseconds:.7f}\r\n")
        f.write(f"PEPMASS={pepmass:.6f}\r\n")
        if pepseq is not None:
            f.write(f"PEPSEQ={pepseq}\r\n")

        for mz, intensity in zip(mz, intensity):
            f.write(f"{mz:.5f} {intensity:.1f}\r\n")

        f.write("END IONS\r\n")
        f.write("\r\n")


def write_mgf(
    filename, title, charge, rtinseconds, pepmass, mz_values, intensities, pepseq
):
    with open(filename, "w") as f:
        f.write("BEGIN IONS\n")
        f.write(f"TITLE={title}\n")
        f.write(f"CHARGE={charge}\n")
        f.write(f"RTINSECONDS={rtinseconds:.7f}\n")
        f.write(f"PEPMASS={pepmass:.6f}\n")
        f.write(f"PEPSEQ={pepseq}\n")

        for mz, intensity in zip(mz_values, intensities):
            f.write(f"{mz:.5f} {intensity:.1f}\n")

        f.write("END IONS\n")
