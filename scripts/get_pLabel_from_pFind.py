# 使用方法 / Usage:
# python get_pLabel_from_pFind.py [pFind项目文件夹路径]
# python get_pLabel_from_pFind.py [pFind project folder path]

# 注意事项 / Notes:
# 1.把所有mgf/raw/pf2文件都放到同一个文件夹，pFind默认会把mgf或者pf2放到同一个目录下。
#   Place all mgf/raw/pf2 files in the same folder. pFind typically exports mgf or pf2 into the same directory by default.
# 2.raw名字除了扩展名前面的不要有"."。
#   The raw file name should not contain any "." before the extension.
# 3.要让pFind导出mgf，大多数情况下默认是导出的，在pFind “MS Data” 界面下的Data Extraction里。
#   Ensure pFind exports mgf files. This is usually enabled by default under the "Data Extraction" section in the pFind "MS Data" interface.
# 4.最终会产生.plabel文件，存到pFind项目下的plabel文件夹里，使用pLabel软件可以导入直接可视化谱图(https://pfind.net/software/pLabel/index.html)
#   This script will generate .plabel files stored in the plabel folder within the pFind project directory.
#   These can be imported directly into the pLabel software for spectral visualization (https://pfind.net/software/pLabel/index.html)

# jiale 2025.6.14
import argparse
import csv
import os
from collections import defaultdict
from os.path import join


def read_msms_paths(file_path):
    """
    从指定路径的文本配置文件中读取所有键名为 'msmspath...' 的值，
    并返回一个包含这些路径的列表。

    Reads values corresponding to keys starting with 'msmspath...' from a text configuration file,
    and returns a list of these paths.

    参数 / Parameters:
        file_path (str): 配置文件的路径 / Path to the configuration file.

    返回 / Returns:
        List[str]: 包含所有符合条件的路径字符串列表 /
                   A list containing all matching path strings.
    """
    paths = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # 跳过空行和注释行 / Skip empty lines and comment lines
            if not line or line.startswith("#"):
                continue

            # 检查是否为 key=value 形式 / Check whether the line is in key=value format
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().lower()

                # 判断键是否以 'msmspath' 开头 / Check whether the key starts with 'msmspath'
                if key.startswith("msmspath"):
                    paths.append(value.strip())
    return paths


def read_spectra(file_path):
    """
    从以制表符分隔的文本文件中提取三列数据：
    - File_Name
    - Sequence
    - Modification

    Extracts three columns ('File_Name', 'Sequence', 'Modification') from a tab-separated text file.

    参数 / Parameters:
        file_path (str): 文本文件的路径 / Path to the input text file.

    返回 / Returns:
        tuple: (file_names, sequences, modifications)
            - file_names: 包含 File_Name 的列表 / List containing File_Name entries.
            - sequences: 包含 Sequence 的列表 / List containing Sequence entries.
            - modifications: 包含 Modification 的列表 / List containing Modification entries.
    """
    file_names = []
    sequences = []
    modifications = []

    with open(file_path, "r", encoding="utf-8") as f:
        # 读取列头 / Read the header line
        headers = f.readline().strip().split("\t")

        # 确定所需列的索引 / Determine the indices of required columns
        try:
            file_idx = headers.index("File_Name")
            seq_idx = headers.index("Sequence")
            mod_idx = headers.index("Modification")
        except ValueError as e:
            raise KeyError(f"未找到必要列名 / Required column not found: {e}")

        # 逐行读取数据 / Read data line by line
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")

            # 提取数据 / Extract data
            file_name = parts[file_idx]
            sequence = parts[seq_idx]
            modification = parts[mod_idx] if mod_idx < len(parts) else ""

            # 添加到列表 / Append to lists
            file_names.append(file_name)
            sequences.append(sequence)
            modifications.append(modification)

    return file_names, sequences, modifications


def generate_plabel_files(pfind_path):
    # 确保输出目录存在 / Ensure the output directory exists
    output_dir = join(pfind_path, "plabel")
    result_file = join(pfind_path, "result", "pFind-Filtered.spectra")
    cfg_path = join(pfind_path, "param", "pFind.cfg")
    os.makedirs(output_dir, exist_ok=True)

    # 按 raw 名分组存储条目 / Group entries by raw name
    raw_groups = defaultdict(list)

    file_names, sequences, modifications = read_spectra(result_file)
    raw_name_set = set()
    for i in range(len(file_names)):
        raw_name = file_names[i].split(".")[0]

        mod_entries = []
        if len(modifications[i]) > 0:
            mod_info = modifications[i]
            for mod_str in mod_info.split(";"):
                mod_str = mod_str.strip()
                if not mod_str:
                    continue
                parts = mod_str.split(",", 1)
                if len(parts) != 2:
                    continue  # 忽略格式错误项 / Ignore malformed entries
                pos, mod_type = parts[0].strip(), parts[1].strip()
                mod_entries.append((pos, mod_type))
        raw_name_set.add(raw_name)
        raw_groups[raw_name].append(
            {
                "filename": file_names[i],
                "peptide": sequences[i],
                "mod_entries": mod_entries,
            }
        )

    paths = read_msms_paths(cfg_path)
    raw_name2mgf = {}
    for raw_name in raw_name_set:
        for path in paths:
            if raw_name + "_" in path:
                if "_" not in path.split(raw_name + "_")[1]:
                    raw_name2mgf[raw_name] = path.split(".")[0] + ".mgf"

    # 处理每个 raw 组，生成配置文件 / Process each raw group and generate the configuration file
    for raw_name, entries in raw_groups.items():
        # 收集所有修饰类型，保持顺序并去重 / Collect all modification types, preserving order and removing duplicates
        mod_types = []
        for entry in entries:
            for pos, mod_type in entry["mod_entries"]:
                if mod_type not in mod_types:
                    mod_types.append(mod_type)

        # 生成修饰映射：类型 -> 编号 / Generate modification mapping: type -> index
        mod_dict = {mod: idx + 1 for idx, mod in enumerate(mod_types)}

        # 构建配置文件内容 / Build the configuration file content
        config_content = []

        # [FilePath]
        config_content.append("[FilePath]")
        config_content.append(f"File_Path={raw_name2mgf[raw_name]}")
        config_content.append("")

        # [Modification]
        config_content.append("[Modification]")
        if not mod_types:
            config_content.append("1=None")
        else:
            for idx, mod in enumerate(mod_types):
                config_content.append(f"{idx+1}={mod}")
        config_content.append("")

        # [xlink]
        config_content.append("[xlink]")
        config_content.append("xlink=NULL")
        config_content.append("")

        # [Total]
        config_content.append("[Total]")
        config_content.append(f"total={len(entries)}")
        config_content.append("")

        # 每个 Spectrum 条目 / Each Spectrum entry
        for spectrum_id, entry in enumerate(entries, start=1):
            config_content.append(f"[Spectrum{spectrum_id}]")
            config_content.append(f'name={entry["filename"]}')

            # 构建 pep1 行：修饰位置和类型编号 / Build the pep1 line: modification positions and type indices
            mod_parts = []
            for pos, mod_type in entry["mod_entries"]:
                mod_num = mod_dict.get(mod_type, 0)
                if mod_num == 0:
                    continue  # 忽略未识别的修饰类型（理论上不应出现） / Ignore unrecognized modification types (should not happen in theory)
                mod_parts.append(f"{pos},{mod_num}")

            pep_line = f'pep1=0 {entry["peptide"]} 1'
            if mod_parts:
                pep_line += " " + " ".join(mod_parts)
            config_content.append(pep_line)
            config_content.append("")

        # 写入文件 / Write to file
        output_file = os.path.join(output_dir, f"{raw_name}.plabel")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(config_content))


if __name__ == "__main__":
    # 创建命令行参数解析器 / Create the command-line argument parser
    parser = argparse.ArgumentParser(description="处理得到 plabel 文件 / Generate plabel files")
    parser.add_argument("pFind_path", help="输入 pFind 项目文件路径 / Input pFind project folder path")
    args = parser.parse_args()

    generate_plabel_files(args.pFind_path)