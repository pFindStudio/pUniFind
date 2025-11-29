# Usage:
# python get_pLabel_from_pFind.py [path to pFind project folder]
# Notes:
# 1. Place all mgf/raw/pf2 files in the same folder. By default, pFind will place mgf or pf2 files in the same directory.
# 2. The raw file name (before the extension) should not contain any "." characters.
# 3. Make sure pFind exports mgf files. In most cases, this is enabled by default. You can check under "Data Extraction" in the "MS Data" interface of pFind.
# 4. This script will generate .plabel files, which are saved in the 'plabel' folder under the pFind project directory. These files can be imported into the pLabel software for direct spectral visualization (https://pfind.net/software/pLabel/index.html ).
# jiale 2025.6.14
import argparse
import csv
import os
from collections import defaultdict
from os.path import join


def read_msms_paths(file_path):
    """
    Read all key-value pairs with keys starting with 'msmspath...' from a configuration text file,
    and return a list of the corresponding path values.

    Parameters:
        file_path (str): Path to the configuration file.

    Returns:
        List[str]: A list containing all matching paths as strings.
    """
    paths = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Skip empty lines and comment lines
            if not line or line.startswith("#"):
                continue

            # Check for key=value format
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().lower()

                # Check if the key starts with 'msmspath'
                if key.startswith("msmspath"):
                    paths.append(value.strip())
    return paths


def read_spectra(file_path):
    """
    Extract three columns from a tab-separated text file:
    - File_Name
    - Sequence
    - Modification

    Parameters:
        file_path (str): Path to the input file.

    Returns:
        tuple: (file_names, sequences, modifications)
            - file_names: List of File_Name entries
            - sequences: List of Sequence entries
            - modifications: List of Modification entries
    """
    file_names = []
    sequences = []
    modifications = []

    with open(file_path, "r", encoding="utf-8") as f:
        # Read headers
        headers = f.readline().strip().split("\t")

        # Identify column indices
        try:
            file_idx = headers.index("File_Name")
            seq_idx = headers.index("Sequence")
            mod_idx = headers.index("Modification")
        except ValueError as e:
            raise KeyError(f"Required column missing: {e}")

        # Read data line by line
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")

            # Extract fields
            file_name = parts[file_idx]
            sequence = parts[seq_idx]
            modification = parts[mod_idx] if mod_idx < len(parts) else ""

            # Append to lists
            file_names.append(file_name)
            sequences.append(sequence)
            modifications.append(modification)

    return file_names, sequences, modifications


def generate_plabel_files(pfind_path):
    # Ensure output directory exists
    output_dir = join(pfind_path, "plabel")
    result_file = join(pfind_path, "result", "pFind-Filtered.spectra")
    cfg_path = join(pfind_path, "param", "pFind.cfg")
    os.makedirs(output_dir, exist_ok=True)

    # Group entries by raw file name
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
                    continue  # Skip malformed entries
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

    # Process each group and generate plabel config file
    for raw_name, entries in raw_groups.items():
        # Collect all modification types, preserving order and removing duplicates
        mod_types = []
        for entry in entries:
            for pos, mod_type in entry["mod_entries"]:
                if mod_type not in mod_types:
                    mod_types.append(mod_type)
        # Generate modification mapping: type -> number
        mod_dict = {mod: idx + 1 for idx, mod in enumerate(mod_types)}
        # Build config content
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
        # Each Spectrum entry
        for spectrum_id, entry in enumerate(entries, start=1):
            config_content.append(f"[Spectrum{spectrum_id}]")
            config_content.append(f'name={entry["filename"]}')
            # Construct pep1 line: modification positions and type numbers
            mod_parts = []
            for pos, mod_type in entry["mod_entries"]:
                mod_num = mod_dict.get(mod_type, 0)
                if mod_num == 0:
                    continue  # Skip unknown modification types (shouldn't happen)
                mod_parts.append(f"{pos},{mod_num}")
            pep_line = f'pep1=0 {entry["peptide"]} 1'
            if mod_parts:
                pep_line += " " + " ".join(mod_parts)
            config_content.append(pep_line)
            config_content.append("")
        # Write to file
        output_file = os.path.join(output_dir, f"{raw_name}.plabel")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(config_content))


if __name__ == "__main__":
    # Create command-line argument parser
    parser = argparse.ArgumentParser(description="Generate plabel files")
    parser.add_argument("pFind_path", help="Input path to pFind project directory")
    args = parser.parse_args()

    generate_plabel_files(args.pFind_path)
