# Usage:
# python get_pLabel_from_pUniFind_rescoring_English.py [pFind project folder path] [pUniFind result.spectra file path]
# Notes:
# 1. Place all mgf/raw/pf2 files in the same folder. pFind typically exports mgf or pf2 into the same directory by default.
# 2. The raw file name should not contain any "." before the extension.
# 3. Ensure pFind exports mgf files. This is usually enabled by default under the "Data Extraction" section in the pFind "MS Data" interface.
# 4. This script will generate .plabel files stored in the plabel folder within the pFind project directory. These can be imported directly into the pLabel software for spectral visualization (https://pfind.net/software/pLabel/index.html).

# jiale 2025.7.8
import argparse
import csv
import os
from collections import defaultdict
from os.path import join


def read_msms_paths(file_path):
    """
    Reads values corresponding to keys starting with 'msmspath...' from a text configuration file,
    and returns a list of these paths.

    Parameters:
        file_path (str): Path to the configuration file.

    Returns:
        List[str]: A list containing all matching path strings.
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
    Extracts three columns ('File_Name', 'Sequence', 'Modification') from a tab-separated text file.

    Parameters:
        file_path (str): Path to the input text file.

    Returns:
        tuple: (file_names, sequences, modifications)
            - file_names: List containing File_Name entries.
            - sequences: List containing Sequence entries.
            - modifications: List containing Modification entries.
    """
    file_names = []
    sequences = []
    modifications = []

    with open(file_path, "r", encoding="utf-8") as f:
        # Read headers
        headers = f.readline().strip().split("\t")

        # Determine column indices
        try:
            file_idx = headers.index("File_Name")
            seq_idx = headers.index("Sequence")
            mod_idx = headers.index("Modification")
        except ValueError as e:
            raise KeyError(f"Required column not found: {e}")

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


def generate_plabel_files(pfind_path, pUniFind_spectra_path):
    # Make sure output directory exists
    output_dir = join(pfind_path, "plabel")
    result_file = pUniFind_spectra_path
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
                    continue  # Ignore malformed entries
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
    # Process each raw group and generate config files
    for raw_name, entries in raw_groups.items():
        # Collect all modification types, preserving order and removing duplicates
        mod_types = []
        for entry in entries:
            for pos, mod_type in entry["mod_entries"]:
                if mod_type not in mod_types:
                    mod_types.append(mod_type)
        # Generate modification mapping: type -> index
        mod_dict = {mod: idx + 1 for idx, mod in enumerate(mod_types)}
        # Build configuration content
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
            # Build pep1 line: modification positions and type indices
            mod_parts = []
            for pos, mod_type in entry["mod_entries"]:
                mod_num = mod_dict.get(mod_type, 0)
                if mod_num == 0:
                    continue  # Skip unrecognized modification types (shouldn't happen)
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
    parser = argparse.ArgumentParser(description="Process to generate plabel files")
    parser.add_argument("pFind_path", help="Input path to the pFind project folder")
    parser.add_argument(
        "pUniFind_spectra_path", help="Input path to the pUniFind .spectra result file"
    )
    args = parser.parse_args()

    # Optional: Check if files or paths exist
    if not os.path.exists(args.pFind_path):
        parser.error(
            "The specified pFind project path does not exist: " + args.pFind_path
        )
    if not os.path.exists(args.pUniFind_spectra_path):
        parser.error(
            "The specified pUniFind .spectra file does not exist: "
            + args.pUniFind_spectra_path
        )

    generate_plabel_files(args.pFind_path, args.pUniFind_spectra_path)
