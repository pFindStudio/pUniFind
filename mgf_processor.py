#!/usr/bin/env python3
"""
MGF File Processor - v1.2

Modifies MGF files with the following changes:
1. Adds SCANS value to TITLE line
2. Truncates PEPMASS values at first space
3. Removes specified metadata lines

Usage:
    python mgf_processor.py -i /input/dir -o /output/dir

Author: Your Name <marshmallowzjl@gmail.com>
"""

import os
import argparse

def process_mgf_files(input_dir, output_dir):
    """Process MGF files with specified modifications"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.mgf'):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
             
            current_spectrum = []
            scans_value = None
            
            for line in infile:
                line = line.strip()
                
                if line == "BEGIN IONS":
                    current_spectrum = [line]
                    scans_value = None
                elif line == "END IONS":
                    if current_spectrum:
                        current_spectrum.append(line)
                        processed = process_spectrum(current_spectrum, scans_value)
                        outfile.write('\n'.join(processed) + '\n\n')
                    current_spectrum = []
                elif current_spectrum:
                    current_spectrum.append(line)
                    if line.startswith("SCANS="):
                        scans_value = line.split('=')[1].strip()

def process_spectrum(spectrum_lines, scans_value):
    """Apply processing rules to individual spectrum"""
    processed = []
    remove_prefixes = {"NCE=", "HCD=", "FBR=", "TB=", "FB=", "MB="}
    
    for line in spectrum_lines:
        if any(line.startswith(prefix) for prefix in remove_prefixes):
            continue
            
        if line.startswith("TITLE=") and scans_value:
            title = line.split('=', 1)[1].strip()
            line = f"TITLE={title}_SCANS{scans_value}"
            
        elif line.startswith("PEPMASS="):
            parts = line.split('=', 1)
            if len(parts) > 1:
                pepmass = parts[1].split(maxsplit=1)[0]
                line = f"PEPMASS={pepmass}"
                
        processed.append(line)
    
    return processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-i', '--input', 
                       required=True,
                       help='Input directory containing .mgf files')
    parser.add_argument('-o', '--output',
                       required=True,
                       help='Output directory for processed files')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        raise SystemExit(f"Error: Input directory '{args.input}' does not exist")
        
    process_mgf_files(os.path.abspath(args.input), os.path.abspath(args.output))
    print(f"Processing completed. Files saved to: {args.output}")