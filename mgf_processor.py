import os
import argparse
from multiprocessing import Pool
from functools import partial

def process_single_file(input_path, output_path):
    """Process a single MGF file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            current_spectrum = []
            scans_value = None
            charge = 0

            for line in infile:
                line = line.strip()

                if line == "BEGIN IONS":
                    current_spectrum = [line]
                    scans_value = None
                elif line == "END IONS":
                    if current_spectrum:
                        current_spectrum.append(line)
                        processed = process_spectrum(current_spectrum, scans_value)
                        if charge > 0:
                            outfile.write('\n'.join(processed) + '\n\n')
                    current_spectrum = []
                elif current_spectrum:
                    current_spectrum.append(line)
                    if line.startswith("CHARGE="):
                        charge = int(line.split('=')[1].strip().replace("+", ""))
                    if line.startswith("SCANS="):
                        scans_value = line.split('=')[1].strip()

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False
    return True

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

def main():
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
    parser.add_argument('-p', '--processes',
                       type=int,
                       default=8,
                       help='Number of parallel processes (default: 8)')

    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        raise SystemExit(f"Error: Input directory '{args.input}' does not exist")
        
    os.makedirs(args.output, exist_ok=True)
    
    # Collect all mgf files
    files = [f for f in os.listdir(args.input) if f.lower().endswith('.mgf')]
    if not files:
        print("No MGF files found in input directory")
        return

    # Prepare arguments for multiprocessing
    process_func = partial(
        process_single_file,
        input_dir=args.input,
        output_dir=args.output
    )
    
    with Pool(processes=args.processes) as pool:
        results = []
        for filename in files:
            input_path = os.path.join(args.input, filename)
            output_path = os.path.join(args.output, filename)
            results.append(pool.apply_async(process_single_file, (input_path, output_path)))
        
        # Wait for all processes to complete
        success = 0
        total = len(results)
        for result in results:
            if result.get():
                success += 1
        
        print(f"Processing completed: {success}/{total} files processed successfully")
        print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()