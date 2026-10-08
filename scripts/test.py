import os
import subprocess
import re

# Set the path to your directory containing TIF files
path = '/Volumes/RECUPERAÇÃO/Terreno'

def combine_tifs(output_path, input_files):
    command = ['gdal_merge.py', '-o', output_path] + input_files
    try:
        subprocess.run(command, check=True)
        print(f"Combined TIF saved as {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error combining TIF files: {e}")

try:
    # Initialize an empty list to store file names
    tif_files_list = []

    # List all files in the directory
    files = os.listdir(path)

    # Iterate over each file in the directory
    for file in files:
        # Extract the numeric part from the filename
        if file.endswith('num.tif'):
            # Regular expression pattern to match two-digit numbers from 60 to 69
            pattern = r'W06([6-9][0-9])'

            # Using re.findall to extract all occurrences of the pattern
            matches = re.findall(pattern, file)

            if matches:
                print(f"Extracted numbers from {file}: {matches}")
                tif_files_list.append(file)
            else:
                print(f"No numbers in the range 60-69 found in {file}.")

            # Append the full path of the file to the list
            tif_files_list.append(os.path.join(path, file))

    # Print the list of matching files with comma-separated elements
    #if tif_files_list:
        print("ok")
    else:
        print("No 'num.tif' files found.")

    # Path to the combined output file
    output_combined = '/Users/marcelamello/Documents/GitHub/itmlogic/data/output_combined.tif'

    # Combine the TIF files if there are any
    if tif_files_list:
        combine_tifs(output_combined, tif_files_list)
    else:
        print("No 'num.tif' files found to combine.")

except OSError as e:
    # Handle any errors that occur
    print(f'Error: {e}')

except FileNotFoundError:
    print(f"Directory not found: {path}")
