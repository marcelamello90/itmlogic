"""
Point to Point prediction mode runner.

Referred to as qkpfl in the original Fortran codebase.

Written by Ed Oughton

June 2019

"""
import configparser
import os
import subprocess
import re
import csv
import math
import numpy as np
import pandas as pd

from functools import partial
from collections import OrderedDict

import fiona
from fiona.crs import from_epsg
from pyproj import Transformer
from shapely.geometry import LineString, mapping
from shapely.ops import transform

from itmlogic.misc.qerfi import qerfi
from itmlogic.preparatory_subroutines.qlrpfl import qlrpfl
from itmlogic.statistics.avar import avar
from terrain_module import terrain_p2p


# #set up file paths
CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
RESULTS = os.path.join(BASE_PATH, '..', 'results')

#Run terrain module
dem_folder = '/Volumes/RECUPERAÇÃO/Terreno'


output_combined = '/Users/marcelamello/Documents/GitHub/itmlogic/results/output_combined.tif'

        # Check if the output file already exists
if os.path.exists(output_combined):
    os.remove(output_combined)
    print(f"Deleted existing file: {output_combined}")
else:
    print("Aborting TIFF combination.")
                
import rasterio
import numpy as np

def stack_tif_bands(tif_files, output_file):
    # Open the first file to get profile
    with rasterio.open(tif_files[0]) as src:
        profile = src.profile
        # Update the profile to accommodate multiple bands
        profile.update(count=len(tif_files))
        
        # Create an array to hold all bands
        stack = np.zeros((len(tif_files), src.height, src.width), dtype=profile['dtype'])

    # Read each .tif file and stack as bands
    for i, tif in enumerate(tif_files):
        print(i)
        with rasterio.open(tif) as src:
            stack[i] = src.read(1)  # Read the first band

    # Write the stacked bands to the output file
    with rasterio.open(output_file, 'w', **profile) as dst:
        for i in range(stack.shape[0]):
            print(i)
            dst.write(stack[i], i + 1)

# List of .tif files to stack as bands
tif_files_list = []

    # List all files in the directory
files = os.listdir(dem_folder)

# Iterate over each file in the directory
for file in files:
                # Extract the numeric part from the filename
    if file.endswith('num.tif'):
        tif_files_list.append(os.path.join(dem_folder, file))


# Stack .tif files as bands
stack_tif_bands(tif_files_list, output_combined)



