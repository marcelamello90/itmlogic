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
raw_data_folder = '/Users/marcelamello/Dropbox/Evangelical Churches in Brazil/ITMlogic/raw_data'
results_folder = '/Users/marcelamello/Dropbox/Evangelical Churches in Brazil/ITMlogic/results'
dem_folder = '/Volumes/RECUPERAÇÃO/Terreno'

YEAR = [#"1992", 
    #"1993", "1994", "1995", 
    #"1996" , "1997", "1998","1999", "2000"
    #"2001", "2002", 
    "2003", 
    #"2004", "2005", "2006", "2007"
        #"41", "42", 
        #"43", 
        #"50", 
        #"51","52", 
        #"53",
        #"11", "12", "13", "14", "15", "16", "17"
        ]
for year in YEAR:
        # Open Data
    raw_file = f'top_10_distances_RecordTV_{year}.csv'
    raw_data_file = os.path.join(raw_data_folder, raw_file)
    # Initialize an empty list to store the data
    data_test = []


    # Open the CSV file
    with open(raw_data_file, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            data_test.append(row)

    # Convert data_test to a DataFrame if it's not already
    if not isinstance(data_test, pd.DataFrame):
        data_test = pd.DataFrame(data_test, columns=header)

    # Initialize a list to store all results
    all_results = []

    # Check if the CSV file exists and delete it if it does

    results_file = f'p2p_results_top10_{year}.csv'
    results_csv_path = os.path.join(results_folder, results_file)

    if os.path.exists(results_csv_path):

        os.remove(results_csv_path)
        print(f"Deleted existing file: {results_csv_path}")
    else:
        print(f"No existing file to delete at: {results_csv_path}")


    # Write results
    def csv_writer(data, directory, filename):
        """
        Write data to a CSV file. If the file does not exist, it will be created.
        If the file exists, data will be appended.
        
        Parameters
        ----------
        data : list of dicts
            Data to be written.
        directory : string
            Folder to write the results to.
        filename : string
            Name of the file to write.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, filename)

        # Check if the file exists and if it's empty
        file_exists = os.path.isfile(file_path)
        file_empty = not file_exists or os.stat(file_path).st_size == 0

        fieldnames = list(data[0].keys()) if data else []
        
        with open(file_path, mode='a', newline='') as raw_data_file:
            writer = csv.DictWriter(raw_data_file, fieldnames=fieldnames)
            
            # Write header only if the file is empty
            if file_empty:
                writer.writeheader()
            
            # Write the data rows
            writer.writerows(data)

        print(f"Data written to {file_path}")

    # Print the data
        #print(row)
    #print(data_test[1][1])

    # Convert data_test to a DataFrame if it's not already
    if not isinstance(data_test, pd.DataFrame):
        data_test = pd.DataFrame(data_test)

    for index, row in data_test.iterrows():
        print(index)
        def itmlogic_p2p(main_user_defined_parameters, surface_profile_m):
            """
            Run itmlogic in point to point (p2p) prediction mode.

            Parameters
            ----------
            main_user_defined_parameters : dict
                User defined parameters.
            surface_profile_m : list
                Contains surface profile measurements in meters.

            Returns
            -------
            output : list of dicts
                Contains model output results.

            """
            prop = main_user_defined_parameters

            #DEFINE ENVIRONMENTAL PARAMETERS
            # Terrain relative permittivity
            prop['eps'] = 15

            # Terrain conductivity (S/m)
            prop['sgm']   = 0.001 #0.005

            # Climate selection (1=equatorial,
            # 2=continental subtropical, 3=maritime subtropical,
            # 4=desert, 5=continental temperate,
            # 6=maritime temperate overland,
            # 7=maritime temperate, oversea (5 is the default)
            prop['klim']  =   5

            # Surface refractivity (N-units): also controls effective Earth radius
            prop['ens0']  =   301

            #DEFINE STATISTICAL PARAMETERS
            # Confidence  levels for predictions
            qc = [10, 50, 90]

            # Reliability levels for predictions
            qr = [10, 50, 90]

            # Number of points describing profile -1
            pfl = []
            pfl.append(len(surface_profile_m) - 1)
            pfl.append(0)

            for profile in surface_profile_m:
                pfl.append(profile)

            # Refractivity scaling ens=ens0*exp(-zsys/9460.)
            # (Average system elev above sea level)
            zsys = 0

            # Note also defaults to a continental temperate climate

            # Setup some intermediate quantities
            # Initial values for AVAR control parameter: LVAR=0 for quantile change,
            # 1 for dist change, 2 for HE change, 3 for WN change, 4 for MDVAR change,
            # 5 for KLIM change
            prop['lvar'] = 5

            # Inverse Earth radius
            prop['gma']  = 157E-9

            # Conversion factor to db
            db = 8.685890

            #Number of confidence intervals requested
            nc = len(qc)

            #Number of reliability intervals requested
            nr = len(qr)

            #Length of profile in km
            dkm = prop['d']

            #Profile range step, select option here to define range step from profile
            #length and # of points
            xkm = 0

            #If DKM set <=0, find DKM by mutiplying the profile step by number of
            #points (not used here)
            if dkm <= 0:
                dkm = xkm * pfl[0]

            #If XKM is <=0, define range step by taking the profile length/number
            #of points in profile
            if xkm <= 0:

                xkm = dkm // pfl[0]

                #Range step in meters stored in PFL(2)
                pfl[1] = dkm * 1000 / pfl[0]

                #Store profile in prop variable
                prop['pfl'] = pfl
                #Zero out error flag
                prop['kwx'] = 0
                #Initialize omega_n quantity
                prop['wn'] = prop['fmhz'] / 47.7
                #Initialize refractive index properties
                prop['ens'] = prop['ens0']

            #Scale this appropriately if zsys set by user
            if zsys != 0:
                prop['ens'] = prop['ens'] * math.exp(-zsys / 9460)

            #Include refraction in the effective Earth curvature parameter
            prop['gme'] = prop['gma'] * (1 - 0.04665 * math.exp(prop['ens'] / 179.3))

            #Set surface impedance Zq parameter
            zq = complex(prop['eps'], 376.62 * prop['sgm'] / prop['wn'])

            #Set Z parameter (h pol)
            prop['zgnd'] = np.sqrt(zq - 1)

            #Set Z parameter (v pol)
            if prop['ipol'] != 0:
                prop['zgnd'] = prop['zgnd'] / zq

            #Flag to tell qlrpfl to set prop.klim=prop.klimx and set lvar to initialize avar routine
            prop['klimx'] = 0

            #Flag to tell qlrpfl to use prop.mdvar=prop.mdvarx and set lvar to initialize avar routine
            prop['mdvarx'] = 11

            #Convert requested reliability levels into arguments of standard normal distribution
            zr = qerfi([x / 100 for x in qr])
            #Convert requested confidence levels into arguments of standard normal distribution
            zc = qerfi([x / 100 for x in qc])

            #Initialization routine for point-to-point mode that sets additional parameters
            #of prop structure
            prop = qlrpfl(prop)

            ## Here HE = effective antenna heights, DL = horizon distances,
            ## THE = horizon elevation angles
            ## MDVAR = mode of variability calculation: 0=single message mode,
            ## 1=accidental mode, 2=mobile mode, 3 =broadcast mode, +10 =point-to-point,
            ## +20=interference

            #Free space loss in db
            fs = db * np.log(2 * prop['wn'] * prop['dist'])

            #Used to classify path based on comparison of current distance to computed
            #line-of-site distance
            q = prop['dist'] - prop['dlsa']

            #Scaling used for this classification
            q = max(q - 0.5 * pfl[1], 0) - max(-q - 0.5 * pfl[1], 0)

            #Report dominant propagation type predicted by model according to parameters
            #obtained from qlrpfl
            if q < 0:
                print('Line of sight path')
            elif q == 0:
                print('Single horizon path')
            else:
                print('Double-horizon path')
            if prop['dist'] <= prop['dlsa']:
                print('Diffraction is the dominant mode')
            elif prop['dist'] > prop['dx']:
                print('Tropospheric scatter is the dominant mode')

            print('Estimated quantiles of basic transmission loss (db)')
            print('Free space value {} db'.format(str(fs)))

            print('Confidence levels {}, {}, {}'.format(
                str(qc[0]), str(qc[1]), str(qc[2])))

            qc = [10, 50, 90]

            # Reliability levels for predictions
            qr = [10, 50, 90]

            output = []

            # Create a single row
            row = {
                'latitude_transmitter': lat_t,
                'longitude_transmitter': long_t,
                'latitude_receiver': lat_r,
                'longitude_receiver': long_r,
                'year': year,
                'cod_mun': cod_mun,
                'erp': erp,
                'distance_km': None,  # We'll fill this in the loop
                'free_space': fs
            }

            # Loop through all combinations of reliability and confidence
            for jr in range(len(qr)):
                for jc in range(len(qc)):
                    # Compute corrections to free space loss based on requested confidence
                    # and reliability quantities
                    avar1, prop = avar(zr[jr], 0, zc[jc], prop)
                    
                    # Create a column name based on the reliability and confidence levels
                    column_name = f'propagation_loss_dB_R{qr[jr]}_C{qc[jc]}'
                    
                    # Add the propagation loss to the row
                    row[column_name] = fs + avar1
                    
                    # Set the distance (it should be the same for all combinations)
                    if row['distance_km'] is None:
                        row['distance_km'] = prop['d']

            # Append the single row to the output
            output.append(row)

            return output



        def write_shapefile(data, directory, filename, crs):
            """
            Write geojson data to shapefile.

            Parameters
            ----------
            data : list of dicts
                Data to be written.
            directory : string
                Folder to write the results to.
            filename : string
                Name of the file to write.
            crs : string
                Defines the coordinate reference system.

            """
            prop_schema = []
            for name, value in data[0]['properties'].items():
                fiona_prop_type = next((
                    fiona_type for fiona_type, python_type in \
                        fiona.FIELD_TYPES_MAP.items() if \
                        python_type == type(value)), None
                    )

                prop_schema.append((name, fiona_prop_type))

            sink_driver = 'ESRI Shapefile'
            sink_crs = {'init': crs}
            sink_schema = {
                'geometry': data[0]['geometry']['type'],
                'properties': OrderedDict(prop_schema)
            }

            if not os.path.exists(directory):
                os.makedirs(directory)

            with fiona.open(
                os.path.join(directory, filename), 'w',
                driver=sink_driver, crs=sink_crs, schema=sink_schema) as sink:
                for datum in data:
                    sink.write(datum)


        def straight_line_from_points(a, b):
            """
            Generate a geojson LineString object from two geojson points.

            Parameters
            ----------
            a : geojson
                Point A
            b : geojson
                Point B

            Returns
            -------
            line : geojson
                A geojson LineString object.

            """
            line = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [
                        (
                            a['geometry']['coordinates'][0],
                            a['geometry']['coordinates'][1]
                        ),
                        (
                            b['geometry']['coordinates'][0],
                            b['geometry']['coordinates'][1]
                        ),
                    ]
                },
                'properties': {
                    'id': 'terrain path'
                }
            }

            return line


        if __name__ == '__main__':
            #results_csv_path = os.path.join(results_folder, 'p2p_results.csv')

            # Setup data folder paths
            directory_shapes = os.path.join(DATA_PROCESSED, 'shapes')

            # Set coordinate reference systems
            old_crs = 'EPSG:4326'
            # new_crs = 'EPSG:3857'

            # DEFINE MAIN USER PARAMETERS
            # Define an empty dict for user defined parameters
            main_user_defined_parameters = {}

            # Define radio operating frequency (MHz)
            main_user_defined_parameters['fmhz'] = float(row[4]) # 41.5

            # Define distance between terminals in km (from Longley Rice docs)
            # main_user_defined_parameters['d'] = 77.8

            # Define antenna heights - Antenna 1 height (m), Antenna 2 height (m)
            main_user_defined_parameters['hg'] = [float(row[5]), 1.3]

            # Polarization selection (0=horizontal, 1=vertical)
            main_user_defined_parameters['ipol'] = 1

            year = float(row[6])
            cod_mun = float(row[7])
            erp = float(row[15])

            lat_t = float(row[0])
            long_t = float(row[1])


            # Create new geojson for Crystal Palace radio transmitter
            transmitter = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': (long_t, lat_t)
                },
                'properties': {
                    'id': 'Crystal Palace radio transmitter'
                }
            }

            # Create new geojson for Mursley
            lat_r = float(row[2])		
            long_r = float(row[3])
            

            start_lat = min(int(lat_r), int(lat_t))
            end_lat   = max(int(lat_r), int(lat_t))
            

            start_long = min(int(long_r), int(long_t))
            end_long   = max(int(long_r), int(long_t))

            receiver = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': (long_r, lat_r)
                },
                'properties': {
                    'id': 'Mursley'
                }
            }

            #Create new geojson for terrain path
            line = straight_line_from_points(transmitter, receiver)


            output_combined = '/Users/marcelamello/Documents/GitHub/itmlogic/results/output_combined.tif'

            # Check if the output file already exists
            if os.path.exists(output_combined):
                os.remove(output_combined)
                print(f"Deleted existing file: {output_combined}")
            else:
                print("Aborting TIFF combination.")

            # Combine Tifs       
            def combine_tifs(output_path, input_files):
                command = ['gdal_merge.py', '-o', output_path] + input_files
                try:
                    subprocess.run(command, check=True)
                    #print(f"Combined TIF saved as {output_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error combining TIF files: {e}")



            try:
                # Initialize an empty list to store file names
                tif_files_list = []

                # List all files in the directory
                files = os.listdir(dem_folder)

                # Iterate over each file in the directory
                for file in files:
                    # Extract the numeric part from the filename
                    if file.endswith('num.tif'):
                        pattern = r'(\w{1})(\d{2})W0(\d{2})_num\.tif'
                        matches = re.findall(pattern, file)
                        hemisphere = matches[0][0]  # Extract the first matched number (Sxx)
                        #print(matches)
                        if matches:
                            w_number = -int(matches[0][2])  # Extract the second matched number (Wxx)
                            #print(w_number)

                            if hemisphere == 'N' :
                                s_number = int(matches[0][1])  # Extract the first matched number (Sxx)
                                #print(s_number)
                            else:
                                s_number = -int(matches[0][1])  # Extract the first matched number (Sxx)
                                #print(s_number)

                            if lat_r <0 and lat_t<0:
                                if (start_long-1 <= w_number <= end_long+1) and (start_lat-1 <= s_number <= end_lat+1 and  hemisphere == 'S'): #or (0 <= s_number <= 1 and hemisphere == 'N')):
                                        # Append the full path of the file to the list
                                    tif_files_list.append(os.path.join(dem_folder, file))
                                        #print({matches})
                                    #print(f"Extracted numbers from {file}: {matches}")

                            if (lat_r >0 and lat_t<0) or (lat_r <0 and lat_t>0):
                                if (start_long-1 <= w_number <= end_long+1) and ((start_lat-1 <= s_number <= 0 and hemisphere == 'S') or (0 <= s_number <= end_lat+1 and hemisphere == 'N')):
                                        # Append the full path of the file to the list
                                    tif_files_list.append(os.path.join(dem_folder, file))
                                        #print({matches})
                                    #print(f"Extracted numbers from {file}: {matches}")

                            if lat_r >0 and lat_t>0:
                                if (start_long-1 <= w_number <= end_long+1) and (start_lat-1 <= s_number <= end_lat+1 and hemisphere == 'N'): # or (-1 <= s_number <= 0  and hemisphere == 'S')):
                                        # Append the full path of the file to the list
                                    tif_files_list.append(os.path.join(dem_folder, file))
                                        #print({matches})
                                    #print(f"Extracted numbers from {file}: {matches}")

                # Combine the TIF files if there are any
                if tif_files_list:
                    combine_tifs(output_combined, tif_files_list)
                #    print("'dem.tif' files found to combine.")
                else:
                    print("No 'dem.tif' files found to combine.")




            except OSError as e:
                # Handle any errors that occur
                print(f'Error: {e}')

            # Process the combined TIF file
            measured_terrain_profile, distance_km, points = terrain_p2p(output_combined, line)
            print('Distance is {}km'.format(distance_km))

            # Define distance between terminals in km (from Longley Rice docs)
            main_user_defined_parameters['d'] = distance_km

            # Check (out of interest) how many measurements are in each profile
            print('len(measured_terrain_profile) {}'.format(len(measured_terrain_profile)))

            # Run model and get output
            output = itmlogic_p2p(main_user_defined_parameters, measured_terrain_profile)

            # Grab coordinates for transmitter and receiver for writing to .csv
            transmitter_x = transmitter['geometry']['coordinates'][0]
            transmitter_y = transmitter['geometry']['coordinates'][1]
            receiver_x = receiver['geometry']['coordinates'][0]
            receiver_y = receiver['geometry']['coordinates'][1]

            transmitter_shape = []
            transmitter_shape.append(transmitter)
            write_shapefile(transmitter_shape, directory_shapes, 'transmitter.shp', old_crs)

            receiver_shape = []
            receiver_shape.append(receiver)
            write_shapefile(receiver_shape, directory_shapes, 'receiver.shp', old_crs)

            write_shapefile(points, directory_shapes, 'points.shp', old_crs)

            # Write results to .csv
            csv_writer(output,results_folder,  results_file)
            

            print('Completed run')