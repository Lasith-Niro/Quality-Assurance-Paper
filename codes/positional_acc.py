import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import shape
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import load_processed_data, wgsto2157

samples = load_processed_data('Exp_0\\processed_data')
all_osi = [i[0] for i in samples]
all_osm = [i[1] for i in samples]
all_pred = [i[2] for i in samples]
itm_crs = CRS.from_epsg(2157)

category = "osm"

processed = []
positional_accuracy = {}
for osi_geojson in all_osi:
    # print(f"{osi_geojson} is being processed...")
    osi_filename = os.path.basename(osi_geojson)
    osi_geojson = gpd.read_file(osi_geojson)
    osi_shape = shape(osi_geojson.iloc[0].geometry)
    
    osm_geojson = os.path.join("Exp_0", 'processed_data', category, osi_filename)
    print(osm_geojson)
    osm_filename = os.path.basename(osm_geojson)
    if osm_filename not in processed:
        if os.path.exists(osm_geojson):
            osm_geojson = gpd.read_file(osm_geojson)
            osm_shape = shape(osm_geojson.iloc[0].geometry)
            if osi_shape.intersects(osm_shape):
                processed.append(osm_filename)
                # print(osm_geojson)
                
                osm_gdf = osm_geojson
                osi_gdf = osi_geojson

                osm_coords = osm_gdf["geometry"][0].exterior.coords[:]
                osi_coords = osi_gdf["geometry"][0].exterior.coords[:]
                ITM_osm_coords = wgsto2157(osm_coords)
                ITM_osi_coords = wgsto2157(osi_coords)
                # print(ITM_coords)

                # replace the geometry with the transformed coordinates
                osm_gdf.loc[0, "geometry"] = shape({"type": "Polygon", "coordinates": [ITM_osm_coords]})
                osi_gdf.loc[0, "geometry"] = shape({"type": "Polygon", "coordinates": [ITM_osi_coords]})

                osi_centr = osi_gdf.centroid
                osm_centr = osm_gdf.centroid

                distance = osi_centr.distance(osm_centr)[0]

                positional_accuracy[osm_filename] = distance

                print(f"> Positional Accuracy ({osm_filename}): {distance:.2f} meters") 

np_acc_values = np.array(list(positional_accuracy.values()))
mean = np.mean(np_acc_values)
std = np.std(np_acc_values)
highest = np.max(np_acc_values)
lowest = np.min(np_acc_values)

print("Positional Accuracy Metrics:")
print(f"Total number of samples: {len(np_acc_values)}")
print(f"Mean: {mean:.2f}")
print(f"Std. Deviation: {std:.2f}")
print(f"Highest: {highest:.2f}")
print(f"Lowest: {lowest:.2f}")