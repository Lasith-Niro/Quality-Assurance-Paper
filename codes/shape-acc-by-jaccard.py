from shapely.geometry import shape
import geopandas as gpd
from pyproj import CRS, Transformer
import numpy as np
import os
from utils import load_processed_data, wgsto2157

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# TODO:
# 1. select 1-1 mapping footprint pairs
# 2. convert to ITM crs
# 2. calculate the Jaccard index 

samples = load_processed_data('Exp_0/processed_data')
all_osi = [i[0] for i in samples]
all_osm = [i[1] for i in samples]
all_pred = [i[2] for i in samples]
itm_crs = CRS.from_epsg(2157)

csv_file = "jaccard_values_osm_paper.csv"
category = "osm"

processed = []
jaccard_values = {}
for osi_geojson in all_osi:
    # print(f"{osi_geojson} is being processed...")
    osi_filename = os.path.basename(osi_geojson)
    osi_geojson = gpd.read_file(osi_geojson)
    osi_shape = shape(osi_geojson.iloc[0].geometry)
    
    osm_geojson = os.path.join('Exp_0', 'processed_data', category, osi_filename)
    osm_filename = os.path.basename(osm_geojson)
    if osm_filename not in processed:
        if os.path.exists(osm_geojson):
            osm_geojson = gpd.read_file(osm_geojson)
            osm_shape = shape(osm_geojson.iloc[0].geometry)
            if osi_shape.intersects(osm_shape):
                processed.append(osm_filename)

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


                # Calculate the intersection of the geometries
                intersection = gpd.overlay(osm_gdf, osi_gdf, how='intersection')

                # Calculate the union of the geometries
                union = gpd.overlay(osm_gdf, osi_gdf, how='union')

                # Calculate the Jaccard index
                jaccard_index = intersection.area.sum() / union.area.sum()

                # save into a dictionary
                jaccard_values[osm_filename] = jaccard_index

                # Print the Jaccard index
                print(f"Jaccard Index ({osm_filename}): {jaccard_index:.2f}")

np_jaccard_values = np.array(list(jaccard_values.values()))
mean = np.mean(np_jaccard_values)
std = np.std(np_jaccard_values)
highest = np.max(np_jaccard_values)
lowest = np.min(np_jaccard_values)

print("Jaccard Similarity Coefficient Metrics:")
print(f"Total number of samples: {len(np_jaccard_values)}")
print(f"Mean: {mean:.2f}")
print(f"Std. Deviation: {std:.2f}")
print(f"Highest: {highest:.2f}")
print(f"Lowest: {lowest:.2f}")

# save jaccard_values to csv
import csv
with open(f'{csv_file}.csv', 'w') as f:
    for key in jaccard_values.keys():
        f.write("%s,%s\n"%(key,jaccard_values[key]))
