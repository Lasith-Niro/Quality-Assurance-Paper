from shapely.geometry import shape
from shapely.ops import unary_union
import geopandas as gpd
import os
from tqdm import tqdm
from utils import load_processed_data

import warnings
warnings.filterwarnings('ignore')

"""
    1:1 = OSi footprint matches exactly one OSM footprint
    1:0 = OSi footprint not matches with any OSM footprint 
    1:many = osi footprint matches with more than one osm footprint
    many:1 = more than one OSi footprints match with one OSM footprint
    many: many  = more than one OSi footprints match with more than one OSM footprints
"""

def calculate_completeness(osi, osm):
    relationships = {}
    osm_relationships = {}
    # Count the number of intersections
    for osi_geojson in tqdm(osi, total=len(osi), desc="Calculating completeness"):
        osi_filename = os.path.basename(osi_geojson)
        relationships[osi_filename] = []
        osi_geojson = gpd.read_file(osi_geojson)
        osi_shape = shape(osi_geojson.geometry.iloc[0])
        for osm_geojson in osm:
            osm_filename = os.path.basename(osm_geojson)
            osm_geojson = gpd.read_file(osm_geojson)
            osm_shape = shape(osm_geojson.geometry.iloc[0])
            if osi_shape.intersects(osm_shape):
                relationships[osi_filename].append(osm_filename)
                osm_relationships.setdefault(osm_filename, []).append(osi_filename)

    completeness_1_1 = 0
    completeness_1_0 = 0
    completeness_1_many = 0
    completeness_many_1 = 0
    completeness_many_many = 0

    # Calculate completeness metrics based on the relationships
    for osi_filename, osm_filenames in relationships.items():
        num_osm_files = len(osm_filenames)

        if num_osm_files == 1:
            completeness_1_1 += 1
        elif num_osm_files == 0:
            completeness_1_0 += 1
        elif num_osm_files > 1:
            completeness_1_many += 1

    for osm_filename, osi_filenames in osm_relationships.items():
        num_osi_files = len(osi_filenames)

        if num_osi_files == 1:
            completeness_many_1 += 1
        elif num_osi_files > 1:
            completeness_many_many += 1

    total_files = len(osi) + len(osm)


    # Calculate the completeness percentages
    completeness_1_1 = (completeness_1_1 / total_files) * 100
    completeness_1_0 = (completeness_1_0 / total_files) * 100
    completeness_1_many = (completeness_1_many / total_files) * 100
    completeness_many_1 = (completeness_many_1 / total_files) * 100
    completeness_many_many = (completeness_many_many / total_files) * 100

    return {
        "1:1": completeness_1_1,
        "1:0": completeness_1_0,
        "1:many": completeness_1_many,
        "many:1": completeness_many_1,
        "many:many": completeness_many_many
    }

samples = load_processed_data("processed_data")
all_osi = [i[0] for i in samples]
all_osm = [i[1] for i in samples]
all_pred = [i[2] for i in samples]

completeness = calculate_completeness(all_osi, all_pred)
print("Completeness based on overlap:")
print("1:1 =", completeness['1:1'])
print("1:0 =", completeness['1:0'])
print("1:many =", completeness['1:many'])
print("many:1 =", completeness['many:1'])
print("many:many =", completeness['many:many'])

print(completeness)


# plot bar chart
import matplotlib.pyplot as plt
import numpy as np

labels = ['1:1', '1:0', '1:many', 'many:1', 'many:many']
values = [completeness['1:1'], completeness['1:0'], completeness['1:many'], completeness['many:1'], completeness['many:many']]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
plt.figure(figsize=(10, 5))
plt.bar(x, values, width, label='Completeness')
plt.xticks(x, labels)
# use percentage to 100 
plt.yticks(np.arange(0, 101, 10))
plt.ylabel('Percentage')
plt.xlabel('Relationship')
plt.title('Completeness based on overlap')
plt.legend()
plt.show()
