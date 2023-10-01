import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import shape
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import load_processed_data, wgsto2157
from ShapeDiscriptor import ShapeDiscriptor

def area_by_shoelace(coords):
    transformed_coords = coords #wgsto2157(coords)
    # print("Transformer: ", transformed_coords)
    area = 0
    for i in range(len(transformed_coords)-1):
        area += transformed_coords[i][1]*transformed_coords[i+1][0] - transformed_coords[i+1][1]*transformed_coords[i][0]
    area = abs(area/2)    
    return area

def area(coords):
    return area_by_shoelace(coords)

def perimeter(coords):
    return np.sum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1)))

samples = load_processed_data('Exp_0\processed_data')
all_osi = [i[0] for i in samples]
all_osm = [i[1] for i in samples]
all_pred = [i[2] for i in samples]
itm_crs = CRS.from_epsg(2157)
category = "osm"

processed = []
osm_sci_values = {}
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

                shape_disc = ShapeDiscriptor(osm_gdf)
                a = area(ITM_osm_coords)
                p = perimeter(ITM_osm_coords)
                r =  shape_disc.roundness()
                e = shape_disc.elongation()
                c = shape_disc.convexity()
                co = shape_disc.compactness()
                s = shape_disc.solidity()
                sci = (p*e*c)/(a*co*s)

                osm_sci_values[osm_filename] = sci

# save to csv
df = pd.DataFrame.from_dict(osm_sci_values, orient='index', columns=['SCI'])
df.index.name = 'file'
df.to_csv(os.path.join(f"{category}_paper.csv"))

# read osm_paper.csv
df = pd.read_csv(os.path.join(f"osm_paper.csv"))
np_acc_values = np.array(df['SCI'].values)
np_acc_values = np.array(list(osm_sci_values.values()))
mean = np.mean(np_acc_values)
std = np.std(np_acc_values)
highest = np.max(np_acc_values)
lowest = np.min(np_acc_values)

print("SCI Metrics for OSM footprints:")
print(f"Total number of samples: {len(np_acc_values)}")
print(f"Mean: {mean:.2f}")
print(f"Std. Deviation: {std:.2f}")
print(f"Highest: {highest:.2f}")
print(f"Lowest: {lowest:.2f}")