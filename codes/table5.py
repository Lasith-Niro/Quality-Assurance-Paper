import sys
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


if __name__ == "__main__":
    samples = load_processed_data('Exp_0\processed_data')
    all_osi = [i[0] for i in samples]
    all_osm = [i[1] for i in samples]
    all_pred = [i[2] for i in samples]
    itm_crs = CRS.from_epsg(2157)
    category = "osm"

    idx = sys.argv[1]

    file_path = os.path.join("Exp_0", "processed_data")
    file_name = os.path.join(file_path, category, f"{idx}.geojson")
    print(f"{file_name} is {os.path.exists(file_name)}")

    osm_gdf = gpd.read_file(file_name)
    # extract the coordinates of the polygon
    coords = osm_gdf["geometry"][0].exterior.coords[:]
    print(coords)
    ITM_coords = wgsto2157(coords)
    print(ITM_coords)

    # replace the geometry with the transformed coordinates
    osm_gdf.loc[0, "geometry"] = shape({"type": "Polygon", "coordinates": [ITM_coords]})
    shape_disc = ShapeDiscriptor(osm_gdf)

    a = area(ITM_coords)
    p = perimeter(ITM_coords)
    r =  shape_disc.roundness()
    e = shape_disc.elongation()
    c = shape_disc.convexity()
    co = shape_disc.compactness()
    s = shape_disc.solidity()
    sci = (p*e*c)/(a*co*s)

    df = pd.DataFrame()
    df["perimeter"] = [p]
    df["elongation"] = [e]
    df["convexity"] = [c]
    df["area"] = [a]
    df["compactness"] = [co]
    df["solidity"] = [s]
    df["SCI"] = [sci]
    print(df)

