import os
import csv
import sys
import shutil
import overpy
import json
from glob import glob
import shapely.geometry
import geojson
import geopandas as gpd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, draw
import rdp

quadkey_lib = r"libs"
sys.path.append(quadkey_lib)
import QuadKey.quadkey as quadkey



# Read GeoJSON from a file
def read_geojson_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_osm_building(input_geojson, osm_db, name):

    api = overpy.Overpass()

    # Extract coordinates from GeoJSON
    coordinates = input_geojson['features'][0]['geometry']['coordinates'][0]

    # Create a bounding box from the coordinates
    min_lon, min_lat = min(coordinates, key=lambda c: c[0])[0], min(coordinates, key=lambda c: c[1])[1]
    max_lon, max_lat = max(coordinates, key=lambda c: c[0])[0], max(coordinates, key=lambda c: c[1])[1]

    # Query the Overpass API for building footprints within the bounding box
    query = f"""
    way
    [building]
    ({min_lat},{min_lon},{max_lat},{max_lon});
    (._;>;);
    out;
    """
    result = api.query(query)

    # Return the first building footprint found (if any)
    print(f"Found {len(result.ways)} buildings.")
    if len(result.ways) > 0:
        # way = result.ways[0]
        j=0
        for way in result.ways:
        # save the geojson file
            with open(os.path.join(osm_db, f'{name}_{j}.geojson'), 'w') as file:
                rawNodes = []
                for node in way.get_nodes(resolve_missing=True):
                    rawNodes.append( (node.lon, node.lat) )
                try:
                    geom = shapely.geometry.Polygon(rawNodes)
                    tags = way.tags
                    tags['wayOSMId'] = way.id

                    features = []            
                    features.append( geojson.Feature(geometry=geom, properties=tags))
                    featureCollection = geojson.FeatureCollection(features)
                    # print(featureCollection)
                    
                    file.write(geojson.dumps(featureCollection))
                    j+=1        
                except Exception as expt:
                    print(f"Error: {expt}")
        return result.ways[0]
    else:
        return None


# Process multiple GeoJSON files
def process_geojson_files(file_paths, osm_db, limit=10):
    mapping = {}
    osm_not_found = []
    i=0
    for file_path in file_paths:
        if i==limit:
            break
        # Read the GeoJSON file
        geojson = read_geojson_file(file_path)

        # Find the corresponding OSM building
        index = file_path.split('\\')[-1].split('.')[0]
        osm_building = find_osm_building(geojson, osm_db, index)

        # Process the result
        if osm_building:
            print(f"OSM Building ID: {osm_building.id}")
            osi_path = file_path
            osm_path = os.path.join(osm_db, str(osm_building.id)+'.geojson')
            # Store the mapping
            mapping[osm_building.id] = {
                'osi_path': osi_path,
                'osm_path': osm_path
            }
            # print()
        else:
            # print(f"GeoJSON File: {file_path}")
            print("No corresponding OSM building found.")
            osm_not_found.append(file_path)
        i+=1
    return mapping, osm_not_found


# problem: some osi files have more than one osm file
# solution: use the osm file with the largest intersection area
def calculate_intersection_percentage(geojson_file1, geojson_file2):
    # Read GeoJSON files using geopandas
    gdf1 = gpd.read_file(geojson_file1)
    gdf2 = gpd.read_file(geojson_file2)

    # Perform intersection
    intersection = gpd.overlay(gdf1, gdf2, how='intersection')

    # Calculate the area of intersection and total area of the first GeoJSON
    intersection_area = intersection.geometry.area.sum()
    total_area1 = gdf1.geometry.area.sum()

    # Calculate the percentage of intersection
    percentage = (intersection_area / total_area1) * 100

    return percentage

def plot_figures(figures, nrows=1, ncols=1, figsize=(25, 25)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    if len(figures) == 1:
        axeslist.imshow(figures[0], cmap=plt.gray())
        axeslist.set_title(0, fontsize=40)
        axeslist.set_axis_off()
    else:
        for ind, title in enumerate(figures):
            axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional

def getContours(image_path, min_area=50):
    image = cv2.imread(image_path, 0)
    maxImageSize = image.shape[0] * 3
    image = cv2.resize(
        image, (maxImageSize, maxImageSize), interpolation=cv2.INTER_CUBIC
    )

    blur = cv2.medianBlur(image, 5)
    thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    true_cnts = []
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area > min_area:
            true_cnts.append(c)
    return true_cnts, image


def draw_contours(image, contours):
    # image = cv2.imread(image_path, 0)
    maxImageSize = image.shape[0]
    image = cv2.resize(
        image, (maxImageSize, maxImageSize), interpolation=cv2.INTER_CUBIC
    )
    imageNoMasks = np.copy(image)
    cnts_fig = {}
    for i, c in enumerate(contours):
        mask = np.zeros(image.shape, image.dtype)
        xs, ys = map(list, zip(*c[:, 0, :].tolist()))
        r, c = draw.polygon(xs, ys, (maxImageSize, maxImageSize))
        mask[c, r] = 255
        cnts_fig[i] = mask
        image = np.copy(imageNoMasks)
    plot_figures(figures=cnts_fig, nrows=1, ncols=len(cnts_fig))

def image2geojson(data_path, feature_index, building_coords, quadKeyStr, qkRoot, tilePixel):
    coords = np.array([i[0] for i in building_coords.tolist()])
    l = len(coords)
    coords = coords.reshape((l,1,2))

    geo_tagged_coords = []
    for pt in coords:
            geo = quadkey.TileSystem.pixel_to_geo( (pt[0,0]+tilePixel[0],pt[0,1]+tilePixel[1]),qkRoot.level)
            #https://wiki.openstreetmap.org/wiki/Node
            geo_tagged_coords.append((geo[1],geo[0]))

    # Create a Polygon feature from the coordinates
    polygon = geojson.Polygon([geo_tagged_coords])
    # Create a FeatureCollection from the Polygon feature
    feature_collection = geojson.FeatureCollection([geojson.Feature(geometry=polygon)])
    jsonFileName = os.path.join(data_path, f'{quadKeyStr}_{feature_index}.geojson')
    # Save the FeatureCollection to a file
    with open(jsonFileName, "w") as f:
            geojson.dump(feature_collection, f)
    print(f"> {len(geo_tagged_coords)} coordinates saved to {quadKeyStr}_{feature_index}.geojson")
    return l


root_dir = r"C:\Users\lasit\Desktop\DeepMapper\frontend\DeepMapper-frontend\data\d2ae6c59-8e60-4921-918e-c444125e9985"
osi_dir = os.path.join(root_dir, "osi_coco_2")
osm_dir = os.path.join(root_dir, "coco_images")
gan_res_dir = os.path.join(root_dir, "osm-gan", "results", "osm-gan-carswell-23", "test_latest", "images")
osm = "Exp_1/data/osm_2"
osi = "Exp_1/data/osi_2"
images = "Exp_1/data/image_2"

for test_case in os.listdir(osi_dir)[588:]:

    print(">>> Processing {}".format(test_case))

    osm = "Exp_1/data/osm_2"
    osi = "Exp_1/data/osi_2"
    images = "Exp_1/data/image_2"
    
    osi_txt = os.path.join(osi_dir, test_case, test_case + ".txt")
    osm_txt = os.path.join(osm_dir, test_case, test_case + ".txt")

    # read osi txt
    osi_list = []
    with open(osi_txt, "r") as f:
        # ignore first two lines
        next(f)
        next(f)
        for line in f:
            line = line.strip()
            osi_list.append(line)

    # read osm txt
    osm_list = []
    with open(osm_txt, "r") as f:
        # ignore first two lines
        next(f)
        next(f)
        for line in f:
            line = line.strip()
            osm_list.append(line)
    
    print(len(osi_list))
    print(len(osm_list))
    
    osm = os.path.join(osm, test_case)
    osi = os.path.join(osi, test_case)
    image_path = os.path.join(images, test_case)

    # create folder
    if not os.path.exists(osm):
        os.makedirs(osm)
    if not os.path.exists(osi):
        os.makedirs(osi)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for file in osi_list:
        gfile = file.split(".")[0] + ".geojson"
        # copy file to osi
        print("copy {} to {}".format(gfile, osi))
        shutil.copy(gfile, osi)
    
    mapping, errors = process_geojson_files(glob(os.path.join(osi, '*.geojson')), osm, limit=len(osi_list))
    print(mapping)
    print(errors)

    # acqurie the osm files
    for file in glob(os.path.join(osi, '*.geojson')):
        idx = file.split('\\')[-1].split('.')[0]
        osm_files = glob(os.path.join(osm, idx + '*.geojson'))
        print(f"processing {idx}")
        print(f"{len(osm_files)} osm files found")
        if len(osm_files) < 2:
            continue
        ious = []
        for osm_file in osm_files:
            iou = calculate_intersection_percentage(file, osm_file)
            ious.append(iou)
            print(f"{osm_file} -> {iou}")
        
        max_iou = max(ious)
        max_idx = ious.index(max_iou)
        print(f"max osm file: {osm_files[max_idx]}")
        # delete other osm files
        for i in range(len(osm_files)):
            if i != max_idx:
                os.remove(osm_files[i])
                print(f"deleted {osm_files[i]}")
    
    # rename osm files to remove _0 and _1
    for file in glob(os.path.join(osm, '*.geojson')):
        idx = file.split('\\')[-1].split('.')[0].split('_')[0]
        os.rename(file, os.path.join(osm, idx + '.geojson'))

    
    gan_prediction = os.path.join(gan_res_dir, test_case + '_fake_B.png')


    contours, image = getContours(gan_prediction, min_area=250)
    # draw_contours(image, contours)
    # plt.show()

    with open(osm_txt, "r") as f:
        quadKeyStr = f.readline().strip()

    qkRoot = quadkey.from_str(quadKeyStr)
    tilePixel = quadkey.TileSystem.geo_to_pixel(qkRoot.to_geo(), qkRoot.level)
    
    possible_changes = contours
    maxImageSize = 256 * 3
    j=0
    for pred_building in possible_changes:
        pred_mask = np.zeros((maxImageSize, maxImageSize), dtype=np.uint8)
        coords = np.array([i[0] for i in pred_building.tolist()])
        coords = coords.reshape((len(coords),1,2))
        # apply RDP algorithm to reduce the number of points
        rdp_coords = rdp.rdp(coords, epsilon=0.9)
        rdp_coords = np.array(rdp_coords)
        image2geojson(image_path, j, rdp_coords, quadKeyStr, qkRoot, tilePixel)
        print(f"{len(coords)} -> {len(rdp_coords)}")
        _xs, _ys = map(list, zip(*rdp_coords[:, 0, :].tolist()))
        pred_mask.fill(0)
        r, c = draw.polygon(
            _xs, _ys, (maxImageSize, maxImageSize)
        )  # , clip=True)
        pred_mask[c, r] = 255
        io.imsave(f"{image_path}/{quadKeyStr}_{j}.png", pred_mask, check_contrast=False)
        j+=1

    predicted_geojsons = glob(os.path.join(image_path, '*.geojson'))
    geo_map = {}

    # Precompute intersection percentages
    iou_matrix = {}
    for osm_file in glob(os.path.join(osi, '*.geojson')):
        osm_idx = os.path.splitext(os.path.basename(osm_file))[0]
        iou_matrix[osm_idx] = {}
        
        for pred_file in predicted_geojsons:
            pred_idx = os.path.splitext(os.path.basename(pred_file))[0]
            iou = calculate_intersection_percentage(osm_file, pred_file)
            iou_matrix[osm_idx][pred_idx] = iou

    # Find the maximum IoU for each OSM file
    for osm_file in glob(os.path.join(osi, '*.geojson')):
        osm_idx = os.path.splitext(os.path.basename(osm_file))[0]
        max_iou = 0
        max_pred = None
        
        for pred_file in predicted_geojsons:
            pred_idx = os.path.splitext(os.path.basename(pred_file))[0]
            iou = iou_matrix[osm_idx][pred_idx]
            
            if iou > max_iou:
                max_iou = iou
                max_pred = pred_file
            
            print(f"{pred_file} -> {iou}")
        
        geo_map[osm_file] = max_pred
        print(f"Max OSM file: {max_pred}")

    print(geo_map)
    # rename the predicted geojson files to match the osi files using the geo_map
    for key, value in geo_map.items():
        key_name = os.path.basename(key)
        if value is not None:
            if os.path.exists(value):
                value_name = os.path.basename(value)
                print(f"renaming {value_name} to {key_name}")
                try:
                    os.rename(value, os.path.join(image_path, key_name))
                    # rename the image
                    os.rename(os.path.join(image_path, value_name.split('.')[0] + '.png'), os.path.join(image_path, key_name.split('.')[0] + '.png'))
                except FileExistsError as expt:
                    print(f"Error: {expt}")
        else:
            print(f"no matching file for {key_name}")
    
    print("=====================================================")