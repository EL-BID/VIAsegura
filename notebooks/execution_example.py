# ## Viasegura

import os
from tqdm.notebook import tqdm
from pathlib import Path
import json

### Viasegura
from viasegura import ModelLabeler
from viasegura import LanesLabeler

### Analyze
import pandas as pd
import numpy as np
import tensorflow as tf

### graphics
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import geopandas as gpd
import plotly.express as px

### Carga de imagenes en memoria
def load_image(routes):
    imgs = np.array([tf.image.decode_image(tf.io.read_file(str(route))).numpy() for route in tqdm(routes)])
    return imgs

### Creacion de capas para mapa
def create_layer(color,gdf_work):
    Path('files').mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(gdf_work.loc[gdf_work.color==color].reset_index(drop=True)[['geometry','id']]).to_file(f'files/file_{color}.json', driver = "GeoJSON")
    with open(f'files/file_{color}.json') as geofile:
        j_file = json.load(geofile)
        i=0
        for feature in j_file["features"]:
            feature['id'] = str(i).zfill(5)
            i += 1
    layer = {
            'source': {
              'type': "FeatureCollection",
              'features': j_file['features'],
              'name':color,

            },
            'type': 'line',
            'below': 'traces',
            "color": color
          }
    return layer

### joining multiple linestring
def multi(x):
	  lista = list(x)
	  return MultiLineString(lista)

### Creacion de mapa
def get_graph(data, varia):
    colors_dict = {
        'Present':'#22a800',
        'Adequate':'#22a800',
        'Poor':'#ee0b00',
        'Not present':'#ee0b00',
        'Carriageway of a divided road':'#1ac7b0',
        'Undivided road':'#6817d1',
        'Urban':'#1ac7b0',
        'Rural':'#6817d1',
        'Underdeveloped areas': '#1ac7b0',
        'Unknown':'#6817d1',
        'One': '#1ac7b0',
        'Two': '#6817d1',
        'Three':'#cc8e12'
    }
    variable = varia
    data['color'] = list(map(lambda x:colors_dict.get(x,'#000000'), data[variable].values))
    data['geometry'] = list(map(lambda LS,LaS, LE, LaE:LineString([[LS,LaS], [LE,LaE]]),data.longitud_first, data.latitud_first, data.longitud_last, data.latitud_last))
    data = gpd.GeoDataFrame(data)
    data['centroid'] = list(map(lambda x: x.centroid, data.geometry.values))
    data['coords'] = list(map(lambda x: x.coords, data.centroid.values ))
    data['latitud'] = list(map(lambda x: x[0][1], data.coords.values))
    data['longitud'] = list(map(lambda x: x[0][0], data.coords.values))
    center = {'lat':data.centroid.y.mean(),'lon':data.centroid.x.mean()}
    data_g2 = gpd.GeoDataFrame(data.groupby(['color']).aggregate({'geometry':lambda x:multi(x)}).reset_index())
    data_g2['geometry'] = list(map(lambda x: linemerge(x), data_g2.geometry.values))
    data_g2['id'] = [str(i).zfill(5) for i in range(len(data_g2))]
    layers = []
    list(tqdm(map(lambda color: layers.append(create_layer(color,data_g2)), list(data_g2.color.unique()) )))
    fig_map = px.scatter_mapbox(data,
        lat='latitud',
        lon='longitud',
        # color=variable,
        hover_data=[variable],
        opacity=0
    )
    fig_map.update_layout(
        showlegend=False,
        autosize=True,
        mapbox = {
            'style': "carto-positron",
            'center': center,
            'zoom': 12,
            'layers': layers,
            },
        margin={"r":0,"t":0,"l":0,"b":0},
        )
    return fig_map

# ### Inputs
work_path = Path('./examples')
master_input_frontal = work_path / 'images' / 'frontal'
master_input_lateral = work_path / 'images' / 'lateral'
gps_data_route = work_path / 'gps_info' / 'example.csv'
BATCH_SIZE = 2

# ### Carga de imagenes
archives_frontal = sorted(os.listdir(master_input_frontal))
archives_frontal_paths = [master_input_frontal/item for item in archives_frontal]
frontal_images = load_image(archives_frontal_paths)

archives_lateral = sorted(os.listdir(master_input_lateral))
archives_lateral_paths = [master_input_lateral/item for item in archives_lateral]
lateral_images = load_image(archives_lateral_paths)

number_groups = frontal_images.shape[0]//5 + (1 if (frontal_images.shape[0]%5)>0 else 0)
print(f'El numero de grupos es de {number_groups}')


# device='/device:GPU:0'
device='/device:CPU:0'
system_path='../'
frontallabeler = ModelLabeler(system_path=system_path, model_type = 'frontal', model_filter = ['delineation','street_lighting','carriageway'], device=device)
laterallabeler = ModelLabeler(system_path=system_path, model_type = 'lateral', device=device)
lanes_labeler = LanesLabeler(system_path=system_path, models_device=device)

### Frontal Labeler Execution
frontal_results = frontallabeler.get_labels(frontal_images, batch_size = BATCH_SIZE)

### Lanes Labeler Execution
lanes_results = lanes_labeler.get_labels(frontal_images, batch_size = BATCH_SIZE)

### Lateral Labeler Execution
lateral_results = laterallabeler.get_labels(lateral_images, batch_size = BATCH_SIZE)

frontal_results.keys()

laterallabeler.classes

# ## Resultados
results_df = pd.concat([pd.DataFrame(result['clasification']) for result in [frontal_results, lateral_results, lanes_results]], axis=1)
results_df

gps_data = pd.read_csv(gps_data_route, sep = ';', decimal = ',')
gps_data['image_number'] = list(map(lambda x: int(x.split('.')[0].split('_')[0]), gps_data.img_cen.values))
gps_data['group'] = list(map(lambda x:(x-1)//5, gps_data['image_number']))
gps_data = gps_data.groupby(['group']).aggregate({'latitud':['first','last'], 'longitud':['first','last']}).reset_index()
gps_data.columns = [col[0] if col[1]=='' else f'{col[0]}_{col[1]}' for col in gps_data.columns]
gps_data

results_df = pd.concat([gps_data, results_df ], axis=1)

map_variable = get_graph(results_df, 'street_lighting')## delineation,	carriageway,	street_lighting,	area_type,	land_use,	number_of_lanes
map_variable
