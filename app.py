import pandas as pd
import numpy as np
import os
from contextlib import suppress
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle  
import requests
from flask import  Flask, request, jsonify, render_template, send_file, safe_join, abort
from commons import *

with open('dist/dict_columns_groups.pkl', 'rb') as pkl:
        dict_columns_groups = pickle.load(pkl)
        
with open('dist/columns_not_categorized.pkl', 'rb') as pkl:
        columns_not_categorized = pickle.load(pkl)
        
with open('dist/lgbm_clf.pkl', 'rb') as pkl:
        LightGBM = pickle.load(pkl)
        
with open('dist/sc.pkl', 'rb') as pkl:
        sc = pickle.load(pkl)

df_scaled = pd.read_csv('df_scaled.csv',sep=",",index_col='indice')
data_index = pd.read_csv('data_index.csv',sep=",",index_col='indice.1')

# Iniciamos nuestra API
app = Flask(__name__)
app.config["FILES_PATH"] = "Full_MIDI"

@app.route("/search_titles",methods=['GET'])
def search_titles(data_index=data_index, sc=sc):   
    search=request.args['search']
    page_number =int(request.args['page_number'])
    cant_resultados = int(request.args['cant_resultados'])

    page_from = page_number*cant_resultados
    page_to = page_from + cant_resultados
    print(page_from,page_to)
    result = data_index[data_index['indice_lower_search'].str.contains(search.lower())].indice[page_from:page_to].to_dict()
    response = jsonify(result)
    # Enable Access-Control-Allow-Origin
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route("/related_songs",methods=['GET'])
def get_related_songs(df_scaled=df_scaled, sc=sc):   
    data=df_scaled.copy()
    search=request.args['search']
    predict=request.args['predict']
    filtrar=request.args['filtrar']
    duracion_notas =float(request.args['duracion_notas'])
    amplitud_tonal = float(request.args['amplitud_tonal'])
    ritmica_instrument = float(request.args['ritmica_instrument']) 
    ritmica_drums = float(request.args['ritmica_drums'])
    armonia = float(request.args['armonia'])
    dinamica = float(request.args['dinamica'])
    instrumentacion = float(request.args['instrumentacion'])
    tempo = float(request.args['tempo'])
    notas_simultaneas = float(request.args['notas_simultaneas']) 
    duracion_tema = float(request.args['duracion_tema']) 
    cant_resultados = int(request.args['cant_resultados'])
    others = float(request.args['others'])
    penalizacion = float(request.args['penalizacion'])
    page_number =int(request.args['page_number'])
    clusterizar=request.args['clusterizar']
    
    print(search)
    search_song = data_index[data_index['indice_lower'] == search.lower()].indice.iloc[0]
    new_midi = ""
    cluster = ""
    if (search.lower() == 'false'):
        new_midi = get_midi_from_path()   
        tema = new_midi.tema 
        midi_df_cols = pd.DataFrame(columns=data.columns).drop(['Cluster'],axis=1)
        mask_columns = list(set(df_scaled.columns) & set(new_midi.columns))
        new_midi = new_midi[mask_columns]
        midi_df = pd.concat([midi_df_cols, new_midi])
        midi_df.fillna(0, inplace=True)
        new_midi = sc.transform(midi_df) 
        cluster = LightGBM.predict(new_midi)[0]
        print('predicted cluster:',cluster) 
        midi_df = pd.DataFrame(new_midi, columns=midi_df_cols.columns)
        midi_df['Cluster'] = cluster
        data = pd.concat([data, midi_df]) 
        path = '..\GET_FILE\\'         
        data.iloc[data.shape[0] - 1:data.shape[0],:].index = path + tema
        search_song = data.iloc[data.shape[0] - 1:data.shape[0],:].index    
        print('tema:',search_song)
    else:
        cluster = data[data.index == search_song].Cluster.iloc[0]     
        print('cluster:',cluster)   
        
    mask_cluster = data['Cluster'] == cluster
    if (clusterizar.lower() == 'true'):
        data = data[mask_cluster]
        print(cluster, data.shape)
    
    dict_key_values = {'instrumentacion':instrumentacion,
                     'ritmica_drums':ritmica_drums ,
                     'ritmica_instrument':ritmica_instrument ,
                     'amplitud_tonal':amplitud_tonal ,
                     'dinamica':dinamica, 
                     'duracion_notas1':duracion_notas, 
                     'duracion_notas2':duracion_notas, 
                     'notas_simultaneas':notas_simultaneas ,
                     'tempo':tempo, 
                     'duracion_tema':duracion_tema ,
                     'armonia1':armonia, 
                     'armonia2':armonia, 
                     'armonia3':armonia, 
                     'armonia4':armonia}

    min_sim_distance = 1 / 10**penalizacion 
    
    for keys in dict_key_values.keys():
        cant_keys = len(dict_columns_groups[keys])
        for column in dict_columns_groups[keys]:
            data[column] = (data[column] / np.sqrt(cant_keys)) * (dict_key_values[keys] + min_sim_distance)

    for column in columns_not_categorized:
        data[column] = (data[column] / np.sqrt(cant_keys) ) * (others + min_sim_distance)
     
    song = data.loc[search_song,:]
    if (isinstance(song, pd.Series) == False):
        song = song.iloc[0]
    
    if (filtrar.lower() == 'true'):
        song_values_mask = (song > 0).index
        data = data.loc[:,song_values_mask]
    print(type(data.loc[search_song,:]))
    
    similarity = cosine_similarity_row(data.to_numpy(), song.array, data.index)
    page_from = page_number*cant_resultados
    page_to = page_from + cant_resultados
    result = similarity.iloc[page_from:page_to:].reset_index()
    result.columns = ['path', 'value']
    result = result.to_dict()
    response = jsonify(result)
    # Enable Access-Control-Allow-Origin
    response.headers.add("Access-Control-Allow-Origin", "*")
    
    return response

@app.route("/get-file/<path:filename>")
def get_file(filename):
    safe_path = safe_join(app.config["FILES_PATH"], filename)
    print(safe_path)
    try:
        response = send_file(safe_path, as_attachment=True)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':  
    app.run(host='0.0.0.0', debug=True)       