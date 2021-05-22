from mido import MidiFile
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import clear_output
import pandas as pd
import numpy as np
import plotly.express as px
import math
import os
import pickle  
from sklearn.preprocessing import StandardScaler

ROOT = ""

## Tabla auxiliar: numeros de notas a nombre y octava
midi_notes = pd.read_csv(ROOT + 'midi_notes.csv',sep=";")
## Tabla auxiliar: Escalas
midi_scales_chords = pd.read_csv(ROOT + 'scales.csv',sep=";")
midi_scales_full = pd.read_csv(ROOT + 'scales_full.csv',sep=";")

## Tabla auxiliar: Golpes de percusion
midi_drum_sounds = pd.read_csv(ROOT + 'drums_sounds.csv',sep=";")
midi_drum_sounds.set_index('note', drop=True, inplace=True)
midi_drum_sounds.drop('sound',axis=1, inplace=True)
midi_drum_sounds_dict = midi_drum_sounds.sound_group.to_dict()

## Tabla auxiliar: Instrumentos de General MIDI agrupados
midi_instruments = pd.read_csv(ROOT + 'instruments.csv',sep=";")

## Levanta el archivo midi, devuelve cantidad de tracks y mensajes contenidos
def load_midi_file(files_mid):
    mid = MidiFile(files_mid, clip=False)
    if (mid.length >= 30):
      return mid
    else:
      return 0
        
## Itera sobre los mensaje midi de los tracks con mensjaes y genera un dataframe con notas, velocity, tick_start tick_stop
def get_theme_df(mid):
    dict_notes = {}
    dict_notes_end = {}
    dict_active_notes = {}
    count_notes = 0
    count_notes_end = 0
    last_note_on = 0
    n_track = 0
    n_tracks_used = 0
    tempo = 0
    tempo_changes = 0
    bpm = 0
    time_print = 0
    count_notes_quantified = 0
    controls = []
    key_signatures = []
    time_signatures = []
    dict_time_signature = {}
    dict_time_signature_aux = {}
    dict_time_signature_count = 0    
    ticks = mid.ticks_per_beat
    ticks_quantify = round(ticks / 8)
    for track in mid.tracks:
        track_number = 0
        track_name = track.name + str(n_track)
        n_track = n_track + 1 
        if len(track) > 100:      
            n_tracks_used = n_tracks_used + 1
        time = 0
        has_note_off = any(msg.type == 'note_off' for msg in track)
        for msg in track:
            time = time + msg.time
            time_print = round((time) / ticks_quantify, 0) * ticks_quantify
            if (msg.type in ['note_on', 'note_off']) and (msg.note > 0):
                if (has_note_off and (msg.type == 'note_on')) or (not has_note_off and msg.velocity > 0):
                    if (time_print != time):
                        count_notes_quantified = count_notes_quantified + 1
                    dict_notes[count_notes] = {"note_num": msg.note, "start": time_print, "velocity": msg.velocity, "track_name": track_number, "channel": msg.channel}
                    dict_active_notes[msg.note] = time_print
                    count_notes = count_notes + 1
                    last_note_on = time
                else:
                    dict_notes_end[count_notes_end] = {"note_num": msg.note,"track_name": track_number, "start": dict_active_notes[msg.note], "end": time_print}
                    count_notes_end = count_notes_end + 1
            else:
                if (msg.type == 'control_change'):
                    controls.append(msg.value)
                elif (msg.type == 'key_signature'):
                    key_signatures.append(msg.key)
                elif (msg.type == 'time_signature'):
                    time_signatures.append(str(msg.numerator) + '/' + str(msg.denominator))
                    if (dict_time_signature_count != 0):
                        dict_time_signature[dict_time_signature_count] = {"start": dict_time_signature_aux[dict_time_signature_count - 1]['start'], "numerator": dict_time_signature_aux[dict_time_signature_count - 1]['numerator'], "denominator": dict_time_signature_aux[dict_time_signature_count - 1]['denominator'], "end": time_print}
                    dict_time_signature_aux[dict_time_signature_count] = {"start": time_print, "numerator": msg.numerator, "denominator": msg.denominator}               
                    dict_time_signature_count = dict_time_signature_count + 1
                elif (msg.type == 'program_change'):
                    track_number = msg.program
                elif (msg.type == 'set_tempo'):
                    if (tempo != msg.tempo) and (tempo != 0):
                        tempo_changes = 1
                    tempo = msg.tempo
                    bpm = round(500000*120/msg.tempo,0) 
    avg_notes_quantified = count_notes_quantified / count_notes    
    tema_df = pd.DataFrame.from_dict(dict_notes, "index")
    max_note = tema_df.start.max() + ticks_quantify
    dict_time_signature[dict_time_signature_count] = {"start": dict_time_signature_aux[dict_time_signature_count - 1]['start'], "numerator": dict_time_signature_aux[dict_time_signature_count - 1]['numerator'], "denominator": dict_time_signature_aux[dict_time_signature_count - 1]['denominator'], "end": max_note}
    tema_df_notes_end = pd.DataFrame.from_dict(dict_notes_end, "index")
    df_time_signature = pd.DataFrame.from_dict(dict_time_signature, "index")       
    df_time_quantify = pd.DataFrame(range(0,int(max_note),int(ticks_quantify)), columns=['start'])
    ## Agrega time signature a tema_df
    for index, row in df_time_signature.iterrows():
        row_start = row.start
        row_end = row.end
        mask_signature_start = (df_time_quantify.start > row_start) 
        mask_signature_end = (df_time_quantify.start <= row_end)
        df_time_quantify.loc[mask_signature_start & mask_signature_end,'numerator'] = row.numerator
        df_time_quantify.loc[mask_signature_start & mask_signature_end,'denominator'] = row.denominator
    df_time_quantify.loc[:,'compas_val'] = ticks_quantify / (ticks * df_time_quantify.numerator)    
    df_time_quantify.loc[:,'compas_num'] = df_time_quantify.compas_val.cumsum()     
    tema_df = tema_df.join(df_time_quantify[['start','compas_num']].set_index('start'), on='start', how='left') 
    tema_df_merged = pd.merge(tema_df, tema_df_notes_end,on=['note_num','start','track_name'])
    controls = pd.Series(controls).head(40)
    controls = controls[controls > 10].sum() / n_tracks_used
    return tema_df_merged, controls, key_signatures, time_signatures, n_tracks_used, tempo, bpm, tempo_changes, avg_notes_quantified                  

def limit_outlyer_duration_notes(tema_df):
    notes_weight = pd.cut(tema_df.duration, 6)
    outlyeras_duration = pd.DataFrame(tema_df.duration.quantile([0.05,0.95]))
    mask_outlyers_lower = tema_df.duration < outlyeras_duration.duration[0.05]
    tema_df.loc[mask_outlyers_lower,'duration'] = outlyeras_duration.duration[0.05]
    mask_outlyers_higher = tema_df.duration > outlyeras_duration.duration[0.95]
    tema_df.loc[mask_outlyers_higher,'duration'] = outlyeras_duration.duration[0.95]
    notes_weight = pd.cut(tema_df.duration, 6)
    return tema_df

def get_theme_stats(file_path, file_name):
    ## Instancia el archivo midi
    mid = load_midi_file(file_path)
    ticks_per_beat = mid.ticks_per_beat
    duracion_tema = mid.length
    tema_df, controls, key_signatures, time_signatures, n_tracks_used, tempo, bpm, tempo_changes, avg_notes_quantified  = get_theme_df(mid)  
    ## Toma fraccion de golpe sin el compas
    tema_df.compas_num = tema_df.compas_num.fillna(method='ffill')
    tema_df.compas_num = tema_df.compas_num.fillna(0)
    tema_df.loc[:,'compas_fraction'] = tema_df.compas_num.apply(lambda x: round(x - int(x),3))
    tema_df.loc[tema_df.compas_fraction == 1,'compas_fraction'] = 0   
    ## Parsea a enteros los valores numéricos
    for col in tema_df.loc[:,tema_df.columns!="track_name"].columns:
        tema_df[col] = pd.to_numeric(tema_df[col])
    ## Calcula la duración de cada nota
    tema_df.loc[:,'duration'] = tema_df.end - tema_df.start
    ## Agregamos informacion de instrumentos y batería
    tema_df = pd.merge(tema_df, midi_instruments,how='left',left_on='track_name',right_on='num_mid').drop('num_mid',axis=1)
    tema_df.loc[tema_df.channel == 9,['intrument_subcat']] = tema_df[tema_df.channel == 9].note_num.apply(lambda x: 'dd_' + midi_drum_sounds_dict[x])
    tema_df.loc[tema_df.channel != 9,['intrument_subcat']] = "ii_" + tema_df.loc[tema_df.channel != 9,['intrument_subcat']]
    tema_df.loc[tema_df.channel == 9,['intrument_cat']] = 'Drums'
    ## Genera un frame agrupando golpes x el momento del compas
    cant_compases = math.trunc(tema_df.compas_num.max()) + 1
    df_compas = tema_df.groupby(['intrument_subcat']).compas_fraction.value_counts() / cant_compases
    df_compas = df_compas[df_compas > 0.1]
    ## Generamos datos estadisticos de la instrumentación
    instrumentos_por_seg = pd.Series(tema_df.intrument_subcat.value_counts() / duracion_tema)
    ## Eliminamos las notas de batería de nuestro analisis musical 
    tema_df = tema_df.loc[tema_df.intrument_cat != 'Drums']
    ## agrega el nombre y octava de notas a la tabla
    tema_df = pd.merge(tema_df, midi_notes,how='left',left_on='note_num',right_on='note_number').drop('note_number',axis=1)
    ##elimina notas demasiado cortas y demasiado largas que pueden afectar al análisis
    tema_df = limit_outlyer_duration_notes(tema_df)
    ## Categoriza duración
    tema_df.loc[:,'cat_duration'] = tema_df.duration / mid.ticks_per_beat
    ## Categoriza VELOCITY
    cat_velocity = pd.cut(tema_df.velocity, 6, labels=['pp','p','m','mf','f','ff'])
    tema_df.loc[:,'cat_velocity'] = cat_velocity
    ## Describe, muchos de estos valores van a ser utiles como predictores
    tema_describe = tema_df.describe()
    ## Reemplaza los tiempos de notas muy cercanas por notas simultaneas
    #tema_df = cuantize(tema_df)
    ## calcula la cantidad de ticks x segundo
    ticks_por_seg = tema_df.end.max() / duracion_tema
    ## Calcula la cantidad de notas que existen en simultaneo
    tema_df_simultaneous =  tema_df.start.value_counts()
    tema_df_simultaneous_times = tema_df_simultaneous.loc[tema_df_simultaneous > 1].index.to_list()
    tema_df.loc[:,'note_simultaneous'] = tema_df.start.apply(lambda x: 1 if x in tema_df_simultaneous_times else 0) 
    ## Convierte unidad de medida de timpo Ticks a segundos en cada nota
    tema_df.loc[:,'segundo'] = tema_df.start / ticks_por_seg
    ## Shape final del dataset
    notas_totales = tema_df.shape[0]
    ## indice de actividad (cantidad de notas) por tiempo
    cant_eventos_individuales = (notas_totales - len(tema_df_simultaneous_times) / 2)
    cant_eventos_piano =  tema_df[tema_df.intrument_cat == "Piano"].shape[0]
    actividad_por_tiempo = cant_eventos_individuales / duracion_tema
    velocity_avg = tema_df.cat_velocity.value_counts(normalize=True)
    length_notes_avg = tema_df.cat_duration.value_counts(normalize=True)
    ## Analiza proporciones de notas y duraciones mas repetidas
    notes_weight = round(tema_df.note.value_counts(normalize=True) * 100,2)
    notes_weight = round(tema_df.note_name.value_counts(normalize=True) * 100,2)
    all_values_notes = pd.DataFrame(notes_weight).reset_index()
    most_probable_scale = all_values_notes.head(7)
    scale_coverage = notes_weight.head(7).sum()
    avr_vertical_notes = tema_df.note_simultaneous.sum() / notas_totales
    cant_pedal_sustain = controls
    cant_eventos_por_pedal = cant_eventos_piano / cant_pedal_sustain if cant_pedal_sustain > 5 else np.NaN
    cant_eventos_por_pedal = cant_eventos_por_pedal if cant_eventos_por_pedal < 9999 else np.NaN
    cant_pedales_seg = cant_pedal_sustain / duracion_tema if cant_pedal_sustain > 5 else np.NaN
    #obtiene informacion de la escala
    nombre_escala = pd.merge(most_probable_scale, midi_scales_chords, how='left', left_on='index', right_on='note_name')
    nombre_escala.fillna(0,inplace=True)
    nombre_escala_T = nombre_escala.T
    nombre_escala_T
    mask = nombre_escala_T.apply(lambda x: True if all(x != 0) else False, axis=1)
    mask
    tabla_esacla = nombre_escala_T[mask].T
    tabla_esacla
    nombre_columna_Tmaj = tabla_esacla.columns[3]
    tonalidad  = 0
    tonalidad_escala = 'M'
    if nombre_columna_Tmaj != "U":
        tabla_esacla.set_index(tabla_esacla.columns[3],inplace=True,drop=False)
        mayor_chord_coverage = tabla_esacla.loc[[1,3,5],:'note_name_x'].sum()[1]
        minor_chord_coverage = tabla_esacla.loc[[6,1,3],:'note_name_x'].sum()[1]
        tonalidad = nombre_columna_Tmaj
    elif len(key_signatures) > 0:
        if 'b' in key_signatures[0]:
            dict_keys = {'Db':'C#', 'Eb':'D#', 'Gb':'F#', 'Ab':'G#', 'Bb':'A#','Dbm':'C#m', 'Ebm':'D#m', 'Gbm':'F#m', 'Abm':'G#m', 'Bbm':'A#m'}
            tonalidad =  dict_keys[key_signatures[0]]
        else:
            tonalidad = key_signatures[0]
        midi_scales_chords_weighted = pd.merge(midi_scales_chords[['note_name', tonalidad]], all_values_notes, how='left', left_on='note_name', right_on='index')
        midi_scales_chords_weighted.drop('index',axis=1,inplace=True)
        midi_scales_chords_weighted.columns = ['note_name', 'scale', 'weight']
        midi_scales_chords_weighted.set_index(midi_scales_chords_weighted.columns[1],inplace=True,drop=False)
        mayor_chord_coverage = midi_scales_chords_weighted.loc[[1,3,5],'weight'].sum()
        minor_chord_coverage = midi_scales_chords_weighted.loc[[6,1,3],'weight'].sum()        
        midi_scales_chords_weighted.sort_index(inplace=True)
        midi_scales_chords_weighted.drop('scale',axis=1,inplace=True)
        scale_coverage = midi_scales_chords_weighted.head(7).sum()[1]
        tabla_esacla = midi_scales_chords_weighted
        tabla_esacla
    else:
        tonalidad = 'U'
        tonalidad_escala = 'U'
        tabla_esacla = pd.DataFrame([ [ 0 for y in range( 2 ) ] for x in range( 7 ) ],columns=['1','2'])
        tabla_esacla.iloc[0:7,:] = 0
        mayor_chord_coverage = 0.5
        minor_chord_coverage = 0.5
    time_signatures_fix = time_signatures    
    if (len(time_signatures) == 0):
        time_signatures_fix = ""       
    midi_scale_full = midi_scales_full.set_index('note_name', inplace=True,drop=False)    
    clear_output(wait=True)
    print(file_name, tonalidad)
    print("sec:", mid.length)
    print("comp:",cant_compases)
    midi_scale_full = midi_scales_full.loc[:,[tonalidad]]
    midi_scale_full.columns = ['nota_relativa']
    tema_df = pd.merge(tema_df, midi_scale_full,on=['note_name']) 
    ## Calcula la cantidad de notas que existen en simultaneo por instrumento
    ## Calcula apariciones de acordes por compas
    dict_sim_notes_by_instrument_cat = {}
    list_chords = []
    for instrument_cat in midi_instruments.intrument_cat.unique():
        mask_cat_instrument = tema_df.intrument_cat == instrument_cat
        df_instrument = tema_df[mask_cat_instrument]
        if (df_instrument.shape[0] > 0):
            ## Calcula la cantidad de notas que existen en simultaneo
            tema_df_simultaneous =  df_instrument.start.value_counts()
            tema_df_simultaneous_times = tema_df_simultaneous.loc[tema_df_simultaneous > 1].index.to_list()
            notes_simul_by_instrument_mask = df_instrument.start.apply(lambda x: True if x in tema_df_simultaneous_times else False)
            avg_notes_simul_by_instrument = notes_simul_by_instrument_mask.sum() / df_instrument.shape[0]
            dict_sim_notes_by_instrument_cat['avg_simult_'+instrument_cat] = avg_notes_simul_by_instrument
            ## Calcula apariciones de acordes por compas
            chords_by_inst = df_instrument[notes_simul_by_instrument_mask].groupby(['start']).nota_relativa.unique()
            chords_by_inst_trnasform = chords_by_inst.reset_index().nota_relativa.apply(lambda x: '_'.join( np.sort(x) ) if len(x) > 2 else np.NaN)#.value_counts().dropna()
            list_chords.extend(chords_by_inst_trnasform.tolist())  
    chords_series = pd.Series(list_chords).dropna().value_counts()
    head_chord_series = chords_series.head(5) / cant_compases 
    cant_dist_chords = len(chords_series)        
    df_sim_notes_by_instrument = pd.DataFrame.from_dict(dict_sim_notes_by_instrument_cat, "index")
    ## crea el Music stats    
    music_stats = pd.DataFrame(columns=["time_signature_" + time_signatures[0], 'time_signature_cant', 'bpm', 'compases', 'cant_dist_chords',
                                        'avg_notes_quantified', 'tempo_changes','scale_coverage','mayor_chord_coverage','minor_chord_coverage',
                                          'scale_note_avg_1','scale_note_avg_2','scale_note_avg_3','scale_note_avg_4',
                                        'scale_note_avg_5','scale_note_avg_6','scale_note_avg_7', 'avr_simult_notes',
                                        'cant_eventos_por_pedal', 'cant_pedales_seg','duracion_seg','tracks_used','indice','Cluster'])
    music_stats.loc[0] = [1, len(time_signatures), bpm, cant_compases, cant_dist_chords, 
                          avg_notes_quantified, tempo_changes, scale_coverage,  mayor_chord_coverage, minor_chord_coverage, 
                           tabla_esacla.iloc[0,1], tabla_esacla.iloc[1,1], tabla_esacla.iloc[2,1], 
                           tabla_esacla.iloc[3,1], tabla_esacla.iloc[4,1], tabla_esacla.iloc[5,1], 
                           tabla_esacla.iloc[6,1] if tabla_esacla.shape[0] > 6 else 0, avr_vertical_notes,cant_eventos_por_pedal,
                           cant_pedales_seg,duracion_tema,n_tracks_used,file_path, 0]    
    tema_describe = tema_df.describe()
    data_describe = pd.DataFrame(tema_describe.loc[tema_describe.index != 'count',['note_num','Octave','duration']].unstack())
    data_describe.reset_index(inplace=True)
    data_describe.loc[:,'name'] = data_describe.level_0 + "_" + data_describe.level_1
    data_describe.set_index('name',inplace=True)
    data_describe.drop(['level_0','level_1'],axis=1, inplace=True)
    ## Agrego informacion de genero y grupo tomado de los archivos
    instrumentos_por_seg = instrumentos_por_seg.add_prefix('inst_')
    data_describe = data_describe[data_describe.columns[0]].add_prefix('describe_')
    length_notes_avg = length_notes_avg.add_prefix('length_note_')
    velocity_avg = velocity_avg.add_prefix('velocity_cat_')
    head_chord_series = head_chord_series.add_prefix('chord_')
    df_final = pd.concat([music_stats.T,df_sim_notes_by_instrument, instrumentos_por_seg, data_describe,velocity_avg,length_notes_avg, df_compas, head_chord_series]) 
    return df_final

def get_midi_from_path():
    path_midi = 'GET_FILE'
    # the dictionary to pass to pandas dataframe
    dict = {}
    count_files = 0
    for root, dirs, files in os.walk(path_midi, topdown=False):
        for name_file in files:
            dict[count_files] = {"file": os.path.join(root, name_file), "file_name": name_file}
            count_files = count_files + 1
    df_files_analize = pd.DataFrame.from_dict(dict, "index").iloc[0,:]
    display(df_files_analize)
    return get_theme_stats(df_files_analize.file, df_files_analize.file_name).T  

def cosine_similarity_row(X__sc, vec_a,indice):
    vec_b = X__sc
    cosine_list =cosine_similarity([vec_a], vec_b).reshape(-1)
    similarity = pd.Series(cosine_list,index=indice).sort_values(ascending=False)
    return similarity

def SaveAllFiles(dict, files_path):
    print('SAVING!!')
    data_midi = pd.DataFrame.from_dict(dict, "index")   
    keep_columns = (data_midi.isnull().sum() / data_midi.shape[0]) <= 0.98
    data_midi = data_midi.loc[:,keep_columns]
    data_midi.rename(lambda x: "".join(str(x)).replace('\'','') if isinstance(x, tuple) else x.replace('\'',''), axis='columns', inplace=True)
    data_midi.indice = data_midi.indice.apply(lambda x: x.replace(files_path + '\\', ""))

    data_midi.set_index(data_midi.indice,inplace=True,drop=False)
    data_index = data_midi[['indice', 'Cluster']]

    data_midi.fillna(0, inplace=True)
    X=data_midi.drop(['indice','Cluster'],axis=1)
    data_midi.index

    sc = StandardScaler()
    sc.fit(X)
    X__sc = sc.transform(X)
    del data_midi

    df_scaled = pd.DataFrame(X__sc,columns=X.columns)
    df_scaled.set_index(data_index['indice'],inplace=True)
    df_scaled['Cluster'] = data_index['Cluster']
    data_index['indice_lower_search'] = data_index['indice'].apply(lambda x: x.lower().replace('-',' ').replace('_',' '))
    data_index['indice_lower'] = data_index['indice'].apply(lambda x: x.lower())
    del data_index['Cluster']

    group_columns = {'ii_':'ritmica_instrument', 
                     'dd_': 'ritmica_drums', 
                     'inst_':'instrumentacion', 
                     'describe_note_num_':'amplitud_tonal',
                     'velocity_cat_':'dinamica',
                     'describe_duration_':'duracion_notas1',
                     'length_note_':'duracion_notas2',
                     'avg_simult_':'notas_simultaneas', 
                     'bpm':'tempo',
                     'duracion_seg':'duracion_tema', 
                     'scale_note_avg_':'armonia1',
                     '_coverage':'armonia2',
                     'scale_coverage':'armonia3', 
                     'chord_':'armonia4'}

    dict_columns_groups = {}
    for column_grup in group_columns.keys():
        arr_column_group = []  
        for column_name in df_scaled.columns:
            if (isinstance(column_name, tuple)):
                if (column_name[0].find(column_grup) != -1):
                    arr_column_group.append(column_name)
            elif (column_name.find(column_grup) != -1 and column_grup != 'ii_'  and column_grup != 'dd_'):
                arr_column_group.append(column_name)
        dict_columns_groups[group_columns[column_grup]] = arr_column_group

    colums_categorized = []
    for keys in dict_columns_groups:
        for values in dict_columns_groups[keys]:
            colums_categorized.append(values)

    columns_not_categorized = []
    for col in df_scaled.columns:
        if col not in colums_categorized:
            columns_not_categorized.append(col)

    columns_not_categorized.remove('Cluster')

    with open('dist/dict_columns_groups.pkl', 'wb') as wb_pkl:
        pickle.dump(dict_columns_groups, wb_pkl) 

    with open('dist/columns_not_categorized.pkl', 'wb') as wb_pkl:
        pickle.dump(columns_not_categorized, wb_pkl) 

    with open('dist/sc.pkl', 'wb') as wb_pkl:
        pickle.dump(sc, wb_pkl) 

    df_scaled.to_csv('df_scaled.csv')
    data_index.to_csv('data_index.csv')
    print('SAVED!!')