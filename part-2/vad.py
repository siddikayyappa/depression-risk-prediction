import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
from tqdm import tqdm

song_features= pkl.load(open('vad_features_spotify.pkl', 'rb'))
print("Loaded song features")
dir_1 = "./../dataset/1year_top500tracks_with_tags/"
file_list = os.listdir(dir_1)


def non_weigthed_vad(user_file, song_features):
    print(user_file, end = ' ')
    file_1 = open(user_file)
    temp = file_1.read()
    temp = temp.split('\n')
    tuple_array = []
    play_count_array = []
    for i in (temp[1:]):
        x = i.split(',')
        if(len(x) < 3):
            continue
        play_count_idx = -1
        for j in range(len(x)):
            if(x[j].isdigit()):
                play_count = int(x[j])
                play_count_idx = j
                break
        words = x[:]
        song_tuple = (words[0], words[1])
        play_count_array.append(play_count)
        tuple_array.append(song_tuple)
    print("Tuple Appended", end = ' ')
    temp_array = [[0, 0]]
    for i in (range(len(tuple_array))):
        if(tuple_array[i] in song_features):
            temp_array.append(song_features[tuple_array[i]]*play_count_array[i])
    temp_array = temp_array[1:]
    print("Features Appended")
    print(temp_array)
    return (np.mean(temp_array, axis=0))

# vad = extract_vad(extract_playcount('user.csv'), song_features)
# non_weigthed_vad('user.csv', song_features)
all_user_va_songs = dict()
for i in tqdm(range(len(file_list))):
    print(i, end = ' ')
    all_user_va_songs[file_list[i]] = non_weigthed_vad(dir_1 + file_list[i], song_features)

print("Dumping all_user_va_songs")
pkl.dump(all_user_va_songs, open('all_user_va_songs.pkl', 'wb'))
print("Dumped all_user_va_songs")