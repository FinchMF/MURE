
import time
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split

import sqlite3 


dataset = {

    'text_data': 'data/song_data/train_triplets.txt',
    'meta_data': 'data/track_metadata.db'
}

def load():

    start = time.time()
    print('[+] Loading....')
    data = pd.read_csv(filepath_or_buffer=dataset['text_data'],
                          sep='\t',
                          header=None, 
                          names=['user', 'song', 'play_count'])
    end = time.time()
    print(f'[i] Triplet dataset loaded in {end - start} seconds')

    return data

def get_user_play_counts():

    start = time.time()
    output = {}
    with open(dataset['text_data'], 'r') as f:

        for line_idx, line in enumerate(f):

            user = line.split('\t')[0]
            play_count = int(line.split('\t')[2])

            if user in output:

                play_count += output[user]
                output.update({user:play_count})

            output.update({user:play_count})

    data = pd.DataFrame([{'user':k, 'play_count':v} for k,v in output.items()])
    data = data.sort_values(by='play_count', ascending=False)

    data.to_csv(path_or_buf='data/user_playcount.csv', index=False)
    end = time.time()

    print(f'[i] Time to load and save users and user play counts: {end - start} seconds')
    print(f'[i] Number of total users: {len(list(data.user))}')
    return data

def get_song_play_counts():

    start = time.time()
    output = {}
    with open(dataset['text_data'], 'r') as f:

        for line_idx, line in enumerate(f):

            song = line.split('\t')[1]
            play_count = int(line.split('\t')[2])

            if song in output:

                play_count += output[song]
                output.update({song:play_count})

            output.update({song:play_count})

    data = pd.DataFrame([{'song':k, 'play_count':v} for k,v in output.items()])
    data = data.sort_values(by='play_count', ascending=False)

    data.to_csv(path_or_buf='data/song_playcount.csv', index=False)
    end = time.time()

    print(f'[i] Time to load and save songs and song play counts: {end - start} seconds')
    print(f'[i] Number of total songs: {len(list(data.song))}')
    return data

def fetch_top_users(data):

    return data.head(10)

def fetch_top_songs(data):

    return data.head(10)

def trim_data(song_data, user_data):

    print('[+] Subset Song and User Data...')
    user_subset = list(user_data.user.head(n=100000))
    song_subset = list(song_data.song.head(n=30000))
    print('[i] Subset Complete...')
    return user_subset, song_subset

def subset_triplets(song_subset, user_subset):
    print('[+] Subset Triplet Data with Song and User Subsets...')
    triplet_dataset = load()
    triplet_subset = triplet_dataset[triplet_dataset.user.isin(user_subset)]
    triplet_subset = triplet_subset[triplet_subset.song.isin(song_subset)]
    print('[i] Subset Saving...')
    triplet_subset.to_csv(path_or_buf='data/triplet_subset.csv', index=False)
    print('[i] Triplet Data Subset Complete...')
    print(f'[i] Triplet Data Shape: {triplet_subset.shape}')

    return triplet_subset

def subset_meta(meta_data, song_subset):

    print('[+] Subsetting Meta Data with Song Subset...')
    meta_data = meta_data[meta_data.song_id.isin(song_subset)]

    return meta_data

def load_and_subset_meta(song_subset):

    print('[+] Fetching Track Meta Data...')
    conn = sqlite3.connect(dataset['meta_data'])
    curr = conn.cursor()
    curr.execute("SELECT name FROM sqlite_master WHERE type='table'")
    curr.fetchall()

    meta_data = pd.read_sql(con=conn, sql='SELECT * FROM songs') 
    meta_data = subset_meta(meta_data, song_subset)

    meta_data.to_csv(path_or_buf='data/track_meta_sub.csv', index=False)
    print('[i] Meta Data Subset Complete...')
    print(f'[i] Meta Data Subset Shape: {meta_data.shape}')

    return meta_data

def combine_and_clean(meta_data, triplet_subset):

    print('[+] Cleaning Meta Data...')
    del(meta_data['track_id'])
    del(meta_data['artist_mbid'])

    meta_data = meta_data.drop_duplicates(['song_id'])
    print('[+] Joining Datasets...')
    merged = pd.merge(triplet_subset, meta_data, 
                        how='left',
                        left_on='song',
                        right_on='song_id')
    merged.rename(columns={'play_count': 'listen_count'}, inplace=True)
    print('[+] Cleaning Joined Data Set...')
    del(merged['song_id'])
    del(merged['artist_id'])
    del(merged['duration'])
    del(merged['artist_familiarity'])
    del(merged['artist_hotttnesss'])
    del(merged['track_7digitalid'])
    del(merged['shs_perf'])
    del(merged['shs_work'])
    print('[+] Saving Joined Dataset...')
    merged.to_csv('data/merged_data.csv', index=False)

    return merged

def data_split(dataset):

    train_data, test_data = train_test_split(dataset, test_size=0.40, random_state=0)

    return train_data, test_data


class Data_Collect:

    def retrieve(self):
        # retrieve user and user total play counts 
        user_data = get_user_play_counts()
        # retrieve song and song total play counts
        song_data = get_song_play_counts()
        # subset user and song datasets in order to decrease computational load 
        user_subset, song_subset = trim_data(song_data, user_data)
        print(f'[i] Number of Subset Users: {len(user_subset)}')
        print(f'[i] Number of Subset Songs: {len(song_subset)}')
        # subset triplet data set in total with subsets of engineered extractions of user and song subsets
        triplet_subset = subset_triplets(song_subset, user_subset)
        # enhance by loading, subsetting and merging meta data with triplet subset
        meta_data = load_and_subset_meta(song_subset)
        merged = combine_and_clean(meta_data, triplet_subset)

        return merged



if __name__ == '__main__':

    Data_Collect().retrieve()





















