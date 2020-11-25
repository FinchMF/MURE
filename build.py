
import pandas as pd
from scipy.sparse import coo_matrix 

import data.load_data as data
from data.load_data import Data_Collect, data_split
import models.recommenders as rec 


def retrieve_split_data():

    dataset = Data_Collect().retrieve()
    dataset.head()
    train_data, test_data = data_split(dataset)

    return dataset, train_data, test_data

def get_user_ids(dataset):

    return list(dataset.user)

def get_song_ids(dataset):

    return list(dataset.song)

def popularity_recommendation(dataset, user):

    model = rec.Popularity_Recommender()
    model.generate_model(dataset, 'user', 'title')
    recommendations = model.recommend(user)

    return recommendations

def similarity_recommendation(dataset, user_idx):

    user_id = list(dataset.user)[user_idx]
    model = rec.Similarity_Recommender()
    model.generate_model(dataset, 'user', 'title')
    user_items = model.get_user_items(user_id)

    recommendations = model.recommend(user_id)

    return recommendations 

def merge_data_for_matrix_factorization(dataset):

    df = dataset[['user', 'listen_count']].groupby('user').sum().reset_index()
    df.rename(columns={'listen_count': 'total_listen_count'}, inplace=True)
    data_merged = pd.merge(dataset, df)

    data_merged['fractional_play_count'] = data_merged['listen_count'] / data_merged['total_listen_count']

    return data_merged

def generate_data_sparse(dataset):

    user_codes = dataset.user.drop_duplicates().reset_index()
    song_codes = dataset.song.drop_duplicates().reset_index()

    user_codes.rename(columns={'index': 'user_index'}, inplace=True)
    song_codes.rename(columns={'index': 'song_index'}, inplace=True)

    song_codes['so_idx_value'] = list(song_codes.index)
    user_codes['us_idx_value'] = list(user_codes.index)

    dataset = pd.merge(dataset, song_codes, how='left')
    dataset = pd.merge(dataset, user_codes, how='left')

    matrix_candidate = dataset[['us_idx_value', 'so_idx_value', 'fractional_play_count']]

    data_arr = matrix_candidate.fractional_play_count.values
    row_arr = matrix_candidate.us_idx_value.values
    col_arr = matrix_candidate.so_idx_value.values

    sparse = coo_matrix((data_arr, (row_arr, col_arr)), dtype=float)

    return dataset, sparse

def prepare_data_for_matrix(dataset):

    data_merged = merge_data_for_matrix_factorization(dataset)
    data, sparse = generate_data_sparse(data_merged)

    return data, sparse

def matrix_factorization_recommendation(sparse, uTest):

    model = rec.Matrix_Factorization_Recommender(urm=sparse)
    U, S, Vt = model.compute_svd()

    uTest_recommendations = model.compute_estimated_matrix(U, S, Vt, uTest, True)

    return uTest_recommendations

def print_matrix_recommendations(data, uTest, uTest_recommendations):

    for user in uTest:
        print(f'\n[i] Recommendation for user: {user}')
        rank = 1
        for i in uTest_recommendations[user, 0:10]:
            song_deets = data[data.so_idx_value == i].drop_duplicates('so_idx_value')[['title', 'artist_name']]
            print(f'[i] The number {rank} recommended song is {list(song_deets.title)[0]} BY {list(song_deets.artist_name)[0]}')
            rank += 1

    return None

    

    










