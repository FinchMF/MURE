
import sys
import time

import math as mt 
import numpy as np
import pandas as pd 

from scipy.sparse.linalg import *
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix 


class Popularity_Recommender:

    def __init__(self):

        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None

    def generate_model(self, train_data, user_id, item_id):

        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {user_id: 'score'}, inplace=True)

        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[0,1])
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        self.popularity_recommendations = train_data_sort.head(10)

    def recommend(self, user_id):

        user_recommendations = self.popularity_recommendations
        user_recommendations['user_id'] = user_id

        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations

class Similarity_Recommender:

    def __init__(self):

        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.similarity_recommendations = None

    def generate_model(self, train_data, user_id, item_id):

        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    def get_user_items(self, user):

        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items

    def get_item_users(self, item):

        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users

    def get_all_items_train_data(self):

        return list(self.train_data[self.item_id].unique())

    
    def construct_cooccurance_matrix(self, user_songs, all_songs):

        user_songs_users = []

        x = 0
        for i in range(0, len(user_songs)):
            x += 1
            user_songs_users.append(self.get_item_users(user_songs[i]))
            z = ('[+] Constructing User Matrix' + '.' * x)
            sys.stdout.write('\r'+z)
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
        print('\n')
        print('[i] User Matrix Constructed..')

        u = 0
        for i in range(0, len(all_songs)):

            song_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(song_data[self.user_id].unique())
            
            for j in range(0, len(user_songs)):
                
                users_j = user_songs_users[j]
                z = (f'[+] Song: {u+1} Cooccuring Matrix')
                users_intersection = users_i.intersection(users_j)

                if len(users_intersection) != 0:

                    users_union = users_i.union(users_j)

                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))

                else:
                    cooccurence_matrix[j,i] = 0
                sys.stdout.write('\r'+z)
                u += 1

        return cooccurence_matrix


    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print('\n')
        print(f'[i] Non zero values in cooccurence matrix: {np.count_nonzero(cooccurence_matrix)}')

        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        sort_idx = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)

        columns = ['user_id', 'songs', 'score', 'rank']
        df = pd.DataFrame(columns=columns)

        rank = 1

        for i in range(0, len(sort_idx)):

            if ~np.isnan(sort_idx[i][0]) and all_songs[sort_idx[i][1]] not in user_songs and rank <=10:
                df.loc[len(df)] = [user, all_songs[sort_idx[i][1]], sort_idx[i][0], rank]

                rank += 1 

        if df.shape[0] == 0:
            print('The Current user has no songs for training the similarity based recommendation model')

            return -1
        else:

            return df

    def recommend(self, user):

        user_songs = self.get_user_items(user)
        print(f'[i] Number of Unique songs for that user: {len(user_songs)}')

        all_songs = self.get_all_items_train_data()
        print(f'[i] Number of Unique songs in subset: {len(all_songs)}')
        
        cooccurence_matrix = self.construct_cooccurance_matrix(user_songs, all_songs)
        recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

        return recommendations

    def get_similar(self, item_list):

        user_songs = item_list
        all_songs = self.get_all_items_train_data()
        print(f'[i] Number of Unique songs in the training set {len(all_songs)}')

        cooccurence_matrix = self.construct_cooccurance_matrix(user_songs, all_songs)

        user=''

        recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

        return recommendations

class Matrix_Factorization_Recommender:

    def __init__(self, urm, K=50):

        self.urm = urm
        self.K = K

    def compute_svd(self):

        U, s, Vt = svds(self.urm, self.K)

        mat_dim = (len(s), len(s))
        S = np.zeros(mat_dim, dtype=np.float32)

        for i in range(0, len(s)):
            S[i,i] = mt.sqrt(s[i])

        U = csc_matrix(U, dtype=np.float32)
        S = csc_matrix(S, dtype=np.float32)
        Vt = csc_matrix(Vt, dtype=np.float32)

        return U, S, Vt

    def compute_estimated_matrix(self, U, S, Vt, uTest, test):

        right_term = S*Vt
        max_rec = 250

        estimated_ratings = np.zeros(shape=(self.urm.shape[0], self.urm.shape[1]), dtype=np.float16)
        recommend_ratings = np.zeros(shape=(self.urm.shape[0], max_rec), dtype=np.float16)

        for user_test in uTest:
            
            prod = U[user_test, :]*right_term
            estimated_ratings[user_test, :] = prod.todense()
            recommend_ratings[user_test, :] = (-estimated_ratings[user_test, :]).argsort()[:max_rec]

        return recommend_ratings

    
        


    

