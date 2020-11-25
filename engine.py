
import random
import build


def popular_recommendations(n):

    dataset, train_data, test_data = build.retrieve_split_data()

    user_ids = build.get_user_ids(train_data)

    song_ids = build.get_song_ids(train_data)

    recommendations = build.popularity_recommendation(train_data, user_ids[n])

    return recommendations

def similarity_recommendations(n):

    dataset, train_data, test_data = build.retrieve_split_data()

    recommendations = build.similarity_recommendation(train_data, n)

    return recommendations

def matrix_factorized_recommendations():

    dataset, train_data, test_data = build.retrieve_split_data()

    data, sparse = build.prepare_data_for_matrix(dataset)

    users = build.get_user_ids(train_data)

    uTest = [7,10,34,20,8,854,100,24] #list(random.choices(users, k=10))
    
    recommendations = build.matrix_factorization_recommendation(sparse, uTest)

    build.print_matrix_recommendations(data, uTest, recommendations)

    return None

    




