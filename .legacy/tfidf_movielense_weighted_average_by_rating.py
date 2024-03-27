#%%
from recommenders.datasets.python_splitters import python_chrono_split

from recommenders.models.tfidf.tfidf_utils import TfidfRecommender
from recommenders.datasets import movielens
import pandas as pd
from tqdm import tqdm
import torch

#%%
movies_df = pd.read_csv('/home/ee303/test/dataset/GENRES/movies.dat',
                        delimiter='::', engine= 'python', header=None,
                        names=['movie_name', 'genre'],encoding='latin1')
df = movielens.load_pandas_df(size="1m", local_cache_path='./dataset/')
movie_ids_to_extract=df["itemID"].unique()
movies_df = movies_df[movies_df['itemID'].isin(movie_ids_to_extract)]
#preprocess movie.csv file
movies_df.reset_index(inplace=True)
movies_df.rename(columns={"index": "itemID"}, inplace=True)

#%%
#applying tfidf to genres
recommender=TfidfRecommender(id_col="itemID",tokenization_method='bert')
clean_movies=recommender.clean_dataframe(movies_df,cols_to_clean=["genre"],new_col_name="genres").drop(columns=["genre"])
tf, vectors_tokenized = recommender.tokenize_text(df_clean=clean_movies, text_col="genres")
recommender.fit(tf, vectors_tokenized)

#%%

train, validate, test = python_chrono_split(df, ratio=[0.8,0.1,0.1], filter_by="user",col_user="userID", col_item="itemID", col_timestamp="timestamp")
userID_list = list(train['userID'].unique())

#creat rating matrix
r_matrix_train = train.pivot_table(index='userID', columns='itemID', values='rating')
r_matrix_validate = validate.pivot_table(index='userID', columns='itemID', values='rating')
r_matrix_test = test.pivot_table(index='userID', columns='itemID', values='rating')

#%%
def create_similarity_matrix(similarity_dict):
    # Find the total number of movies
    num_movies = max(similarity_dict.keys())

    # Initialize the similarity matrix with zeros
    similarity_matrix = [[0.0] * num_movies for _ in range(num_movies)]

    # Fill in the similarity values from the dictionary
    for movie_id, similarities in similarity_dict.items():
        for similarity in similarities:
            other_movie_id = similarity[1]
            similarity_value = similarity[0]
            similarity_matrix[movie_id - 1][other_movie_id - 1] = similarity_value

    return similarity_matrix

def dataframe_to_tensor(df, total_num_of_users, total_num_of_movies):
    # Initialize a tensor with zeros
    user_movie_tensor = torch.zeros(total_num_of_users, total_num_of_movies)

    # Iterate over DataFrame and fill in the tensor
    with tqdm(total=len(df)) as pbar:
        for index, row in df.iterrows():
            user_id = index
            for movie_id, rating in row.iteritems():
                user_movie_tensor[user_id - 1, movie_id - 1] = rating
            pbar.update()

    return user_movie_tensor

def recall_at_k(r_matrix_predict, r_matrix_validate_tensor, k):
    # Get the number of users
    num_users = r_matrix_predict.size(0)
    
    # Initialize recall sum
    recall_sum = 0.0
    
    for user_idx in range(num_users):
        # Sort predicted ratings for the user
        predicted_ratings = r_matrix_predict[user_idx]
        _, top_indices = torch.topk(predicted_ratings, k)
        
        # Get the set of movies in the top K for the user
        top_movies_predicted = set(top_indices.numpy())
        
        # Get the set of actual rated movies for the user in the validation set
        actual_movies_rated = set(torch.nonzero(r_matrix_validate_tensor[user_idx]).flatten().numpy())
        
        # Calculate the intersection of predicted and actual movies
        intersection = top_movies_predicted.intersection(actual_movies_rated)
        
        # Calculate Recall@K for this user
        recall_at_k_user = len(intersection) / len(actual_movies_rated) if len(actual_movies_rated) > 0 else 0.0
        
        # Add to recall sum
        recall_sum += recall_at_k_user
    
    # Calculate average recall across all users
    recall_at_k_avg = recall_sum / num_users
    
    return recall_at_k_avg

#%%
K = 20
recommender.recommend_top_k_items(clean_movies, k=K)
item_sim_mat = torch.tensor(create_similarity_matrix(recommender.recommendations), dtype=torch.float32)
print('Shape of item similarity matrix: ', item_sim_mat.shape)

#%%
import os
if os.path.exists('./.tensors/train') and os.path.exists('./.tensors/validation'):
    r_matrix_train_tensor = torch.load('./.tensors/train')
    r_matrix_validate_tensor = torch.load('./.tensors/validation')
else:
    r_matrix_train_tensor = dataframe_to_tensor(r_matrix_train, len(userID_list), len(item_sim_mat))
    r_matrix_validate_tensor = dataframe_to_tensor(r_matrix_validate, len(userID_list), len(item_sim_mat))
    torch.save(r_matrix_train_tensor, './.tensors/train')
    torch.save(r_matrix_validate_tensor, './.tensors/validation')
print('Shape of user-item matrix for train data: ', r_matrix_train_tensor.shape)
print('Shape of user-item matrix for validate data: ', r_matrix_validate_tensor.shape)

r_matrix_predict = torch.matmul(r_matrix_train_tensor, item_sim_mat)
print('Shape of user-item matrix for prediction: ', r_matrix_predict.shape)

#%%
k = 20  # Example value for k
recall_at_20 = recall_at_k(r_matrix_predict, r_matrix_train_tensor, k)
print("Recall@20:", recall_at_20)

#%%
create_similarity_matrix(recommender.recommendations)
# %%
redicted_ratings = r_matrix_predict[6]
_, top_indices = torch.topk(redicted_ratings, 20)
# %%
# Get the set of movies in the top K for the user
top_movies_predicted = set(top_indices.numpy())
top_movies_predicted
# %%
# Get the set of actual rated movies for the user in the validation set
actual_movies_rated = set(torch.nonzero(r_matrix_validate_tensor[6]).flatten().numpy())
actual_movies_rated
# %%
# Calculate the intersection of predicted and actual movies
intersection = top_movies_predicted.intersection(actual_movies_rated)
intersection
# %%
# Calculate Recall@K for this user
recall_at_k_user = len(intersection) / len(actual_movies_rated) if len(actual_movies_rated) > 0 else 0.0

recall_at_k_user
# %%

# %%
