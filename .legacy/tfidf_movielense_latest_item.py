#%%
from recommenders.datasets.python_splitters import python_chrono_split

from recommenders.models.tfidf.tfidf_utils import TfidfRecommender
from recommenders.datasets import movielens
import pandas as pd
from tqdm import tqdm

#%%
movies_df = pd.read_csv('/home/ee303/test/dataset/GENRES/movies.dat',
                        delimiter='::', engine= 'python', header=None,
                        names=['itemID','movie_name', 'genre'],encoding='latin1')
df = movielens.load_pandas_df(size="1m", local_cache_path='./dataset/')

# movies_df.reset_index(inplace=True)
# movies_df.rename(columns={"index": "itemID"}, inplace=True)
merged_df = pd.merge(movies_df, df[['itemID']], on='itemID', how='inner')
movies_df = merged_df.drop_duplicates(subset='itemID')
movies_df.reset_index(drop=True, inplace=True)

#%%
#applying tfidf to genres
recommender=TfidfRecommender(id_col="itemID",tokenization_method='none')
clean_movies=recommender.clean_dataframe(movies_df,cols_to_clean=["genre"],new_col_name="genres").drop(columns=["genre"])
tf, vectors_tokenized = recommender.tokenize_text(df_clean=clean_movies, text_col="genres",ngram_range=(1,4))
recommender.fit(tf, vectors_tokenized)

#%%

train, validate, test = python_chrono_split(df, ratio=[0.8,0.1,0.1], filter_by="user",col_user="userID", col_item="itemID", col_timestamp="timestamp")
userID_list = list(train['userID'].unique())

#creat rating matrix
r_matrix_train = train.pivot_table(index='userID', columns='itemID', values='rating')
r_matrix_validate = validate.pivot_table(index='userID', columns='itemID', values='rating')
r_matrix_test = test.pivot_table(index='userID', columns='itemID', values='rating')

#%%
#Generate recommendations based on the k most similar items to the last item rated by the user,according to timestamp,movies_df,movies info
def recommended_items(user_id, history, k):
    target_item=int(history[history["userID"] == user_id]["itemID"].iloc[-1])
    sim_mov=[t[1] for t in(recommender.recommendations[target_item])][:k]
    return sim_mov

def actual_items(user_id, actual_rating_matrix):   
    user_ratings = actual_rating_matrix.loc[user_id]
    movies_rated = user_ratings.dropna().index.tolist()
    return movies_rated

def recall_at_k(user_id, dataset, k, actual_rating_matrix):
    actual_items_list = actual_items(user_id, actual_rating_matrix)
    recommended_items_list = recommended_items(user_id, dataset, k=k)
    matched_item = set(actual_items_list).intersection(set(recommended_items_list))

    recall = len(matched_item) / len(actual_items_list) if len(actual_items_list) > 0 else 0 
    return recall

def average_recall_at_k(userID_list, dataset, k, evaluate_rating_matrix=r_matrix_validate):
    total_recall = 0
    num_users = len(userID_list)
    
    with tqdm(total=len(userID_list)) as pbar:
        for user_id in userID_list:
            recall_at_k_value = recall_at_k(user_id, dataset, k, evaluate_rating_matrix)       
            total_recall += recall_at_k_value
            pbar.update(1)
        
    average_recall = total_recall / num_users if num_users > 0 else 0
    return average_recall

#%%
recommender.recommend_top_k_items(clean_movies, 20)
average_recall_at_k(userID_list, validate, 20, r_matrix_validate)

#%%
average_recall_at_k(userID_list, test, 20, r_matrix_test)
