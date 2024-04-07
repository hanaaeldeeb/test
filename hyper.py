#%%
import sys
import os
from lightgbm import train
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages
import wandb
import argparse

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split,python_chrono_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from datetime import datetime
#%%
print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))
import wandb

# Inside your notebook cell

# sweep_config = {
#     'method': 'random'
#     }
# metric = {
#     'name': 'eval_ndcg',
#     'goal': 'maximize'   
#     }

# sweep_config['metric'] = metric
# parameters_dict = {
#     'decay': {
#         'values': ['0.01', '0.1']
#         },
#     'n_layers': {
#         'values': [1, 2]
#         },
#     'learning_rate': {
#           'values': [0.1, 0.01]
#         },
#     }

# sweep_config['parameters'] = parameters_dict
# import pprint

# pprint.pprint(sweep_config)

#%%
try:
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset_name', type=str, default='100k')
  parser.add_argument('--embed_size', type=int, default=16)
  parser.add_argument('--n_layers', type=int, default=1)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--learning_rate', type=float, default=0.001)
  parser.add_argument('--decay', type=float, default=0)
  parser.add_argument('--batch_size', type=int, default=1024)
  parser.add_argument('--eval_epoch', type=int, default=1)
  parser.add_argument('--top_k', type=int, default=20)
  args = parser.parse_args()

  en_train_log = True

  config = {'dataset_name': args.dataset_name, 
            'embed_size': args.embed_size, 
            'n_layers': args.n_layers, 
            'batch_size': args.batch_size, 
            'decay': args.decay, 
            'epochs': args.epochs, 
            'eval_epoch': args.eval_epoch, 
            'learning_rate': args.learning_rate, 
            'top_k': args.top_k,
            'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
  print('use cli')
except:
  en_train_log = False
  config = {'dataset_name': '1m', 
            'embed_size': 128, 
            'n_layers': 1, 
            'batch_size': 10000, 
            'decay': 0, #0.0001, 
            'epochs': 2, #20, 
            'eval_epoch': 1, 
            'learning_rate': 0.005, 
            'top_k': 20,
            'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
  print('use jupyter')

print(config)
yaml_file = "./lightgcn.yaml"

# Save initial result as 'failed' before training
config['result'] = 'successful'


# Read existing train log, if it exists


# Save the updated train log, including the initial configuration
# train_log.to_csv('./train_log.csv', index=False)
# prepare movielens dataset
df = movielens.load_pandas_df(size=config['dataset_name'], local_cache_path='./dataset/')
train,validate, test = python_chrono_split(df, ratio=[0.8,0.1,0.1], filter_by="user",col_user="userID", col_item="itemID", col_timestamp="timestamp")
data = ImplicitCF(train=train,test=validate,seed=DEFAULT_SEED)
# data_test = ImplicitCF(train=train,test=test)

hparams = prepare_hparams(yaml_file,
                          embed_size=config['embed_size'],
                          n_layers=config['n_layers'],
                          batch_size=config['batch_size'],
                          decay=config['decay'],
                          epochs=config['epochs'],
                          eval_epoch=1,
                          learning_rate=config['learning_rate'],
                          top_k=config['top_k'],
                         )


model = LightGCN(hparams, data,seed=DEFAULT_SEED)
#%%
# wandb.init(
#     # set the wandb project where this run will be logged
#             project="lgcn_rec",
#             # track hyperparameters and run metadata
#             config={
#             'dataset_name': "1m", 
#                     'embed_size':[ 1024,256], 
#                     'n_layers': [1,2,3,4], 
#                     'batch_size': [50000,10000], 
#                     'decay': [0.01,0.001], 
#                     'epochs': [40], 
#                     'eval_epoch': "1", 
#                     'learning_rate': [0.01,0.05], 
#                     'top_k': "20",
#                     'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#             }
#         )
with Timer() as train_time:
    model.fit()

print("Took {} seconds for training.".format(train_time.interval))


topk_scores = model.recommend_k_items(test, top_k=config['top_k'], remove_seen=True)
topk_scores.head()
eval_map = map_at_k(test, topk_scores, k=config['top_k'])
eval_ndcg = ndcg_at_k(test, topk_scores, k=config['top_k'])
eval_precision = precision_at_k(test, topk_scores, k=config['top_k'])
eval_recall = recall_at_k(test, topk_scores, k=config['top_k'])

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
config['eval_ndcg'] = eval_ndcg
config['eval_map'] = eval_map
config['eval_precision'] = eval_precision
config['eval_recall'] = eval_recall
# Read existing train log, if it exists
if os.path.exists('./train_log.csv'):
  train_log = pd.read_csv('./train_log.csv')
  # Append the current configuration to the log

  train_log = train_log.append(config, ignore_index=True)
else:
  # If no log exists, create a new DataFrame
  train_log = pd.DataFrame([config])





  # If no log exists, create a new DataFrame
  train_log = pd.DataFrame([config])
train_log.to_csv('./train_log.csv', index=False)

# Save the updated train log, including the initial configuration
# train_log.to_csv('./train_log.csv', index=False)
# if en_train_log:
#     train_log = pd.read_csv('./train_log.csv')
#     train_log.drop(train_log.tail(1).index,inplace=True)
#     train_log = train_log.append(config, ignore_index=True)
#     train_log.to_csv('./train_log.csv', index=False)
#%%
api = wandb.Api()
runs = api.runs("hanaaeldeeb/lightgcn_experiment")
artifact0 = runs[0].used_artifacts()
print(artifact0[0].name)

# %%
def get_wandb_artifacts(project_path, type=None, name=None, last_version=True):
    public_api = wandb.Api()
    if type is not None:
        types = [public_api.artifact_type(type, project_path)]
    else:
        types = public_api.artifact_types(project_path)

    res = L()
    for kind in types:
        for collection in kind.collections():
            if name is None or name == collection.name:
                versions = public_api.artifact_versions(
                    kind.type,
                    "/".join([kind.entity, kind.project, collection.name]),
                    per_page=1,
                )
                if last_version: res += next(versions)
                else: res += L(versions)
    return res
# %%
get_wandb_artifacts(project_path="https://wandb.ai/hanaaeldeeb/lightgcn_experiment")
# %%
