#%%
import sys
import os
from lightgbm import train
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

import argparse

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from datetime import datetime

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))

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
    parser.add_argument('--eval_epoch', type=int, default=-1)
    parser.add_argument('--top_k', type=int, default=5)
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
            'epochs': 20, #20, 
            'eval_epoch': 1, 
            'learning_rate': 0.005, 
            'top_k': 5,
            'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    print('use jupyter')

print(config)
yaml_file = "./lightgcn.yaml"

config['result'] = 'failed'
config['score'] = 0

if os.path.exists('./train_log.csv'):
    train_log = pd.read_csv('./train_log.csv')
    train_log = train_log.append(config, ignore_index=True)
else:
    train_log = pd.DataFrame([config])
train_log.to_csv('./train_log.csv', index=False)

#%%
# prepare movielens dataset
df = movielens.load_pandas_df(size=config['dataset_name'], local_cache_path='./dataset/')
train, test = python_stratified_split(df, ratio=0.75)
data = ImplicitCF(train=train, test=test)

#%%
hparams = prepare_hparams(yaml_file,
                          embed_size=config['embed_size'],
                          n_layers=config['n_layers'],
                          batch_size=config['batch_size'],
                          decay=config['decay'],
                          epochs=config['epochs'],
                          eval_epoch=config['eval_epoch'],
                          learning_rate=config['learning_rate'],
                          top_k=config['top_k'],
                         )

#%%
model = LightGCN(hparams, data)

#%%
with Timer() as train_time:
    model.fit()

print("Took {} seconds for training.".format(train_time.interval))

#%%
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

#%%
if en_train_log:
    train_log = pd.read_csv('./train_log.csv')
    train_log.drop(train_log.tail(1).index,inplace=True)
    train_log = train_log.append(config, ignore_index=True)
    train_log.to_csv('./train_log.csv', index=False)

#%%