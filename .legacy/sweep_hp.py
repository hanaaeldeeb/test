#%%
import sys
import os
from lightgbm import train
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Only show error messages
import wandb
import argparse

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split, python_chrono_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from datetime import datetime

# Define sweep configuration for W&B Sweeps
sweep_config = {
    'method': 'random',  # Choose an appropriate search method based on complexity
    'metric': {
        'name': 'ndcg',  # Metric to optimize
        'goal': 'maximize'
    },
    'parameters': {
        'decay': {
            'values': ['0.01', '0.1']
        },
        'n_layers': {
            'values': [1, 2]
        },
        'learning_rate': {
            'values': [0.1, 0.01]
        }
    }
}
#%%
sweep_id = wandb.sweep(sweep_config)  # Initialize the sweep
wandb.agent(sweep_id, function=train)  # Start the sweep agent
# %%
def train():
    """
    Training function with W&B integration for hyperparameter sweeping.
    """

    wandb.init()  # Initialize W&B within the training function
    config = wandb.config  # Access hyperparameters from sweep configuration

    # Parse command-line arguments (optional)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, default='100k')
    parser.add_argument('--embed_size', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=config['n_layers'])  # Use sweep value
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=config['learning_rate'])  # Use sweep value
    parser.add_argument('--decay', type=float, default=config['decay'])  # Use sweep value
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_epoch', type=int, default=-1)
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()
  
    # Create training configuration
    config = {
        'dataset_name': args.dataset_name,
        'embed_size': args.embed_size,
        'n_layers': args.n_layers,
        'batch_size': args.batch_size,
        'decay': args.decay,
        'epochs': args.epochs,
        'eval_epoch': args.eval_epoch,
        'learning_rate': args.learning_rate,
        'top_k': args.top_k,
        'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }
    print('Using hyperparameters from sweep:', config)

    # Load data
    df = movielens.load_pandas_df(size=config['dataset_name'], local_cache_path='./dataset/')
    train, validate, test = python_chrono_split(df, ratio=[0.8, 0.1, 0.1], filter_by="user", col_user="userID",
                                                col_item="itemID", col_timestamp="timestamp")
    data = ImplicitCF(train=train, test=validate, seed=DEFAULT_SEED)

    # Prepare hyperparameters
    hparams = prepare_hparams(yaml_file="./lightgcn.yaml",
                              embed_size=config['embed_size'],
                              n_layers=config['n_layers'],
                              batch_size=config['batch_size'],
                              ecay=config['decay'],
                              epochs=config['epochs'],
                              eval_epoch=1,
                              learning_rate=config['learning_rate'],
                              top_k=config['top_k'],
                              )



    model = LightGCN(hparams, data,seed=DEFAULT_SEED)

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




      train_log = pd.DataFrame([config])
    train_log.to_csv('./train_log.csv', index=False)
    wandb.log({"MAP": eval_map, "NDCG": eval_ndcg, "Precision@K": eval_precision, "Recall@K": eval_recall})  # Log metrics to W&B
    return {"ndcg": eval_ndcg}  # Return the metric to maximize



# %%
