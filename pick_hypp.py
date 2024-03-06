#%%
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import argparse

#%%
if not os.path.exists('./train_log_clone4analyze.csv'):
    shutil.copyfile('./train_log.csv', './train_log_clone4analyze.csv')

train_log = pd.read_csv('./train_log_clone4analyze.csv')
# train_log = train_log[train_log['dataset_name'].str.contains('2018-12-31_18')]

train_log = train_log[(train_log['embed_size'] == 1024) &
                      (train_log['n_layers'] == 1) &
                      (train_log['batch_size'] == 8192) &
                      (train_log['decay'] == 0.0001) &
                      (train_log['epochs'] == 1) &
                      (train_log['learning_rate'] == 0.001)]

# train_log = train_log[train_log['create_time'].str.contains('2022-04-22') | train_log['create_time'].str.contains('2022-04-23')]

# %%
# hyperparameter picking
x_column = 'dataset_name'
y_column = 'score'
hypp_column = ['embed_size', 'n_layers', 'batch_size', 'decay', 'epochs', 'learning_rate']
used_column = hypp_column + [x_column, y_column]

train_log = train_log[used_column]
train_log.sort_values(by=x_column, ascending=False, inplace=True)

X = train_log[x_column].sort_values(ascending=True).unique() # epochs
hypps_list = train_log.groupby(hypp_column).mean().reset_index().sort_values('score', ascending=False)[hypp_column]
hypps_list = hypps_list[hypp_column].drop_duplicates().to_dict(orient='records')
Y_dict = {}

plt_topN = len(hypps_list) #int(4 + math.sqrt(len(hypps_list)))
plt_count = 0
plt.figure(figsize=(10, 6))
for hypps in hypps_list:
    Y = []
    for x in X:
        selected = pd.DataFrame(train_log)
        for column, value in hypps.items():
            selected = selected[selected[column] == value]
        selected = selected[selected[x_column] == x]
        y = selected[y_column].mean()
        Y.append(y)
    Y_dict[str(hypps)] = Y
    
    if plt_count < plt_topN:
        line_width = 3 - plt_count * (3 / plt_topN)
        plt.plot(X, Y, label=str(hypps), linewidth=line_width)
        plt_count += 1
plt.ylabel(y_column)
plt.xlabel(x_column)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

#%%