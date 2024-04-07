#%%
# import
import numpy as np

#%%

# hyper parameters poolse
dataset_name_list = ["1m"]
embed_size_list =  [64,512, 1024]
n_layers_list =  [1, 2, 3,4]
batch_size_list = [64,1024]
epochs_list = [50]
learning_rate_list = [0.00001]
decay_list = [0]

search_method = 'grid'
n_trial = 1000 # number of trials for random search
script = ''

hypp_list = []
for dataset_name in dataset_name_list:
    for embed_size in embed_size_list:
        for n_layers in n_layers_list:
            for batch_size in batch_size_list:
                for epochs in epochs_list:
                    for learning_rate in learning_rate_list:
                        for decay in decay_list:
                            tmp_dict = {'dataset_name': dataset_name, 'embed_size': embed_size, 'n_layers': n_layers, 'batch_size': batch_size, 'epochs': epochs, 'learning_rate': learning_rate, 'decay': decay}
                            hypp_list.append(tmp_dict)

#%%
# random search
if search_method == 'random':
    np.random.shuffle(hypp_list)
    hypp_list = hypp_list[:n_trial]
    total_epochs = 0
    for i in range(325,len(hypp_list)+325):
        script += 'echo {}/{}\n'.format(i+1, len(hypp_list))
        script += 'python hyper.py {} --embed_size {} --n_layers {} --batch_size {} --epochs {} --learning_rate {} --decay {}\n'.format(hypp_list[i]['dataset_name'], hypp_list[i]['embed_size'], hypp_list[i]['n_layers'], hypp_list[i]['batch_size'], hypp_list[i]['epochs'], hypp_list[i]['learning_rate'], hypp_list[i]['decay'])
        total_epochs += hypp_list[i]['epochs']

#%%
# grid search
if search_method == 'grid':
    total_epochs = 0
    for i in range(len(hypp_list)):
        script += 'echo {}/{}\n'.format(i+1+324, len(hypp_list)+324)
        script += 'python hyper.py {} --embed_size {} --n_layers {} --batch_size {} --epochs {} --learning_rate {} --decay {}\n'.format(hypp_list[i]['dataset_name'], hypp_list[i]['embed_size'], hypp_list[i]['n_layers'], hypp_list[i]['batch_size'], hypp_list[i]['epochs'], hypp_list[i]['learning_rate'], hypp_list[i]['decay'])
        total_epochs += hypp_list[i]['epochs']
        
#%%
print('Total trials: {}'.format(len(hypp_list)))
print('Total epochs: {}'.format(total_epochs))
print('Total time: {}h'.format(round(total_epochs*0.2/60, 2)))

#%%
# print and save script
# print(script)

fp = open("./run.bat", "w")
fp.write(script)
fp.close()
# %%
