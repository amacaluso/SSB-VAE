import os
import pandas as pd
import numpy as np

measure = 'p@100'
nbits = 32

def load_data(nbits=32, folder="./", ext='.csv'):
    data = pd.DataFrame()
    directory = os.path.join(folder)

    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(str(nbits)+'BITS'+ext):
                dataset = pd.read_csv(folder + file, header=None)
                data=pd.concat([data, dataset])

    colnames=['dataset', 'algorithm', 'level', 'alpha', 'beta', 'gamma', 'p@100', 'r@100', 'p@1000', 'p@5000', 'map@100', 'map@1000', 'map@5000','added_val_flag','seed_used']
    data.columns = colnames
    return data

data = load_data(nbits=nbits)
cols = ['dataset', 'algorithm', 'level', measure]

data = data[cols]
data.algorithm.unique()

data=data.sort_values(by=['algorithm', 'dataset', 'level'])
data_avg = data.groupby(['dataset', 'algorithm', 'level']).mean()
data_avg.unstack(level=[0 , 1 , 2])
results = data_avg.unstack(level=[2]).transpose()

results.to_csv('table_'+str(nbits)+'bits.csv')
print(np.round(results,3).to_latex())