import os
import pandas as pd
import numpy as np
from datetime import datetime
from optparse import OptionParser

op = OptionParser()
op.add_option("-c", "--nbits", type=int, default=16, help="number of bits")
(opts, args) = op.parse_args()
nbits = opts.nbits
folder = "./ResultsTraning"
ext = '.csv'
directory = os.path.join(folder)

def load_data():
    data = pd.DataFrame()

    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(ext) and (str(nbits) +"BITS") in os.path.join(root, file):
                dataset = pd.read_csv(os.path.join(root, file), header=None)
                data=pd.concat([data, dataset])

    colnames=['dataset', 'algorithm', 'level', 'alpha', 'beta', 'lambda', 'p@100', 'r@100', 'p@1000', 'p@5000', 'map@100', 'map@1000', 'map@5000','added_val_flag','seed_used']
    data.columns = colnames
    return data

data = load_data()
cols = ['dataset', 'algorithm', 'level','alpha','beta' ,'lambda', 'p@100']

data = data[cols]
data.algorithm.unique()

data=data.sort_values(by=['algorithm', 'dataset', 'level'])
data_avg = data.groupby(['dataset', 'algorithm', 'level','alpha','beta' ,'lambda']).mean()

date = datetime.today().strftime('%d_%m_%y')
data_avg.to_csv('ResultsPostProcessing/table_' + str(nbits) + 'bits_' + date + ".csv", header =True)
