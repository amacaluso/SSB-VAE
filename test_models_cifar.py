import numpy as np
import keras,gc,nltk
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing
from supervised_BAE import *
from utils import *

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import preprocessing
from utils import sample_test_mask
from sklearn.preprocessing import StandardScaler
import time

name_dat = "CIFAR-10"

__random_state__ = 20
np.random.seed(__random_state__)

def load_data(percentage_supervision,addval=1,reseed=0,seed_to_reseed=20):
    
    (_, aux_t), (_, aux_test) = keras.datasets.cifar10.load_data()

    labels = ["airplane", "automobile","bird", "cat","deer","dog","frog","horse","ship","truck"]
    labels_t = np.asarray([labels[value[0]] for value in aux_t])
    labels_test = np.asarray([labels[value[0]] for value in aux_test])
    labels_t = np.concatenate((labels_t,labels_test),axis=0)

    X_t = np.load("Data/cifar10_VGG_avg.npy") #mejora
    X_t.shape

    mask_train = sample_test_mask(labels_t, N=100)

    X_test = X_t[~mask_train]
    X_t = X_t[mask_train]
    labels_test = enmask_data(labels_t, ~mask_train)
    labels_t = enmask_data(labels_t, mask_train)

    gc.collect()

    std = StandardScaler(with_mean=True, with_std=True)
    std.fit(X_t)

    X_t = std.transform(X_t)
    X_test = std.transform(X_test)

    X_train, X_val, labels_train, labels_val  = train_test_split(X_t, labels_t, random_state=20, test_size=len(X_test))

    del X_t, labels_t
    gc.collect()

    X_train_input = X_train
    X_val_input = X_val
    X_test_input = X_test

    X_total_input = np.concatenate((X_train_input,X_val_input),axis=0)
    X_total = np.concatenate((X_train,X_val),axis=0)
    labels_total = np.concatenate((labels_train,labels_val),axis=0)

    #print("\n=====> Encoding Labels ...\n")
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)

    n_classes = len(labels)

    y_train = label_encoder.transform(labels_train)
    y_val = label_encoder.transform(labels_val)
    y_test = label_encoder.transform(labels_test)

    y_train_input = to_categorical(y_train,num_classes=n_classes)
    y_val_input = to_categorical(y_val,num_classes=n_classes)
    y_test_input = to_categorical(y_test,num_classes=n_classes)

    ##RESEED?
    if reseed > 0:
        np.random.seed(seed_to_reseed)
    else:
        np.random.seed(__random_state__)

    idx_train = np.arange(0,len(y_train_input),1)
    np.random.shuffle(idx_train)
    np.random.shuffle(idx_train)
    n_sup = int(np.floor(percentage_supervision*len(idx_train)))
    idx_sup = idx_train[0:n_sup]
    idx_unsup = idx_train[n_sup:]

    if (len(idx_unsup) > 0):
        for idx in idx_unsup:
            y_train_input[idx,:] = np.zeros(n_classes)

    Y_total_input = y_train_input

    if addval > 0:

        idx_val = np.arange(0,len(y_val_input),1)
        np.random.shuffle(idx_val)
        np.random.shuffle(idx_val)
        n_sup_val = int(np.floor(percentage_supervision*len(idx_val)))
        idx_sup_val = idx_val[0:n_sup_val]
        idx_unsup_val = idx_val[n_sup_val:]

        if (len(idx_unsup_val) > 0):
            for idx in idx_unsup_val:
                y_val_input[idx,:] = np.zeros(n_classes)

        Y_total_input = np.concatenate((y_train_input,y_val_input),axis=0)

    return n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input


def run_CIFAR(model_id,percentage_supervision,nbits_for_hashing,alpha_val,lambda_val,beta_VAL,name_file, addval,reseed,seed_to_reseed, n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input):
 
    batch_size = 512
    tf.keras.backend.clear_session()

    tic = time.perf_counter()

    if model_id == 1:

        vae,encoder,generator = VDSHS(X_total.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val)
        vae.fit(X_total_input, [X_total, Y_total_input], epochs=10, batch_size=batch_size,verbose=1)
        name_model = 'VDSH_S'

    elif model_id == 2:

        #MODIFICA ESEGUITA
		#Vecchia versione: vae,encoder,generator = PSH_GS(X_total.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,gamma=gamma_val)
		#Sostituisci gamma con lambda_ , e gamma_val con lambda_val
        vae,encoder,generator = PSH_GS(X_total.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,lambda_=lambda_val)
        vae.fit(X_total_input, [X_total, Y_total_input], epochs=10, batch_size=batch_size,verbose=1)
        name_model = 'PHS_GS'

    else:#elif model_id == 3:

        #MODIFICA ESEGUITA
		#Vecchia versione: vae,encoder,generator = SSBVAE(X_total.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,gamma=gamma_val)
		#Sostituisci gamma con lambda_ , e gamma_val con lambda_val
        vae,encoder,generator = SSBVAE(X_total.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,lambda_=lambda_val)
        vae.fit(X_total_input, [X_total, Y_total_input], epochs=10, batch_size=batch_size,verbose=1)
        name_model = 'SSB_VAE'


    print("\n=====> Evaluate the Models ... \n")


    if model_id == 1:#Gaussian VAE

        total_hash, test_hash = hash_data(encoder,X_total_input,X_test_input, binary=False)

    else:#Bernoulli VAE

        total_hash, test_hash = hash_data(encoder,X_total_input,X_test_input)
    

    p100_b,r100_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK")
    p5000_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="Patk",K=5000)
    p1000_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="Patk",K=1000)
    map5000_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="MAP",K=5000)
    map1000_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="MAP",K=1000)
    map100_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="MAP",K=100)

    file = open(name_file,"a")

    file.write("%s, %s, %f, %f, %f, %f, %f, %f, %f,  %f, %f, %f, %f, %d, %d\n"%(name_dat,name_model,percentage_supervision,alpha_val,beta_VAL,lambda_val,p100_b,r100_b,p1000_b,p5000_b,map100_b,map1000_b,map5000_b,addval,seed_to_reseed))
    file.close()

    del vae, total_hash, test_hash
    gc.collect()

    print("DONE ...")

import sys
from optparse import OptionParser

def testcifar(model,ps, addvalidation, alpha, beta, lambda_, repetitions, nbits,  ofilename, reseed=0):
    seeds_to_reseed = [20, 144, 1028, 2044, 101, 6077, 621, 1981, 2806, 79]
    nbits = int(nbits)

    if reseed > 0:
        for rep in range(repetitions):
            new_seed = seeds_to_reseed[rep%len(seeds_to_reseed)]
            n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input = load_data(ps,addval=addvalidation,reseed=reseed,seed_to_reseed=new_seed)
            run_CIFAR(model,ps,nbits,alpha,lambda_,beta,ofilename,addvalidation,reseed,new_seed,n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input)
            del n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input
            gc.collect()
    else:
        n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input = load_data(ps,addval=addvalidation,reseed=0,seed_to_reseed=20)
        for rep in range(repetitions):
            run_CIFAR(model,ps,nbits,alpha,lambda_,beta,ofilename,addvalidation,0,20,n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input)

