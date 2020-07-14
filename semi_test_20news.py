import numpy as np
import keras,gc,nltk
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing
from supervised_BAE import *
from utils import *
import gc
import time

###### SEMI-SUPERVISED TESTING ON 20 NEWS ###### 
###### ALL THE METHODS ON TEST SET ###### 

name_dat = "20News"

from sklearn.datasets import fetch_20newsgroups
newsgroups_t = fetch_20newsgroups(subset='train')
labels = newsgroups_t.target_names

from utils import Load_Dataset

__random_state__ = 20
np.random.seed(__random_state__)

def run_20_news(unsup_model,percentage_supervision,nbits_for_hashing,alpha_val,gamma_val,beta_VAL,name_file,addval=1,reseed=0,seed_to_reseed=20):

	filename = 'Data/ng20.tfidf.mat'
	data = Load_Dataset(filename)
	X_train_input = data["train"]
	X_train = X_train_input 
	X_val_input = data["cv"]
	X_val = X_val_input 
	X_test_input = data["test"]
	X_test = X_test_input
	labels_train = np.asarray([labels[value.argmax(axis=-1)] for value in data["gnd_train"]])
	labels_val = np.asarray([labels[value.argmax(axis=-1)] for value in data["gnd_cv"]])
	labels_test = np.asarray([labels[value.argmax(axis=-1)] for value in data["gnd_test"]])

	print("Cantidad de datos Entrenamiento: ",len(labels_train))
	print("Cantidad de datos Validación: ",len(labels_val))
	print("Cantidad de datos Pruebas: ",len(labels_test))

	#outputs as probabolities -- normalized over datasets..
	X_train = X_train/X_train.sum(axis=-1,keepdims=True) 
	X_val = X_val/X_val.sum(axis=-1,keepdims=True)
	X_test = X_test/X_test.sum(axis=-1,keepdims=True)
	print("Output target normalizado en dataset ",name_dat)

	X_train[np.isnan(X_train)] = 0
	X_val[np.isnan(X_val)] = 0
	X_test[np.isnan(X_test)] = 0

	X_total_input = np.concatenate((X_train_input,X_val_input),axis=0)
	X_total = np.concatenate((X_train,X_val),axis=0)
	labels_total = np.concatenate((labels_train,labels_val),axis=0)
	                        
	print("\n=====> Encoding Labels ...\n")


	label_encoder = preprocessing.LabelEncoder()
	label_encoder.fit(labels)

	n_classes = len(labels)

	y_train = label_encoder.transform(labels_train)
	y_val = label_encoder.transform(labels_val)
	y_test = label_encoder.transform(labels_test)

	y_train_input = to_categorical(y_train,num_classes=n_classes)
	y_val_input = to_categorical(y_val,num_classes=n_classes)
	y_test_input = to_categorical(y_test,num_classes=n_classes)

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

	print(y_train_input.shape, y_val_input.shape, y_test_input.shape)
	print("\n=====> Creating and Training the Models (VDSH and BAE) ... \n")

	batch_size = 100

	tic = time.perf_counter()

	if unsup_model == 1:#choose between 1,2,3,4,5

		binary_vae,encoder_Bvae,generator_Bvae = sBAE_Pointwise(X_train.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val)
		binary_vae.fit(X_total_input, [X_total, Y_total_input], epochs=30, batch_size=batch_size,verbose=1)
		name_model = 'sBAE_Pointwise'

	elif unsup_model == 2:

		binary_vae,encoder_Bvae,generator_Bvae = sBAE_Pairwise(X_train.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val)
		binary_vae.fit(X_total_input, [X_total, Y_total_input], epochs=30, batch_size=batch_size,verbose=1)
		name_model = 'sBAE_Pairwise'

	elif unsup_model == 3:

		binary_vae,encoder_Bvae,generator_Bvae = sBAE_Mixed(X_train.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,gamma=gamma_val)
		binary_vae.fit(X_total_input, [X_total, Y_total_input], epochs=30, batch_size=batch_size,verbose=1)
		name_model = 'sBAE_Mixed'

	elif unsup_model == 4:

		binary_vae,encoder_Bvae,generator_Bvae = sBAE_SelfTaught(X_train.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,gamma=gamma_val)
		binary_vae.fit(X_total_input, [X_total, Y_total_input], epochs=30, batch_size=batch_size,verbose=1)
		name_model = 'sBAE_SelfSUP_NAIVE'

	else: #unsup
		binary_vae,encoder_Bvae,generator_Bvae = binary_VAE(X_train.shape[1],Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL)
		binary_vae.fit(X_total_input, X_total, epochs=30, batch_size=batch_size,verbose=1)
		name_model = 'uBAE'


	toc = time.perf_counter()

	print("\n=====> Evaluate the Models ... \n")


	#total_hash_VAE, test_hash_VAE = hash_data(encoder_Tvae,X_total_input,X_test_input, binary=False)
	total_hash_BVAE, test_hash_BVAE = hash_data(encoder_Bvae,X_total_input,X_test_input)
	p_b,r_b = evaluate_hashing_DE(labels,total_hash_BVAE, test_hash_BVAE,labels_total,labels_test,tipo="topK")
	p5k_b = evaluate_hashing_DE(labels,total_hash_BVAE, test_hash_BVAE,labels_total,labels_test,tipo="topK",eval_tipo="Patk",K=5000)
	p1000_b = evaluate_hashing_DE(labels,total_hash_BVAE, test_hash_BVAE,labels_total,labels_test,tipo="topK",eval_tipo="Patk",K=1000)
	map5000_b = evaluate_hashing_DE(labels,total_hash_BVAE, test_hash_BVAE,labels_total,labels_test,tipo="topK",eval_tipo="MAP",K=5000)
	map1000_b = evaluate_hashing_DE(labels,total_hash_BVAE, test_hash_BVAE,labels_total,labels_test,tipo="topK",eval_tipo="MAP",K=1000)
	map100_b = evaluate_hashing_DE(labels,total_hash_BVAE, test_hash_BVAE,labels_total,labels_test,tipo="topK",eval_tipo="MAP",K=100)

	print("precision@100",p_b)
	print("recall@100",r_b)
	print("map@5000",p5k_b)

	file = open(name_file,"a")
	file.write("%s, %s, %f, %f, %f, %f, %f, %f, %f,  %f, %f, %f, %f, %d, %d\n"%(name_dat,name_model,percentage_supervision,alpha_val,beta_VAL,gamma_val,p_b,r_b,p1000_b,p5k_b,map100_b,map1000_b,map5000_b,addval,seed_to_reseed))
	file.close()

	name_time = "TIME-----"+name_file
	fileTIME = open(name_time,"a")
	elapsed_time = toc - tic
	fileTIME.write("%s, %s, %f, %f, %d, %d\n"%(name_dat,name_model,percentage_supervision,elapsed_time,addval,seed_to_reseed))
	fileTIME.close()


	del binary_vae, X_total_input, X_total
	del X_train, X_val, X_test
	del total_hash_BVAE, test_hash_BVAE
	del data
	gc.collect()

	print("DONE ...")

import sys
from optparse import OptionParser

op = OptionParser()
op.add_option("-M", "--model", type=int, default=4, help="model type (int[0,4])")
op.add_option("-p", "--ps", type=float, default=1.0, help="supervision level (float[0.1,1.0])")
op.add_option("-a", "--alpha", type=float, default=0.0, help="alpha value")
op.add_option("-b", "--beta", type=float, default=0.015625, help="beta value")
op.add_option("-g", "--gamma", type=float, default=0.0, help="gamma value")
op.add_option("-r", "--repetitions", type=int, default=1, help="repetitions") 
op.add_option("-o", "--ofilename", type="string", default=None, help="output filename") 
op.add_option("-s", "--reseed", type=int, default=0, help="re seed numpy for each rep?") 
op.add_option("-v", "--addvalidation", type=int, default=1, help="add the validation set to train?") 
op.add_option("-l", "--length_codes", type=int, default=32, help="number of bits") 

(opts, args) = op.parse_args()

ps = float(opts.ps)
nbits = int(opts.length_codes)

seeds_to_reseed = [20,144,1028,2044,101,6077,621,1981,2806,79]

if opts.ofilename == None:
	if opts.model == 1:
		opts.ofilename = 'LAZY_TEST-POINTWISE-20News_%dBITS.txt'%nbits
	elif opts.model == 2:
		opts.ofilename = 'LAZY_TEST-PAIRWISE-20News_%dBITS.txt'%nbits
	elif opts.model == 3:
		opts.ofilename = 'LAZY_TEST-MIXED-20News_%dBITS.txt'%nbits
	elif opts.model == 4:	
		opts.ofilename = 'LAZY_TEST-SELF-NAIVE-20News_%dBITS.txt'%nbits
	else:#0
		opts.ofilename = 'LAZY_TEST-UNSUP-20News_%dBITS.txt'%nbits

for rep in range(opts.repetitions):
	if opts.reseed > 0:
		new_seed = seeds_to_reseed[rep%len(seeds_to_reseed)]
		run_20_news(unsup_model=opts.model,percentage_supervision=ps,nbits_for_hashing=nbits,alpha_val=opts.alpha,gamma_val=opts.gamma,beta_VAL=opts.beta,name_file=opts.ofilename,addval=opts.addvalidation,reseed=opts.reseed,seed_to_reseed=new_seed)
	else:
		run_20_news(unsup_model=opts.model,percentage_supervision=ps,nbits_for_hashing=nbits,alpha_val=opts.alpha,gamma_val=opts.gamma,beta_VAL=opts.beta,name_file=opts.ofilename,addval=opts.addvalidation,reseed=opts.reseed,seed_to_reseed=20)

