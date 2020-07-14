import numpy as np
import keras,gc,nltk
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing
from supervised_BAE import *
from utils import *

###### SEMI-SUPERVISED TUNING ON 20 NEWS ###### 
###### ALL THE METHODS ON VALIDATION SET ###### 

from sklearn.datasets import fetch_20newsgroups
newsgroups_t = fetch_20newsgroups(subset='train')
labels = newsgroups_t.target_names

from utils import Load_Dataset

name_dat = "20News"

__random_state__ = 20
np.random.seed(__random_state__)


def run_20_news(model_id,percentage_supervision,nbits_for_hashing,alpha_val,gamma_val,beta_VAL = 0.015625):

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

	print(y_train_input.shape, y_val_input.shape, y_test_input.shape)


	print("\n=====> Creating and Training the Models (VDSH and BAE) ... \n")

	tf.keras.backend.clear_session()

	batch_size = 100

	if model_id == 1:

		vae,encoder,generator = VDSHS(X_train.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val)
		vae.fit(X_train_input, [X_train, y_train_input], epochs=30, batch_size=batch_size,verbose=1)
		name_model = 'VDSH_S'

	elif model_id == 2:

		vae,encoder,generator = PSH_GS(X_train.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,gamma=gamma_val)
		vae.fit(X_train_input, [X_train, y_train_input], epochs=30, batch_size=batch_size,verbose=1)
		name_model = 'PHS_GS'

	else:#elif model_id == 3:

		vae,encoder,generator = SSBVAE(X_train.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,gamma=gamma_val)
		vae.fit(X_train_input, [X_train, y_train_input],epochs=30, batch_size=batch_size,verbose=0)
		name_model = 'SSB_VAE'


	print("\n=====> Evaluate the Models ... \n")

	if model_id == 1:#Gaussian VAE

		train_hash, val_hash = hash_data(encoder,X_train_input,X_val_input,binary=False)

	else:

		train_hash, val_hash = hash_data(encoder,X_train_input,X_val_input)

	p_b,r_b = evaluate_hashing_DE(labels, train_hash, val_hash, labels_train ,labels_val, tipo="topK")
	p5k_b = evaluate_hashing_DE(labels, train_hash, val_hash, labels_train ,labels_val, eval_tipo="Patk",K=5000)

	if model_id == 1:
		name_file = "TUNING-VDSHS-%s-%.1f.csv"%(name_dat,percentage_supervision)
	elif model_id == 2:
		name_file = "TUNING-PHS-%s-%.1f.csv"%(name_dat,percentage_supervision)
	else:
		name_file = "TUNING-SSBVAE-%s-%.1f.csv"%(name_dat,percentage_supervision)        

	file = open(name_file,"a")
	file.write("%s, %s, %f, %f, %f, %f, %f, %f, %f\n"%(name_dat,name_model,percentage_supervision,alpha_val,beta_VAL,gamma_val,p_b,r_b,p5k_b))
	file.close()

	del vae, train_hash, val_hash
	del X_train, X_val    
	del encoder, generator
	del data    
	gc.collect()

	print("DONE ...")

		
import sys

#### Parametrization <<<- IMPORTANT!

#### (1) VDSH: Unsupervised Loss +  alpha * Pointwise Loss 
#### (2) PHS: Unsupervised Loss +  alpha * Pointwise Loss  + gamma * Pairwise Loss
#### (3) SSBVAE: Unsupervised Loss +  alpha * Pointwise Loss  + alpha*gamma * SelfSup Loss

beta_VAL_Bernoulli = 0.015625 #Weight of KL in Unsupervised Loss is fixed as in previous works
beta_VAL_Gaussian = 0.06250

model_type = int(sys.argv[1])#1,2,3
nbits = int(sys.argv[2])#16,32

ps_init = float(sys.argv[3])##Level of Supervision starts at ps_init
ps_end = float(sys.argv[4])##Level of Supervision ends at ps_end 

pow_init_alpha = int(sys.argv[5])##SEARCH FOR ALPHA STARTS AT 10**pow_init_alpha
pow_end_alpha  = int(sys.argv[6])##SEARCH FOR ALPHA ENDS AT 10**pow_end_alpha

pow_init_gamma = int(sys.argv[7])##SEARCH FOR GAMMA STARTS AT 10**pow_init_gamma
pow_end_gamma = int(sys.argv[8])##SEARCH FOR GAMMA ENDS AT 10**pow_end_gamma

decay = 10.0

alpha_values = [decay**(value) for value in np.arange(pow_init_alpha,pow_end_alpha+0.5,1)]
gamma_values = [decay**(value) for value in np.arange(pow_init_gamma,pow_end_gamma+0.5,1)]
ps_values = np.arange(ps_init,ps_end+0.08,0.1)

if model_type == 1:

	for ps_ in ps_values:

		n_classes, labels, labels_train, labels_val, X_train, X_val, X_train_input, X_val_input, y_train_input, y_val_input = load_CIFAR(ps_)

		for alpha_val_ in alpha_values:
			run_CIFAR(model_type, ps_, nbits, alpha_val_,0,beta_VAL_Gaussian, n_classes,labels, labels_train, labels_val, X_train, X_val, X_train_input, X_val_input, y_train_input, y_val_input)

		del n_classes, labels, labels_train, labels_val, X_train, X_val, X_train_input, X_val_input, y_train_input, y_val_input
		gc.collect()

else:

	for ps_ in ps_values:

		n_classes, labels, labels_train, labels_val, X_train, X_val, X_train_input, X_val_input, y_train_input, y_val_input = load_CIFAR(ps_)

		for gamma_val_ in gamma_values:
			for alpha_val_ in alpha_values:
				run_CIFAR(model_type, ps_, nbits, alpha_val_,gamma_val_,beta_VAL_Bernoulli, n_classes,labels, labels_train, labels_val, X_train, X_val, X_train_input, X_val_input, y_train_input, y_val_input)


