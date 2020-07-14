import numpy as np
import keras,gc,nltk
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing
from supervised_BAE import *
from utils import *

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 

import nltk
nltk.download('wordnet')

name_dat = "Snippets"

tokenizer = TfidfVectorizer().build_tokenizer()
stemmer = SnowballStemmer("english") 
lemmatizer = WordNetLemmatizer()

__random_state__ = 20
np.random.seed(__random_state__)


def read_file(archivo,symb=' '):
    with open(archivo,'r') as f:
        lineas = f.readlines()
        tokens_f = [linea.strip().split(symb) for linea in lineas]
        labels = [tokens[-1] for tokens in tokens_f]
        tokens = [' '.join(tokens[:-1]) for tokens in tokens_f]
    return labels,tokens

"""Extract features from raw input"""
def preProcess(s): #String processor
    return s.lower().strip().strip('-').strip('_')

def get_transform_representation(mode, analizer,min_count,max_feat):
    smooth_idf_b = False
    use_idf_b = False
    binary_b = False

    if mode == 'binary':
        binary_b = True
    elif mode == 'tf':     
        pass #default is tf
    elif mode == 'tf-idf':
        use_idf_b = True
        smooth_idf_b = True #inventa 1 conteo imaginario (como priors)--laplace smoothing
    return TfidfVectorizer(stop_words='english',tokenizer=analizer,min_df=min_count, max_df=0.8, max_features=max_feat
                                ,binary=binary_b, use_idf=use_idf_b, smooth_idf=smooth_idf_b,norm=None
                                  ,ngram_range=(1, 3)) 


def run_snippets(model_id,percentage_supervision,nbits_for_hashing,alpha_val,gamma_val,beta_VAL=0.015625):
 
    labels_t,texts_t = read_file("Data/data-web-snippets/train.txt")
    labels_test,texts_test = read_file("Data/data-web-snippets/test.txt")
    print("Datos de entrenamiento: ",len(texts_t))
    print("Datos de pruebas: ",len(texts_test))

    labels = list(set(labels_t))

    labels_t = np.asarray(labels_t)
    labels_test = np.asarray(labels_test)
    texts_train,texts_val,labels_train,labels_val  = train_test_split(texts_t,labels_t,random_state=20,test_size=0.1)

    print("Cantidad de datos Entrenamiento: ",len(texts_train))
    print("Cantidad de datos Validación: ",len(texts_val))
    print("Cantidad de datos Pruebas: ",len(texts_test))


    tokenizer = TfidfVectorizer().build_tokenizer()
    stemmer = SnowballStemmer("english") 
    lemmatizer = WordNetLemmatizer()

    def stemmed_words(doc):
        results = []
        for token in tokenizer(doc):
            pre_pro = preProcess(token)
            #token_pro = stemmer.stem(pre_pro) #aumenta x10 el tiempo de procesamiento
            token_pro = lemmatizer.lemmatize(pre_pro) #so can explain/interpretae -- aumenta x5 el tiempo de proce
            if len(token_pro) > 2 and not token_pro[0].isdigit(): #elimina palabra largo menor a 2
                results.append(token_pro)
        return results

    min_count = 1 #default = 1
    max_feat = 10000 #Best: 10000 -- Hinton (2000)
    vectorizer = get_transform_representation("tf",stemmed_words,min_count,max_feat)

    vectorizer.fit(texts_train)
    vectors_train = vectorizer.transform(texts_train)
    vectors_val = vectorizer.transform(texts_val)
    vectors_test = vectorizer.transform(texts_test)

    token2idx = vectorizer.vocabulary_
    idx2token = {idx:token for token,idx in token2idx.items()}


    #todense --get representation
    X_train = np.asarray(vectors_train.todense())
    X_val = np.asarray(vectors_val.todense())
    X_test = np.asarray(vectors_test.todense())

    del vectors_train,vectors_val,vectors_test#,vectors_train2,vectors_val2,vectors_test2
    gc.collect()

    ##representacion soft para TF ---mucho mejor!
    X_train_input = np.log(X_train+1) 
    X_val_input = np.log(X_val+1) 
    X_test_input = np.log(X_test+1) 

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
    gc.collect()

    print("DONE ...")

import sys

#### Parametrization <<<- IMPORTANT!

#### (1) VDSH: Unsupervised Loss +  alpha * Pointwise Loss 
#### (2) PHS: Unsupervised Loss +  alpha * Pointwise Loss  + gamma * Pairwise Loss
#### (3) SSBVAE: Unsupervised Loss +  alpha * Pointwise Loss  + alpha*gamma * SelfSup Loss

beta_VAL_Bernoulli = 0.015625 #Weight of KL in Unsupervised Loss is fixed as in previous works
beta_VAL_Gaussian = 0.125

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


