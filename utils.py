from keras.layers import *
from keras.models import Sequential
import numpy as np
import pandas as pd
from scipy import io as sio
import gc, sys,os

def evaluate_hashing_DE(labels,train_hash,test_hash,labels_trainn,labels_testt,tipo="topK",eval_tipo='PRatk',K=100):
    """
        Evaluate Hashing correclty: Query and retrieve on a different set
    """
    test_similares_train =  get_similar(test_hash,train_hash,tipo=tipo,K=K)
    if eval_tipo=="MAP":
        return MAP_atk(test_similares_train,labels_query=labels_testt, labels_source=labels_trainn, K=K) 
    elif eval_tipo == "PRatk":
        return measure_metrics(labels,test_similares_train,labels_testt,labels_source=labels_trainn)
    elif eval_tipo == "Patk":
        return M_P_atk(test_similares_train, labels_query=labels_testt, labels_source=labels_trainn, K=K)

def hash_data(model, x_train, x_test, binary=True):
    encode_train = model.predict(x_train)
    encode_test = model.predict(x_test)
    
    train_hash = calculate_hash(encode_train, from_probas=binary )
    test_hash = calculate_hash(encode_test, from_probas = binary)
    return train_hash, test_hash
    
def compare_hist_train(hist1,hist2, dataset_name="", global_L = True):
    ### binary vs traditional
    plt.figure(figsize=(15,6))
    if global_L:
        history_dict1 = hist1.history
        history_dict2 = hist2.history
        loss_values1 = history_dict1['loss']
        val_loss_values1 = history_dict1['val_loss']
        loss_values2 = history_dict2['loss']
        val_loss_values2 = history_dict2['val_loss']
        epochs_l = range(1, len(loss_values1) + 1)

        plt.figure(figsize=(15,6))
        plt.plot(epochs_l, loss_values1, 'bo-', label = "Train set traditional")
        plt.plot(epochs_l, val_loss_values1, 'bv-', label = "Val set traditional")
        plt.plot(epochs_l, loss_values2, 'go-', label = "Train set binary")
        plt.plot(epochs_l, val_loss_values2, 'gv-', label = "Val set binary")
    else:
        add_hist_plot(hist1, c='b', model_n = "VAE")
        add_hist_plot(hist2, c='g', model_n = "B-VAE")
  
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right", fancybox= True)
    plt.title("VAE loss "+dataset_name)
    plt.show()
    
def add_hist_plot(hist, c='b', model_n = ""):
    history_dict = hist.history
    rec_loss_values = history_dict['REC_loss']
    kl_loss_values = history_dict['KL']
    rec_val_loss_values = history_dict['val_REC_loss']
    kl_val_loss_values = history_dict['val_KL']
    epochs_l = range(1, len(rec_loss_values) + 1)

    plt.plot(epochs_l, rec_loss_values, c+'o-', label = "Train REC loss (%s)"%model_n)
    plt.plot(epochs_l, kl_loss_values, c+'o-.', label = "Train KL loss (%s)"%model_n)

    plt.plot(epochs_l, rec_val_loss_values, c+'v-', label = "Val REC loss (%s)"%model_n)
    plt.plot(epochs_l, kl_val_loss_values, c+'v-.', label = "Val KL loss (%s)"%model_n)
    
    
def visualize_probas(logits, probas):
    sns.distplot(probas.flatten())
    plt.title("Bits probability distribution p(b|x)")
    plt.show()
    
    from base_networks import samp_gumb
    samp_probas = samp_gumb(logits)
    
    plt.hist(samp_probas.flatten())
    plt.title("Gumbel-Softmax sample \hat{b}")
    plt.show()
    
def visualize_mean(data):
    sns.distplot(data.flatten())
    plt.title("Continous Bits distribution (standar VAE)")
    plt.show()
    
    
def visualize_probas_byB(probas):
    bits_prob_mean = probas.mean(axis=0)  # mean(alpha(x))

    B = probas.shape[1]

    f, axx = plt.subplots( 1,2 , figsize=(9,5), sharey=True)
    axx[0].bar(np.arange(B), bits_prob_mean)
    axx[0].set_xlabel("Bit")
    axx[0].set_ylim(0,1)
    axx[0].axhline(0.5, 0,B, c='r')

    sns.distplot(bits_prob_mean, vertical=True)
    axx[1].axhline(0.5, 0,B, c='r')

    f.suptitle("Bit mean probability mean(p(b|x))")
    plt.show()

    
def Load_Dataset(filename):
    dataset = sio.loadmat(filename)
    x_train = dataset['train']
    x_test = dataset['test']
    x_cv = dataset['cv']
    y_train = dataset['gnd_train']
    y_test = dataset['gnd_test']
    y_cv = dataset['gnd_cv']
    
    data = {}
    data["n_trains"] = y_train.shape[0]
    data["n_tests"] = y_test.shape[0]
    data["n_cv"] = y_cv.shape[0]
    data["n_tags"] = y_train.shape[1]
    data["n_feas"] = x_train.shape[1]

    ## Convert sparse to dense matricesimport numpy as np
    train = x_train.toarray()
    nz_indices = np.where(np.sum(train, axis=1) > 0)[0]
    train = train[nz_indices, :]
    train_len = np.sum(train > 0, axis=1)

    test = x_test.toarray()
    test_len = np.sum(test > 0, axis=1)

    cv = x_cv.toarray()
    cv_len = np.sum(cv > 0, axis=1)

    gnd_train = y_train[nz_indices, :]
    gnd_test = y_test
    gnd_cv = y_cv

    data["train"] = train
    data["test"] = test
    data["cv"] = cv
    data["train_len"] = train_len
    data["test_len"] = test_len
    data["cv_len"] = cv_len
    data["gnd_train"] = gnd_train
    data["gnd_test"] = gnd_test
    data["gnd_cv"] = gnd_cv
    
    return data


def define_fit(multi_label,X,Y, epochs=20, dense_=True):
    #function to define and train model

    #define model
    model_FF = Sequential()
    model_FF.add(InputLayer(input_shape=(X.shape[1],) ))
    if dense_:
        model_FF.add(Dense(256, activation="relu"))
    #model_FF.add(Dense(128, activation="relu"))
    if multi_label:
        model_FF.add(Dense(Y.shape[1], activation="sigmoid"))
        model_FF.compile(optimizer='adam', loss="binary_crossentropy")
    else:
        model_FF.add(Dense(Y.shape[1], activation="softmax"))
        model_FF.compile(optimizer='adam', loss="categorical_crossentropy",metrics=["accuracy"])
    model_FF.fit(X, Y, epochs=epochs, batch_size=128, verbose=0)
    return model_FF


class MedianHashing(object):
    def __init__(self):
        self.threshold = None
        self.latent_dim = None
    def fit(self, X):
        self.threshold = np.median(X, axis=0)
        self.latent_dim = X.shape[1]
    def transform(self, X):
        assert(X.shape[1] == self.latent_dim)
        binary_code = np.zeros(X.shape, dtype='int32')
        for i in range(self.latent_dim):
            binary_code[np.nonzero(X[:,i] < self.threshold[i]),i] = 0
            binary_code[np.nonzero(X[:,i] >= self.threshold[i]),i] = 1
        return binary_code
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
#if median is used, my binary codes should use it as well.. a probability 0.6 does not mean that
# the bit is always on..
#median= MedianHashing()
#median.fit(encode_train)
#val_train = median.transform(encode_train)
#val_hash = median.transform(encode_val)
def calculate_hash(data, from_probas=True, from_logits=True):    
    if from_probas: #from probas
        if from_logits:
            from scipy.special import expit
            data = expit(data)
        data_hash = (data > 0.5)*1
    else: #continuos
        data_hash = (np.sign(data) + 1)/2
    return data_hash.astype('int32')

def get_hammD(query, corpus):
    """
        Retrieve similar documents to the query document inside the corpus (source)
    """
    #codify binary codes to fastest data type
    query = query.astype('int8') #no voy a ocupar mas de 127 bits
    corpus = corpus.astype('int8')
    
    query_hammD = np.zeros((len(query),len(corpus)),dtype='int16') #distancia no sera mayor a 2^16
    for i,dato_hash in enumerate(query):
        query_hammD[i] = calculate_hamming_D(dato_hash, corpus) # # bits distintos)
    return query_hammD

def get_similar_hammD_based(query_hammD,tipo="topK", K=100, ball=0):
    """
        Retrieve similar documents to the query document inside the corpus (source)
    """
    query_similares = [] #indices
    for i in range(len(query_hammD)):        
        if tipo=="ball" or tipo=="EM":
            K = np.sum(query_hammD[i] <= ball) #find K over ball radius
            
        #get topK
        ordenados = np.argsort(query_hammD[i]) #indices
        query_similares.append(ordenados[:K].tolist()) #get top-K
    return query_similares


def xor(a,b):
    return (a|b) & ~(a&b)
def calculate_hamming_D(a,B):
    #return np.sum(a.astype('bool')^ B.astype('bool') ,axis=1) #distancia de hamming (# bits distintos)
    #return np.sum(np.logical_xor(a,B) ,axis=1) #distancia de hamming (# bits distintos)
    v = np.sum(a != B,axis=1) #distancia de hamming (# bits distintos) -- fastest
    #return np.sum(xor(a,B) ,axis=1) #distancia de hamming (# bits distintos)
    return v.astype(a.dtype)

def get_similar(query, corpus,tipo="topK", K=100, ball=2):
    """
        Retrieve similar documents to the query document inside the corpus (source)
    """
    #codify binary codes to fastest data type
    query = query.astype('int8') #no voy a ocupar mas de 127 bits
    corpus = corpus.astype('int8')
    
    query_similares = [] #indices
    for dato_hash in query:
        hamming_distance = calculate_hamming_D(dato_hash, corpus) # # bits distintos)
        if tipo=="EM": #match exacto
            ball= 0
        
        if tipo=="ball" or tipo=="EM":
            K = np.sum(hamming_distance<=ball) #find K over ball radius
            
        #get topK
        ordenados = np.argsort(hamming_distance) #indices
        query_similares.append(ordenados[:K].tolist()) #get top-K
    return query_similares

def measure_metrics(labels_name, data_retrieved_query, labels_query, labels_source):
    """
        Measure precision at K and recall at K, where K is the len of the retrieval documents
    """
    if type(labels_source) == list:
        labels_source = np.asarray(labels_source)
        
    multi_label=False
    if type(labels_query[0]) == list or type(labels_query[0]) == np.ndarray: #multiple classes
        multi_label=True
    
    #relevant document for query data
    
    if multi_label:
        count_labels = {label: np.sum([label in aux for aux in labels_source]) for label in labels_name}
    else:
        count_labels = {label: np.sum([label == aux for aux in labels_source]) for label in labels_name}
    
    #count_labels = {label:np.sum([label in aux for aux in labels_source]) for label in labels_name} 
    
    precision = 0.
    recall =0.
    for similars, label in zip(data_retrieved_query, labels_query): #source de donde se extrajo info
        if len(similars) == 0: #no encontro similares:
            continue
        labels_retrieve = labels_source[similars] #labels of retrieved data
        
        if multi_label: #multiple classes
            tp = np.sum([len(set(label)& set(aux))>=1 for aux in labels_retrieve]) #al menos 1 clase en comun --quizas variar
            recall += tp/np.sum([count_labels[aux] for aux in label ]) #cuenta todos los label del dato
        else: #only one class
            tp = np.sum(labels_retrieve == label) #true positive
            recall += tp/count_labels[label]
        precision += tp/len(similars)
    
    return precision/len(labels_query), recall/len(labels_query)

def P_atk(labels_retrieved, label_query, K=1):
    """
        Measure precision at K
    """
    if len(labels_retrieved)>K:
        labels_retrieved = labels_retrieved[:K]

        
    if type(labels_retrieved[0]) == list or type(labels_retrieved[0]) == np.ndarray: #multiple classes
        tp = np.sum([len(set(label_query)& set(aux))>=1 for aux in labels_retrieved]) #al menos 1 clase en comun --quizas variar
    else: #only one class
        tp = np.sum(labels_retrieved == label_query) #true positive
    
    return tp/len(labels_retrieved) #or K

def M_P_atk(datas_similars, labels_query, labels_source, K=1):
    """
        Mean (overall the queries) precision at K
    """
    if type(labels_source) == list:
        labels_source = np.asarray(labels_source)
    return np.mean([P_atk(labels_source[datas_similars[i]],labels_query[i],K=K) if len(datas_similars[i]) != 0 else 0.
                    for i,_ in enumerate(datas_similars)])


def AP_atk(data_retrieved_query, label_query, labels_source, K=0):
    """
        Average precision at K, average all the list precision until K.
    """
    multi_label=False
    if type(label_query) == list or type(label_query) == np.ndarray: #multiple classes
        multi_label=True
        
    if type(labels_source) == list:
        labels_source = np.asarray(labels_source)
        
    if K == 0:
        K = len(data_retrieved_query)
    
    K_effective = K
    if len(data_retrieved_query) < K:
        K_effective = len(data_retrieved_query)
    elif len(data_retrieved_query) > K:
        data_retrieved_query = data_retrieved_query[:K]
        K_effective = K
    
    labels_retrieve = labels_source[data_retrieved_query] 
    
    score = []
    num_hits = 0.
    for i in range(K_effective):
        relevant=False
        
        if multi_label:
            if len( set(label_query)& set(labels_retrieve[i]) )>=1: #at least one label in comoon at k
                relevant=True
        else:
            if label_query == labels_retrieve[i]: #only if "i"-element is relevant 
                relevant=True
        
        if relevant:
            num_hits +=1 
            score.append(num_hits/(i+1)) #precition at k 

    if len(score) ==0:
        return 0
    else:
        return np.mean(score) #average all the precisions until K

def MAP_atk(datas_similars, labels_query, labels_source, K=0):
    """
        Mean (overall the queries) average precision at K
    """
    return np.mean([AP_atk(datas_similars[i], labels_query[i], labels_source, K=K) if len(datas_similars[i]) != 0 else 0.
                    for i,_ in enumerate(datas_similars)]) 


##valores unicos de hash? distribucion de casillas
def hash_analysis(hash_data):
    hash_string = []
    for valor in hash_data:
        hash_string.append(str(valor)[1:-1].replace(' ',''))
    valores_unicos = set(hash_string)
    count_hash = {valor: hash_string.count(valor) for valor in valores_unicos}
    return valores_unicos, count_hash

def compare_cells_plot(nb,train_hash1,train_hash2,test_hash1=[],test_hash2=[]):
    print("Entrenamiento----")
    print("Cantidad de datos a llenar la tabla hash: ",train_hash1.shape[0])

    valores_unicos, count_hash =  hash_analysis(train_hash1)
    print("Cantidad de memorias ocupadas hash1: ",len(valores_unicos))
    plt.figure(figsize=(14,4))
    plt.plot(sorted(list(count_hash.values()))[::-1],'go-',label="Binary")
    
    valores_unicos, count_hash =  hash_analysis(train_hash2)
    print("Cantidad de memorias ocupadas hash2: ",len(valores_unicos))
    plt.plot(sorted(list(count_hash.values()))[::-1],'bo-',label="Traditional")
    plt.legend()
    plt.show()
    
    if len(test_hash1) != 0:
        print("Pruebas-----")
        print("Cantidad de datos a llenar la tabla hash: ",test_hash1.shape[0])
        
        valores_unicos, count_hash =  hash_analysis(test_hash1)
        print("Cantidad de memorias ocupadas hash1: ",len(valores_unicos))
        plt.figure(figsize=(15,4))
        plt.plot(sorted(list(count_hash.values()))[::-1],'go-',label="Binary")
        
        valores_unicos, count_hash =  hash_analysis(test_hash2)
        print("Cantidad de memorias ocupadas hash2: ",len(valores_unicos))
        plt.plot(sorted(list(count_hash.values()))[::-1],'bo-',label="Traditional")
        plt.legend()
        plt.show()
        

from PIL import Image
def check_availability(folder_imgs, imgs_files, labels_aux):
    imgs_folder = os.listdir(folder_imgs)

    mask_ = np.zeros((len(imgs_files)), dtype=bool) 
    for contador, (img_n, la) in enumerate(zip(imgs_files, labels_aux)):
        if contador%10000==0:
            gc.collect()
        
        if img_n in imgs_folder and len(la)!=0: #si imagen fue descargada y tiene labels.
            imagen = Image.open(folder_imgs+img_n)
            aux = np.asarray(imagen)
            if len(aux.shape) == 3 and aux.shape[2] == 3:#si tiene 3 canals
                mask_[contador] = True
            
            imagen.close()
    return mask_

def load_imgs_mask(imgs_files, mask_used, size, dtype = 'uint8'):
    N_used = np.sum(mask_used)
    X_t = np.zeros((N_used, size,size,3), dtype=dtype)
    real_i = 0
    for contador, foto_path in enumerate(imgs_files):
        if contador%10000==0:
            print("El contador de lectura va en: ",contador)
            gc.collect()

        if mask_used[contador]:
            #abrir imagen
            imagen = Image.open(foto_path)
            aux = imagen.resize((size,size),Image.ANTIALIAS)
            X_t[real_i] = np.asarray(aux, dtype=dtype)

            imagen.close()
            aux.close()
            del aux, imagen
            real_i +=1
    return X_t

def get_topK_labels(labels_set, labels, K=1):
    count_labels = {label:np.sum([label in aux for aux in labels_set]) for label in labels} 
    sorted_x = sorted(count_labels.items(), key=lambda kv: kv[1], reverse=True)
    print("category with most data (%s) has = %d, the top-K category (%s) has = %d"%(sorted_x[0][0],sorted_x[0][1],sorted_x[K-1][0], sorted_x[K-1][1]))
    return [value[0] for value in sorted_x[:K]]

def set_newlabel_list(new_labels, labels_set):
    return [[topic for topic in labels_list if topic in new_labels] for labels_list in labels_set]

def enmask_data(data, mask):
    if type(data) == list:
        return np.asarray(data)[mask].tolist()
    elif type(data) == np.ndarray:
        return data[mask]
    
def sample_test_mask(labels_list, N=100, multi_label=True):
    idx_class = {}
    for value in np.arange(len(labels_list)):
        if multi_label:
            for tag in labels_list[value]:
                if tag in idx_class:
                    idx_class[tag].append(value)
                else:
                    idx_class[tag] = [value]
        else:
            tag = labels_list[value]
            if tag in idx_class:
                idx_class[tag].append(value)
            else:
                idx_class[tag] = [value]

    mask_train = np.ones(len(labels_list), dtype='bool')
    selected = []
    for clase in idx_class.keys():
        selected_clase = []
        for dato in idx_class[clase]:
            if dato not in selected:
                selected_clase.append(dato) # si dato no ha sido seleccionado como rep de otra clase se guarda

        v = np.random.choice(selected_clase, size=N, replace=False)
        selected += list(v)
        mask_train[v] = False #test set
    return mask_train

import keras
from IPython.display import display

def evaluate_Top100(encoder,train,val,labels_train, labels_val, binary=True):
    encode_train = encoder.predict(train)
    encode_val = encoder.predict(val)
    
    train_hash = calculate_hash(encode_train, from_probas=binary )
    val_hash = calculate_hash(encode_val, from_probas=binary)

    val_similares_train =  get_similar(val_hash, train_hash, tipo='topK',K=100) 
    return M_P_atk(val_similares_train, labels_query=labels_val, labels_source=labels_train, K=100)

def find_beta(create_model, X_source_inp, X_source_out, X_query_input, labels_source,labels_query, binary=True,values=20,E=30,BS=100):
    decay = 2.
    beta_try = [ decay**(-value) for value in np.arange(values)] #u otros valores?

    P_k100 = []
    for beta_value in beta_try:
        
        p_value = []
        for _ in range(5): #maybe 3 
            vae_model , encoder_vae, _ = create_model(beta_value) #call function that creates model
            vae_model.fit(X_source_inp, X_source_out, epochs=E, batch_size=BS, verbose=0)

            #selected based on P@k=100
            p_value.append(evaluate_Top100(encoder_vae,X_source_inp,X_query_input,labels_source,labels_query,binary=binary))
            keras.backend.clear_session()
            
        P_k100.append(np.mean(p_value))        
        gc.collect()
    
    #Summary!
    df = pd.DataFrame({"beta":beta_try, "score":P_k100})
    df["score"] = df["score"].round(4)

    print("***************************************")
    print("*********** SUMMARY RESULTS ***********")
    print("***************************************")
    display(df)
    idx_max = np.argmax(P_k100)
    idx_min = np.argmin(P_k100)
    print("Best value is %.4f with beta %f"%(P_k100[idx_max], beta_try[idx_max]))
    print("Worst value is %.4f with beta %f"%(P_k100[idx_min], beta_try[idx_min]))
    print("***************************************")

    return beta_try[idx_max] #beta_selected

def find_lambda(create_model, X_source_inp, X_source_out, X_query_input, labels_source,labels_query, binary=True,values=14,E=30,BS=100):
    mitad = int(values/2)
    lambda_try = [ 10.**(value) for value in np.arange(-mitad,mitad)] #u otros valores?

    P_k100 = []
    for lambda_value in lambda_try:
        
        p_value = []
        for _ in range(5): #maybe 3 
            vae_model , encoder_vae, _ = create_model(lambda_value) #call function that creates model
            vae_model.fit(X_source_inp, X_source_out, epochs=E, batch_size=BS, verbose=0)

            #selected based on P@k=100
            p_value.append(evaluate_Top100(encoder_vae,X_source_inp,X_query_input,labels_source,labels_query,binary=binary))
            keras.backend.clear_session()
            
        P_k100.append(np.mean(p_value))        
        gc.collect()
    
    #Summary!
    df = pd.DataFrame({"lambda":lambda_try, "score":P_k100})
    df["score"] = df["score"].round(4)

    print("***************************************")
    print("*********** SUMMARY RESULTS ***********")
    print("***************************************")
    display(df)
    idx_max = np.argmax(P_k100)
    idx_min = np.argmin(P_k100)
    print("Best value is %.4f with lambda %f"%(P_k100[idx_max], lambda_try[idx_max]))
    print("Worst value is %.4f with lambda %f"%(P_k100[idx_min], lambda_try[idx_min]))
    print("***************************************")

    return lambda_try[idx_max] #lambda select