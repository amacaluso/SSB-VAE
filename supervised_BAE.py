import numpy as np
import keras
from keras.layers import *
from keras.models import Sequential,Model
from keras import backend as K
from base_networks import *
import tensorflow as tf

def my_KL_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return - K.sum(y_true*K.log(y_pred), axis=-1) 

def my_binary_KL_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    compl_y_pred = 1.0 - y_pred
    compl_y_pred = K.clip(compl_y_pred, K.epsilon(), 1)
    return - K.sum(y_true*K.log(y_pred) + (1-y_true)*K.log(compl_y_pred), axis=-1) 

def my_binary_KL_loss_stable(y_true, y_pred):

    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logits = K.log(y_pred) - K.log(1-y_pred) # sigmoid inverse
    neg_abs_logits = -K.abs(logits)
    relu_logits    = K.relu(logits)
    loss_vec = relu_logits - logits*y_true + K.log(1 + K.exp(neg_abs_logits))
    return K.sum(loss_vec)

def REC_loss(x_true, x_pred):
    x_pred = K.clip(x_pred, K.epsilon(), 1)
    return - K.sum(x_true*K.log(x_pred), axis=-1) #keras.losses.categorical_crossentropy(x_true, x_pred)

def traditional_VAE(data_dim,Nb,units,layers_e,layers_d,opt='adam',BN=True, summ=True, beta=0):
    pre_encoder = define_pre_encoder(data_dim, layers=layers_e,units=units,BN=BN)
    if summ:
        print("pre-encoder network:")
        pre_encoder.summary()
    generator = define_generator(Nb,data_dim,layers=layers_d,units=units,BN=BN)
    if summ:
        print("generator network:")
        generator.summary()

    ## Encoder
    x = Input(shape=(data_dim,))
    hidden = pre_encoder(x)
    z_mean = Dense(Nb,activation='linear', name='z-mean')(hidden)
    z_log_var = Dense(Nb,activation='linear',name = 'z-log_var')(hidden)
    encoder = Model(x, z_mean) # build a model to project inputs on the latent space

    def sampling(args):
        epsilon_std = 1.0
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Nb),mean=0., stddev=epsilon_std)
        return z_mean + K.exp(0.5*z_log_var) * epsilon #+sigma (desvest)
    
    ## Decoder
    z_sampled = Lambda(sampling, output_shape=(Nb,), name='sampled')([z_mean, z_log_var])
    output = generator(z_sampled)
        
    Recon_loss = REC_loss
    kl_loss = KL_loss(z_mean,z_log_var)
    def VAE_loss(y_true, y_pred): 
        return Recon_loss(y_true, y_pred) + beta*kl_loss(y_true, y_pred)

    traditional_vae = Model(x, output)
    traditional_vae.compile(optimizer=opt, loss=VAE_loss, metrics = [Recon_loss,kl_loss])
    
    return traditional_vae, encoder,generator

def sample_gumbel(shape,eps=K.epsilon()):
    """Inverse Sample function from Gumbel(0, 1)"""
    U = K.random_uniform(shape, 0, 1)
    return K.log(U + eps)- K.log(1-U + eps)



def VDSHS(data_dim,n_classes,Nb,units,layers_e,layers_d,opt='adam',BN=True, summ=True,tau_ann=False,beta=0,alpha=1.0,multilabel=False):

    pre_encoder = define_pre_encoder(data_dim, layers=layers_e,units=units,BN=BN)
    if summ:
        print("pre-encoder network:")
        pre_encoder.summary()
    generator = define_generator(Nb,data_dim,layers=layers_d,units=units,BN=BN)
    if summ:
        print("generator network:")
        generator.summary()

    ## Encoder
    x = Input(shape=(data_dim,))
    hidden = pre_encoder(x)
    z_mean = Dense(Nb,activation='linear', name='z-mean')(hidden)
    z_log_var = Dense(Nb,activation='linear',name = 'z-log_var')(hidden)
    encoder = Model(x, z_mean) # build a model to project inputs on the latent space

    def sampling(args):
        epsilon_std = 1.0
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Nb),mean=0., stddev=epsilon_std)
        return z_mean + K.exp(0.5*z_log_var) * epsilon #+sigma (desvest)
    
    ## Decoder
    z_sampled = Lambda(sampling, output_shape=(Nb,), name='sampled')([z_mean, z_log_var])
    
    output = generator(z_sampled)
        
    Recon_loss = REC_loss
    kl_loss = KL_loss(z_mean,z_log_var)
    def VAE_loss(y_true, y_pred): 
        return Recon_loss(y_true, y_pred) + beta*kl_loss(y_true, y_pred)

    if multilabel:
        supervised_layer = Dense(n_classes, activation='sigmoid',name='sup-class')(z_sampled)#req n_classes  
    else:
        supervised_layer = Dense(n_classes, activation='softmax',name='sup-class')(z_sampled)#req n_classes

    traditional_vae = Model(inputs=x, outputs=[output,supervised_layer])

    if multilabel:
        traditional_vae.compile(optimizer=opt, loss=[VAE_loss,my_binary_KL_loss],loss_weights=[1., alpha], metrics=[Recon_loss,kl_loss])
    else:
        traditional_vae.compile(optimizer=opt, loss=[VAE_loss,my_KL_loss],loss_weights=[1., alpha], metrics=[Recon_loss,kl_loss])

    return traditional_vae, encoder,generator


def binary_VAE(data_dim,Nb,units,layers_e,layers_d,opt='adam',BN=True, summ=True,tau_ann=False,beta=0):
    if tau_ann:
        tau = K.variable(1.0, name="temperature") 
    else:
        tau = K.variable(0.67, name="temperature") #o tau fijo en 0.67=2/3
    
    pre_encoder = define_pre_encoder(data_dim, layers=layers_e,units=units,BN=BN)
    if summ:
        print("pre-encoder network:")
        pre_encoder.summary()
    generator = define_generator(Nb,data_dim,layers=layers_d,units=units,BN=BN)
    if summ:
        print("generator network:")
        generator.summary()

    x = Input(shape=(data_dim,))
    hidden = pre_encoder(x)
    logits_b  = Dense(Nb, activation='linear', name='logits-b')(hidden) #log(B_j/1-B_j)
    #proba = np.exp(logits_b)/(1+np.exp(logits_b)) = sigmoidal(logits_b) <<<<<<<<<< recupera probabilidad
    #dist = Dense(Nb, activation='sigmoid')(hidden) #p(b) #otra forma de modelarlo
    encoder = Model(x, logits_b)

    def sampling(logits_b):
        #logits_b = K.log(aux/(1-aux) + K.epsilon() )
        b = logits_b + sample_gumbel(K.shape(logits_b)) # logits + gumbel noise
        return keras.activations.sigmoid( b/tau )

    b_sampled = Lambda(sampling, output_shape=(Nb,), name='sampled')(logits_b)
    output = generator(b_sampled)
        
    Recon_loss = REC_loss
    kl_loss = BKL_loss(logits_b)
    def BVAE_loss(y_true, y_pred): 
        return Recon_loss(y_true, y_pred) + beta*kl_loss(y_true, y_pred)

    binary_vae = Model(x, output)
    binary_vae.compile(optimizer=opt, loss=BVAE_loss, metrics = [Recon_loss,kl_loss])
    if tau_ann:
        return binary_vae, encoder,generator ,tau
    else:
        return binary_vae, encoder,generator


def PSH_GS(data_dim,n_classes,Nb,units,layers_e,layers_d,opt='adam',BN=True, summ=True,tau_ann=False,beta=0,alpha=1.0,gamma=1.0,multilabel=False):
    if tau_ann:
        tau = K.variable(1.0, name="temperature") 
    else:
        tau = K.variable(0.67, name="temperature") #o tau fijo en 0.67=2/3
    
    pre_encoder = define_pre_encoder(data_dim, layers=layers_e,units=units,BN=BN)
    if summ:
        print("pre-encoder network:")
        pre_encoder.summary()
    generator = define_generator(Nb,data_dim,layers=layers_d,units=units,BN=BN)
    if summ:
        print("generator network:")
        generator.summary()

    x = Input(shape=(data_dim,))
    #y = Input(shape=(n_classes,))

    hidden = pre_encoder(x)
    logits_b  = Dense(Nb, activation='linear', name='logits-b')(hidden) #log(B_j/1-B_j)
    #proba = np.exp(logits_b)/(1+np.exp(logits_b)) = sigmoidal(logits_b) <<<<<<<<<< recupera probabilidad
    #dist = Dense(Nb, activation='sigmoid')(hidden) #p(b) #otra forma de modelarlo
    
    if multilabel:
        supervised_layer = Dense(n_classes, activation='sigmoid',name='sup-class')(hidden)#req n_classes  
    else:
        supervised_layer = Dense(n_classes, activation='softmax',name='sup-class')(hidden)#req n_classes
     
    encoder = Model(x, logits_b)

    def sampling(logits_b):
        #logits_b = K.log(aux/(1-aux) + K.epsilon() )
        b = logits_b + sample_gumbel(K.shape(logits_b)) # logits + gumbel noise
        return keras.activations.sigmoid( b/tau )

    b_sampled = Lambda(sampling, output_shape=(Nb,), name='sampled')(logits_b)
    output = generator(b_sampled)
        
    Recon_loss = REC_loss
    kl_loss = BKL_loss(logits_b)

    def SUP_BAE_loss_pointwise(y_true, y_pred):
        #supervised_loss = keras.losses.categorical_crossentropy(y, supervised_layer)#req y 
        #return alpha*supervised_loss + Recon_loss(y_true, y_pred) + beta*kl_loss(y_true, y_pred)
        return Recon_loss(y_true, y_pred) + beta*kl_loss(y_true, y_pred)

    margin = Nb/3.0

    if multilabel:
        pred_loss = my_binary_KL_loss
    else:
        pred_loss = my_KL_loss

    def Hamming_loss(y_true, y_pred):
        
        #pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        r = tf.reduce_sum(b_sampled*b_sampled, 1)
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(b_sampled, tf.transpose(b_sampled)) + tf.transpose(r) #BXB
     
        similar_mask = K.dot(y_true, K.transpose(y_true)) #BXB  M_ij = I(y_i = y_j)  
        loss_hamming = (1.0/Nb)*K.sum(similar_mask*D + (1.0-similar_mask)*K.relu(margin-D))

        return gamma*pred_loss(y_true, y_pred) + loss_hamming

    #binary_vae = Model(inputs=[x,y], outputs=output)
    #binary_vae.compile(optimizer=opt, loss=SUP_BAE_loss_pointwise, metrics=[Recon_loss,kl_loss])

    binary_vae = Model(inputs=x, outputs=[output,supervised_layer])
    binary_vae.compile(optimizer=opt, loss=[SUP_BAE_loss_pointwise,Hamming_loss],loss_weights=[1., alpha], metrics=[Recon_loss,kl_loss,pred_loss])

    if tau_ann:
        return binary_vae, encoder,generator ,tau
    else:
        return binary_vae, encoder,generator

def SSBVAE(data_dim,n_classes,Nb,units,layers_e,layers_d,opt='adam',BN=True, summ=True,tau_ann=False,beta=0,alpha=1.0,gamma=1.0,multilabel=False):
    if tau_ann:
        tau = K.variable(1.0, name="temperature") 
    else:
        tau = K.variable(0.67, name="temperature") #o tau fijo en 0.67=2/3
    
    pre_encoder = define_pre_encoder(data_dim, layers=layers_e,units=units,BN=BN)
    if summ:
        print("pre-encoder network:")
        pre_encoder.summary()
    generator = define_generator(Nb,data_dim,layers=layers_d,units=units,BN=BN)
    if summ:
        print("generator network:")
        generator.summary()

    x = Input(shape=(data_dim,))
    #y = Input(shape=(n_classes,))

    hidden = pre_encoder(x)
    logits_b  = Dense(Nb, activation='linear', name='logits-b')(hidden) #log(B_j/1-B_j)
    #proba = np.exp(logits_b)/(1+np.exp(logits_b)) = sigmoidal(logits_b) <<<<<<<<<< recupera probabilidad
    #dist = Dense(Nb, activation='sigmoid')(hidden) #p(b) #otra forma de modelarlo
    
    if multilabel:
        supervised_layer = Dense(n_classes, activation='sigmoid',name='sup-class')(hidden)#req n_classes  
    else:
        supervised_layer = Dense(n_classes, activation='softmax',name='sup-class')(hidden)#req n_classes
     
    encoder = Model(x, logits_b)

    def sampling(logits_b):
        #logits_b = K.log(aux/(1-aux) + K.epsilon() )
        b = logits_b + sample_gumbel(K.shape(logits_b)) # logits + gumbel noise
        return keras.activations.sigmoid( b/tau )

    b_sampled = Lambda(sampling, output_shape=(Nb,), name='sampled')(logits_b)
    output = generator(b_sampled)
        
    Recon_loss = REC_loss
    kl_loss = BKL_loss(logits_b)

    def SUP_BAE_loss_pointwise(y_true, y_pred):
        #supervised_loss = keras.losses.categorical_crossentropy(y, supervised_layer)#req y 
        #return alpha*supervised_loss + Recon_loss(y_true, y_pred) + beta*kl_loss(y_true, y_pred)
        return Recon_loss(y_true, y_pred) + beta*kl_loss(y_true, y_pred)

    margin = Nb/3.0

    if multilabel:
        pred_loss = my_binary_KL_loss_stable
    else:
        pred_loss = my_KL_loss

    def Hamming_loss(y_true, y_pred):
        
        #pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        r = tf.reduce_sum(b_sampled*b_sampled, 1)
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(b_sampled, tf.transpose(b_sampled)) + tf.transpose(r) #BXB
     
        similar_mask = K.dot(y_pred, K.transpose(y_pred)) #BXB  M_ij = I(y_i = y_j)  
        loss_hamming = (1.0/Nb)*K.sum(similar_mask*D + (1.0-similar_mask)*K.relu(margin-D))

        return gamma*pred_loss(y_true, y_pred) + loss_hamming

    #binary_vae = Model(inputs=[x,y], outputs=output)
    #binary_vae.compile(optimizer=opt, loss=SUP_BAE_loss_pointwise, metrics=[Recon_loss,kl_loss])

    binary_vae = Model(inputs=x, outputs=[output,supervised_layer])
    binary_vae.compile(optimizer=opt, loss=[SUP_BAE_loss_pointwise,Hamming_loss],loss_weights=[1., alpha], metrics=[Recon_loss,kl_loss,pred_loss])

    if tau_ann:
        return binary_vae, encoder,generator ,tau
    else:
        return binary_vae, encoder,generator

