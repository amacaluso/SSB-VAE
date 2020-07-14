import numpy as np
import pandas as pd
import gc, sys, os
from PIL import Image
from optparse import OptionParser
import keras
def resize_img(img,o_s, w,h,):
    img = img.reshape(o_s)
    return np.array( Image.fromarray(img).resize((w,h), Image.ANTIALIAS) )

op = OptionParser()
op.add_option("-d", "--data", type="string", default='mnist', help="data to project into VGG layers")
op.add_option("-p", "--path", type="string", default='data/', help="path to data")
op.add_option("-m", "--poolm", type="string", default='', help="pooling mode used on VGG (None or empty/avg/max)")
op.add_option("-s", "--size", type=int, default=0, help="image size to redim")

(opts, args) = op.parse_args()
data_used = opts.data.lower().strip()
pool_mo = opts.poolm
folder = opts.path.strip()
size_I = opts.size

if data_used == "mnist":
    (X_img, _), (X_test, _) = keras.datasets.mnist.load_data()
    X_img = np.expand_dims(X_img, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    X_img = np.concatenate((X_img,X_test),axis=0)
    #greyscale to rgb..
    import tensorflow as tf
    X_img = tf.image.grayscale_to_rgb(X_img).eval(session=keras.backend.get_session())
    del X_test
    gc.collect()

elif data_used == "cifar10":
    (X_img, _), (X_test, _) = keras.datasets.cifar10.load_data()
    X_img = np.concatenate((X_img,X_test),axis=0)
    del X_test
    gc.collect()
    
elif data_used == "nuswide":
    import repackage
    repackage.up()
    from utils import load_imgs_mask, enmask_data,get_topK_labels,set_newlabel_list
    list_images = pd.read_csv(folder+"/ImageList/Imagelist_downloaded.txt", header=None).iloc[:,0]
    imgs_files = [folder+"/small_images/" + value for value in list_images.values]
    
    mask_av = np.loadtxt("../Nus-Wide_mask_avail.txt").astype(bool)
    imgs_files = enmask_data(imgs_files, mask_av)

    
    labels = pd.read_csv(folder+'Concepts81.txt',header=None).values.reshape(1,-1)[0]
    labels_t = [[] for _ in range(269648)]
    for concept in labels:
        aux = pd.read_csv(folder+"Groundtruth/AllLabels/Labels_"+concept+".txt",header=None)
        indexs_true = aux.loc[(aux==1).values[:,0]].index

        for value in indexs_true:
            labels_t[value].append(concept)        
    labels_t = enmask_data(labels_t, mask_av)
    
    new_labels = get_topK_labels(labels_t, labels, K=21)
    labels_t = set_newlabel_list(new_labels, labels_t)
    
    mask_used_t = np.asarray(list(map(len,labels_t))) != 0
    X_img = load_imgs_mask(imgs_files, mask_used_t, size=size_I) #images names to load
    
    
elif data_used == "celeba":
    import repackage
    repackage.up()
    from utils import load_imgs_mask

    mask_av = np.loadtxt("../CelebA_mask_avail.txt").astype(bool)
    df_atrr = pd.read_csv(folder+"list_attr_celeba.csv")[mask_av]
    img_names = df_atrr["image_id"].values
    imgs_files = [folder+ "imgs_celebA/"+ name for name in img_names]

    N_total = len(df_atrr)
    mask_used= np.ones(N_total, dtype=bool)
    X_img = load_imgs_mask(imgs_files, mask_used, size=size_I) #images names to load

else:
    print("Dont known dataset selected")
    assert False

print("Images shapes: ",X_img.shape)

if size_I == 0:
    size_I =  X_img.shape[1]
    
    
if X_img.shape[1] != size_I:
    #reshape...    
    new_ = []
    for img in X_img:
        new_.append(np.array( Image.fromarray(img).resize((size_I,size_I), Image.ANTIALIAS) ))
#    del X_img
#    gc.collect()
    X_img = np.asarray(new_)
    del new_ , img
    gc.collect()
    
#now pass through VGG
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as p16
from keras.layers import Input
def through_VGG(X,modelVGG):
    """
        Pass data X through VGG 16
        * pooling_mode: as keras say could be None, 'avg' or 'max' (in order to reduce dimensionality)
    """
    X_vgg = p16(X.astype('float32'))
    return modelVGG.predict(X_vgg)

if pool_mo == "":
    pool_mo = None

input_tensor = Input(shape=X_img.shape[1:])
modelVGG = VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor,pooling=pool_mo ) # LOAD PRETRAIN$
modelVGG.summary()
    
BS = 2500 #change it based on your RAM memory!
new_X = []
for chunk in np.split(X_img, range(0, len(X_img)+BS, BS))[1:-1]:
    new_X.append(through_VGG(chunk, modelVGG))
    gc.collect()
new_X = np.concatenate(new_X, axis=0)
del modelVGG
gc.collect()

#new_X = through_VGG(X_img.astype('float32'),pooling_mode=pool_mo) #process all at once
print("New shape through VGG: ",new_X.shape)
if pool_mo == None:
    np.save(data_used+'_VGG.npy',new_X) #none pooling
else:
    np.save(data_used+'_VGG_'+pool_mo+'.npy',new_X) #avg/max pooling
