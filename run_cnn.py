from collections import defaultdict
import os
import gc
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Embedding, Flatten, concatenate, Activation
from keras.models import Model, Sequential
from sklearn.metrics import recall_score
import operator
import pickle
import time
from sklearn.metrics import f1_score, accuracy_score
from keras.models import model_from_json

MAX_SEQUENCE_LENGTH = 40
N_FOLD = 5
N_REPEAT = 3

train_data = pickle.load(open("data/CNN/train/train_data", "rb"))
train_labels = pickle.load(open("data/CNN/train/train_labels", "rb"))

emb_path = "data/CNN/embedding_matrix"
batch_sz = 32
ep = 2
n_filter = (100, 100, 100)
ker_size = (2, 3, 4)

def score_(y_true, y_pred, scorer, PN_only=False):
    """
    Function that computes a score between a two vectors of labels.
    :param y_true: target labels
    :param y_pred: prediction labels
    :param scorer: scorer function (e.g. one in sklearn.metrics)  
    :return: class_scores, a list with the computed score for each class 
            (negative, neutral, positive) and avg_ret the average of the score among the classes.
    """
    n_class = len(y_true[0])
    true_vects = [[] for i in range(n_class)]
    pred_vects = [[] for i in range(n_class)]
    
    for i in range(len(y_true)):
        for j in range(n_class):
            true_vects[j].append(y_true[i][j])
            pred_vects[j].append(y_pred[i][j])
    
    class_scores = [ scorer(true_vects[i], pred_vects[i]) for i in [0,1,2]]
    avg_ret = np.average(class_scores) if not PN_only else np.average([class_scores[0], class_scores[2]])
    return class_scores, avg_ret

def to_category(y_test_pred):
    """
    Function that outputs one-hot representation of a list of integers (multiclass classification labels).
    :param y_test_pred: label list to be converted  
    :return: one-hot representation of the input
    """
    y_test_mod = []
    for i in range(len(y_test_pred)):
        tmp = y_test_pred[i]
        y_test_mod.append([0.]*3)
        y_test_mod[-1][tmp.argmax()] = 1.
    y_test_mod = np.array(y_test_mod)
    return y_test_mod

def train_test_split_cv(x, y, n_iter, n_fold = 5):
    """
    Split data and labels for a specific cross validaton fold.
    :param x: input data
    :param y: input labels
    :param n_iter: fold for which we want the split (we used fixed folds across our experiments) 
    :param n_fold: total number of folds 
    :return: data and labels for train and validation of fold number n_iter of n_fold
    """   
    x=list(x)
    y=list(y)
    n_iter = n_fold if n_iter > n_fold else n_iter 
    ns_fold = int(len(x)/n_fold)
    
    test_start_idx = (n_iter-1)*ns_fold
    test_end_idx = (n_iter)*ns_fold
    
    x_train, y_train = [], []
    
    if test_start_idx == 0:
        x_train = x[test_end_idx:]
        y_train = y[test_end_idx:]
    elif test_end_idx == len(x):
        x_train = x[:test_start_idx]
        y_train = y[:test_start_idx]
    else:
        x_train = x[:test_start_idx]+x[test_end_idx:] 
        y_train = y[:test_start_idx]+y[test_end_idx:] 

    x_test = x[test_start_idx:test_end_idx]
    y_test = y[test_start_idx:test_end_idx]

    
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def cross_validation(x, y, n_fold = 3, n_repeat = 1, **params):
    """
    Perform n_repeat of n_fold Cross Validation given train data and labels and a parameter dictionary.
    :param x: train data 
    :param y: train labels
    :param n_fold: number of folds
    :param n_repeat: number of times the n_fold-cv will be executed  
    :param params: dictionary of parameters like the one defined in get_params()
    :return: a dictionary containing the collected scores for the cross validation.
    """
    results = []
    batch_size = params.get('batch_size')
    epochs = params.get('epochs')
    ks = params.get('kernel_size')
    nf = params.get('n_filter')
    dropout = params.get('dropout')
    emb = params.get('embedding')
    act = params.get('activation')
    
    #random seed init
    step_tot = 0
    step_avg = []
    step_std = []

    functions=dict({
        'accuracy' : accuracy_score,
        'f1' : f1_score,
        'mavg_recall': recall_score
        })
    scores = dict({
        'accuracy': [],
        'f1': [],
        'mavg_recall': []
        })

    for it in range(n_repeat):
        #fold_res.append([])
        for i in range(1, n_fold+1):
            x_train, y_train, x_val, y_val = train_test_split_cv(x, y, i, n_fold=n_fold)
            
            model = create_model(kernel_size = ks, n_filter = nf, dropout = dropout, activation = act)

            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
            
            #predict + MULTIPLE scoring su test/validation
            
            step_tot+=1
            y_pred = to_category(model.predict(x_val, batch_size = batch_size))

            scr = 0. 

            for key in functions.keys():
                if(key == 'accuracy'):
                    scr = model.evaluate(x_val, y_val, verbose=0)
                    scr = scr[1]
                else: 
                    r_, scr = score_(y_val, y_pred, functions.get(key)) if not(key == 'f1') else score_(y_val, y_pred, functions.get(key), PN_only=True) 
                
                scores[key].append(scr)
                    
            if i == 1:
                print("\n")
            print("["+str(time.asctime())+"] Step #"+str(it+1)+"."+str(i)+" ("+str(step_tot)+"/"+str(n_repeat*n_fold)+") mavg_recall on validation = "+str(scores['mavg_recall'][-1])[:5] + " accuracy on validation = "+str(scores['accuracy'][-1])[:5] + " f1 on validation = "+str(scores['f1'][-1])[:5])#+" - ca = "+str(ca))
            print("---------------------------------------------------------------------------------")
            
            del model
            del y_pred
            del x_train
            del y_train
            del x_val
            del y_val
            
            gc.collect()

    return scores

def create_model(kernel_size = (2, 3, 4), n_filter = (100, 100, 100), dropout = 0., activation = 'relu'):
    """
    Compiles and return a CNN for text classification based on [Zhang and Wallace, 2015].
    """
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    tweet_encoder = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], 
                              input_length=MAX_SEQUENCE_LENGTH, trainable=True)(tweet_input)
    
    conv_branches = []

    for k in range(len(kernel_size)):
        conv_branches.append(Conv1D(filters=n_filter[k], kernel_size=kernel_size[k], padding='valid', activation=activation, strides=1)(tweet_encoder))
        conv_branches[k] = GlobalMaxPooling1D()(conv_branches[k])

    merged = concatenate(conv_branches, axis=1) if len(conv_branches) > 1 else conv_branches[0]

    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(dropout)(merged) if dropout > 0 else merged
    merged = Dense(3)(merged)
    output = Activation('sigmoid')(merged)
    model_ZW = Model(inputs=[tweet_input], outputs=[output])
    model_ZW.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])
    return model_ZW

def switch_param(argument, val):
    if argument == 'b': return dict({'batch_size':int(val)})
    elif argument =='e': return dict({'epochs':int(val)})
    elif argument =='k': 
        value = tuple(map(int,val.split(','))) if ',' in val else tuple([int(val)])
        return dict({'kernel_size':value})
    elif argument =='n':
        value = tuple(map(int, val.split(','))) if ',' in val else tuple([int(val)])
        return dict({'n_filter':value}) 
    elif argument =='x': return dict({'embedding':val})
    elif argument =='d': return dict({'dropout':float(val)})
    elif argument =='r': return dict({'regularization':float(val)})
    elif argument =='a': return dict({'activation':val})
    elif argument =='m': return dict({'mode':val})
    else:
        print("Invalid parameter "+argument)
        return

def get_param():
    params = dict({
            "batch_size":batch_sz,
            "epochs":ep,
            "n_filter" : n_filter,
            "kernel_size" : ker_size,
            "dropout" : 0.,
            "activation" : 'relu',
            "embedding" : "TW200",
            "mode" : "cv"
        })

    if len(sys.argv) > 1:
        print("=========================================\n\nCV parameters:")
        for ar in sys.argv[1:]:
            param = switch_param(ar[0], ar[1:])
            print(param)
            params.update(param)
    #Here we load the PRE-COMPUTED embedding matrix for the lookup layer of the CNN
    embedding_matrix = pickle.load(open(emb_path+params.get('embedding'), "rb"))   
    
    return params, embedding_matrix


params, embedding_matrix = get_param()

batch_size = params.get('batch_size')
epochs = params.get('epochs')
kernel_size = params.get('kernel_size')
n_filter = params.get('n_filter')
dropout = params.get('dropout')
activation = params.get('activation')
embedding = params.get('embedding')
mode = params.get('mode')

functions=dict({
    'accuracy' : accuracy_score,
    'f1' : f1_score,
    'mavg_recall': recall_score
    })
scores = dict({
    'accuracy': [],
    'f1': [],
    'mavg_recall': []
    })

results = dict({})
obj2save = []

print("Setting GPU limitations...")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)
print("GPU limitations set")

if len(kernel_size) != len(n_filter):
    print("kernel_size(k) and n_filter(n) must match in tuple size | k="+str(len(kernel_size))+" n="+str(len(n_filter)))
    quit(-1)
if mode != 'cv' and mode!='test':
    print("mode must be cv or test.")
    quit(-1)
model_tag = "b"+str(batch_size)+"-e"+str(epochs)+"-n"+str(n_filter)+"-k"+str(kernel_size)+"-x"+str(embedding)+"-d"+str(dropout)+"-a"+str(activation)

#Cross Validation mode
if mode == 'cv':
    print("cross_validation mode")
    res = cross_validation(train_data, train_labels, n_fold = N_FOLD, n_repeat = N_REPEAT, **params)
    print("\n=================================================================================================================\n")
    print("["+str(time.asctime())+"] completed a "+str(N_FOLD)+"-fold cv with "+model_tag+
        "\n mavg_recall on validation = "+str(np.average(res['mavg_recall']))[:5] + "+/-"+str(np.std(res['mavg_recall']))[:5] 
        +" accuracy on validation = "+str(np.average(res['accuracy']))[:5] + "+/-"+str(np.std(res['accuracy']))[:5]+ 
        " f1 on validation = "+str(np.average(res['f1']))[:5] + "+/-"+str(np.std(res['f1']))[:5])

    cv_result = dict({model_tag:res})
    #pickle.dump(file=open('resultsCNN/'+mode+'_result_'+model_tag, 'wb'), obj=cv_result)
#Test mode
elif mode == 'test':
    print("test mode")
    print("Start training")
    model = create_model(kernel_size = kernel_size, n_filter = n_filter, dropout = dropout, activation = activation)
    model.fit(train_data, train_labels, batch_size=32, epochs=2, verbose=1)
    print("Scores on training")
    y_pred = to_category(model.predict(train_data, batch_size = batch_size))
    gar, accuracy = model.evaluate(train_data, train_labels, verbose=1)

    gar, mavg = score_(train_labels, y_pred, functions.get('mavg_recall'))

    gar, f1 = score_(train_labels, y_pred, functions.get('f1'), PN_only=True) 

    print("Accuracy: ",accuracy)
    print("Mavg_recall: ",mavg)
    print("F1-score: ",f1)
    print("Class F1",gar)    


    test_files = ['2013','2013sms','2014','2014livej','2014sar','2015','2016','2017']

    print("Start TEST")
        
    for t in test_files:
        print("****************************************************************************")
        print("TEST: ",t)
        x_test = pickle.load(open("data/CNN/test/test_data_"+t, "rb"))
        y_test = pickle.load(open("data/CNN/test/test_labels_"+t, "rb"))

        y_pred = to_category(model.predict(x_test, batch_size = batch_size))

        gar, accuracy = model.evaluate(x_test, y_test, verbose=1)

        gar, mavg = score_(y_test, y_pred, functions.get('mavg_recall'))

        gar, f1 = score_(y_test, y_pred, functions.get('f1'), PN_only=True) 

        print("Accuracy: ",accuracy)
        print("Mavg_recall: ",mavg)
        print("F1-score: ",f1)
        print("Class F1",gar)    

        results[t]=({'accuracy':accuracy, 'mavg_recall':mavg, 'f1':f1})
        #pickle.dump(file=open('cv_result/'+mode+'_result_'+model_tag, 'wb'), obj=results)
