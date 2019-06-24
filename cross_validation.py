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

MAX_SEQUENCE_LENGTH = 40
MAX_NUM_WORDS = 40000
N_FOLD = 5
N_REPEAT = 3

train_data = pickle.load(open("data/train_data", "rb"))
train_labels = pickle.load(open("data/train_labels", "rb"))

emb_path = "data/embedding_matrix"
batch_sz = 32
ep = 2
n_filter = (100, 100, 100)
ker_size = (2, 3, 4)

def score_(y_true, y_pred, scorer, PN_only=False):
    """
    Compute recall for each class and average the result (see SemEval2017)
    :return: macroaveraged_recall.
    """
    n_class = len(y_true[0])
    true_vects = [[] for i in range(n_class)]
    pred_vects = [[] for i in range(n_class)]
    
    for i in range(len(y_true)):
        for j in range(n_class):
            true_vects[j].append(y_true[i][j])
            pred_vects[j].append(y_pred[i][j])
    
    recalls = [ scorer(true_vects[i], pred_vects[i]) for i in [0,1,2]]
    avg_ret = np.average(recalls) if not PN_only else np.average(np.array(recalls[0], recalls[2])) 
    return recalls, avg_ret

def get_test_data(test_sents, tokenizer, word_index):
    test_text = []
    test_labls = []

    for s in test_sents:
        test_text.append(s[0])
        test_labls.append(s[1])

    print('Found %s tweets.' % len(test_sents))

    tokenizer.word_index = word_index
    tokenizer.fit_on_texts(test_text)
    tokenizer.word_index = word_index
    test_sequences = tokenizer.texts_to_sequences(test_text)

    #test_word_index = word_index
    print('Found %s unique tokens.' % len(word_index))

    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    test_labels = to_categorical(np.asarray(test_labls))
    
    print('Shape of data tensor:', test_data.shape)
    print('Shape of label tensor:', test_labels.shape)

    x_test = test_data
    y_test = test_labels
    
    return x_test, y_test

def to_category(y_test_pred):
    y_test_mod = []
    for i in range(len(y_test_pred)):
        tmp = y_test_pred[i]
        y_test_mod.append([0.]*3)
        y_test_mod[-1][tmp.argmax()] = 1.
    y_test_mod = np.array(y_test_mod)
    return y_test_mod

def train_test_split_cv(x, y, n_iter, n_fold = 10):
    """
    ...
    """
   
    x=list(x)
    y=list(y)
    n_iter = n_fold if n_iter > n_fold else n_iter 
    ns_fold = int(len(x)/n_fold)
    
    test_start_idx = (n_iter-1)*ns_fold
    test_end_idx = (n_iter)*ns_fold
    
    #print("Test starts at "+str(test_start_idx)+" ending at "+str(test_end_idx))
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
    #macro_averaged_recall, categorical accuracy on validation
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
        #'accuracy' : accuracy_score,
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
            
            #fold_res[it].append(mavg)
            
            del model
            del y_pred
            del x_train
            del y_train
            del x_val
            del y_val
            
            gc.collect()

        #step_avg.append(np.average(fold_res[it]))
        #step_std.append(np.std(fold_res[it])) 
                        
        #print("\n--------------------------------> Average mavg_recall at step #"+str(it+1)+" = "+str(step_avg[it])[:5]+" +/-"+str(step_std[it])[:5]+"<-------------------------------")
        #results.append(step_avg[it])
    
    return scores

def create_model(kernel_size = (2, 3, 4), n_filter = (100, 100, 100), dropout = 0., activation = 'relu'):
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
            "embedding" : "TW200"
        })

    if len(sys.argv) > 1:
        print("=========================================\n\nCV parameters:")
        for ar in sys.argv[1:]:
            param = switch_param(ar[0], ar[1:])
            print(param)
            params.update(param)
        
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

print("Setting GPU limitations...")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)
print("GPU limitations set")

if len(kernel_size) != len(n_filter):
    print("kernel_size(k) and n_filter(n) must match in tuple size | k="+str(len(kernel_size))+" n="+str(len(n_filter)))
    quit(-1)

model_tag = "b"+str(batch_size)+"-e"+str(epochs)+"-n"+str(n_filter)+"-k"+str(kernel_size)+"-x"+str(embedding)+"-d"+str(dropout)+"-a"+str(activation)

print("\n["+str(time.asctime())+"] Started a "+str(N_FOLD)+"-fold cv with "+model_tag)
print("\n==================================================================================================\n")

res = cross_validation(train_data, train_labels, n_fold = N_FOLD, n_repeat = N_REPEAT, **params)
print("\n=================================================================================================================\n")
print("["+str(time.asctime())+"] completed a "+str(N_FOLD)+"-fold cv with "+model_tag+
    "\n mavg_recall on validation = "+str(np.average(res['mavg_recall']))[:5] + "+/-"+str(np.std(res['mavg_recall']))[:5] 
    +" accuracy on validation = "+str(np.average(res['accuracy']))[:5] + "+/-"+str(np.std(res['accuracy']))[:5]+ 
    " f1 on validation = "+str(np.average(res['f1']))[:5] + "+/-"+str(np.std(res['f1']))[:5])

cv_result = dict({model_tag:res})

pickle.dump(file=open('cv_result/cv_result_'+model_tag, 'wb'), obj=cv_result)
