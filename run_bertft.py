import numpy as np 
import pandas as pd
import re
import gc
import os
import fileinput
import string
import tensorflow as tf
import datetime
import sys
import shutil


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

import modeling
import optimization
import run_classifier_ as run_classifier
import tokenization

from sklearn.metrics import confusion_matrix
# Load all files from a directory in a DataFrame.
def load_data(directory):
    
    lbl = {'negative':0,
           'neutral':1,
          'positive':2}
    
    data = {}
    data["tweet"] = []
    data["label"] = []
    with open(directory, "r", encoding='utf-8') as f:
        for line in f:       
            fields = line.strip().split("\t")
            if (len(fields) > 2):
                data['tweet'].append(fields[-1])
                data['label'].append(lbl.get(fields[-2]))
    return pd.DataFrame.from_dict(data)

def train_test_split_cv(inputs, n_iter):
    
    n_iter = N_FOLD if n_iter > N_FOLD else n_iter
    ns_fold = int(len(inputs)/N_FOLD)

    test_start_idx = (n_iter-1)*ns_fold
    test_end_idx = (n_iter)*ns_fold

    test=inputs[test_start_idx:test_end_idx]
    train=pd.concat([inputs[:test_start_idx],inputs[test_end_idx:]])
     
    return train, test

def create_examples(lines, set_type, labels=None):
#Generate data for the BERT model
    guid = f'{set_type}'
    examples = []
    if guid == 'train':
        for line, label in zip(lines, labels):
            text_a = line
            label = str(label)
            examples.append(
              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    else:
        for line in lines:
            text_a = line
            label = '0'
            examples.append(
              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    print(params)
    batch_size = 32

    num_examples = len(features)

    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

def train_estimator(train):
    train_examples = create_examples(train['tweet'], 'train', labels=train['label'])
    
    print('TRAIN: ', len(train))

    num_train_steps = int(
        len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    model_fn = run_classifier.model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
        num_labels=len(label_list),
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available  
        use_one_hot_embeddings=True,
        softmax=SOFTMAX)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available 
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE)

    train_features = run_classifier.convert_examples_to_features(
        train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
        
    
    train_input_fn = run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)
    print('Starting Train Phase')
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    return estimator

def predict(test, estimator):
    predict_examples = create_examples(test['tweet'], 'test')

    predict_features = run_classifier.convert_examples_to_features(
        predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

    predict_input_fn = input_fn_builder(
        features=predict_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    result = estimator.predict(input_fn=predict_input_fn)

    return result

def print_result(test, result):
    preds = []
    for prediction in result:
          preds.append(np.argmax(prediction['probabilities']))

    '''cm = confusion_matrix(list(test['label']),preds)
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy_=cm.diagonal()/cm.sum(axis=1)#cm.diagonal()'''
    accuracy_=accuracy_score(list(test['label']),preds)            

    recall_=recall_score(list(test['label']),preds, average=None)
    f1_=f1_score(list(test['label']),preds,average=None)

    if(PRINT):
        print('************** Scores for each class **************')
        print("Accuracy:",accuracy_)
        print("Recall: ", recall_)
        print('F1: ', f1_)

    accuracy_avg=np.average(accuracy_)
    recall_avg=np.average(recall_)
    f1_avg=np.average([f1_[0], f1_[2]])

    print('************** Averaged Scores **************')
    print("Accuracy:",accuracy_avg)
    print("Recall: ", recall_avg, '----', recall_score(list(test['label']),preds, average='macro'))
    print('F1: ', f1_avg)

    return (accuracy_avg,recall_avg, f1_avg)

def cross_validation(train):

    accuracy_reps=[]
    recall_reps=[]
    f1_reps=[]

    for it in range(REPS):

        accuracy_fold=[]
        recall_fold=[]
        f1_fold=[]

        for i in range(1,N_FOLD+1):

            train_cv, test_cv= train_test_split_cv(train, i)
            estimator=train_estimator(train_cv)
            result=predict(test_cv, estimator)
            print('**********************[',(N_FOLD*it)+i,'/',N_FOLD*REPS,']**********************')
            
            accuracy_, recall_,f1_=print_result(test_cv, result)
            accuracy_fold.append(accuracy_)
            recall_fold.append(recall_)
            f1_fold.append(f1_)

            try:
                shutil.rmtree('./outputs')
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))

            if(i==N_FOLD):
                accuracy_reps.append(np.mean(accuracy_fold))
                recall_reps.append(np.mean(recall_fold))
                f1_reps.append(np.mean(f1_fold))

        print('=========== REPS '+str(it+1)+' RESULTS ===========')
        print("Accuracy:",accuracy_reps[-1])
        print("Recall: ", recall_reps[-1])
        print('F1: ', f1_reps[-1])

    final_result={
        'accuracy':np.mean(accuracy_reps),
        'mavg_recall':np.mean(recall_reps),
        'f1':np.mean(f1_reps)
    }

    return final_result


def test_estimator(train, path_test):

    estimator=train_estimator(train)

    final_result={}

    for filename in os.listdir(path_test):

        if((not filename == '.') or (not filename == '..')):

            test=load_data(path_test+filename)
            print('Starting Test Phase: '+ filename + ' --- '+str(len(test))+' records')
            
            #test=test.sample(50)
            result=predict(test,estimator)
            
            accuracy_, recall_,f1_=print_result(test, result)
                       
            result={
                'accuracy': accuracy_,
                'mavg_recall': recall_,
                'f1':f1_
            }

            name=filename.split('.')[0]
            final_result[name]=result
    return final_result


BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = '../uncased_L-12_H-768_A-12'
OUTPUT_DIR = './outputs'
print(f'>> Model output directory: {OUTPUT_DIR}')
print(f'>>  BERT pretrained directory: {BERT_PRETRAINED_DIR}')

VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

tpu_cluster_resolver = None #Since training will happen on GPU, we won't need a cluster resolver
#TPUEstimator also supports training on CPU and GPU. You don't need to define a separate tf.estimator.Estimator.

EVAL_BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WARMUP_PROPORTION = 0.1

N_FOLD=5
REPS=3
SOFTMAX=1
# Model Hyper Parameters
TRAIN_BATCH_SIZE = 32
NUM_TRAIN_EPOCHS = 2.0
MAX_SEQ_LENGTH = 40

PRINT=0

MODE='train'

path_train='./data/BERT_data/train/tweet_train_df.tsv'
path_test='./data/BERT_data/test'

if len(sys.argv) > 1:
        for ar in sys.argv[1:]:
            value=ar.split('=')
            if value[0]=='train' : train_path=value[1]
            elif value[0]=='batch_size' : TRAIN_BATCH_SIZE=np.abs(int(value[1]))
            elif value[0]=='epochs' : NUM_TRAIN_EPOCHS=np.abs(int(value[1]))
            elif value[0]=='seq_len' : MAX_SEQ_LENGTH=np.abs(int(value[1]))
            elif value[0]=='reps' : REPS=np.abs(int(value[1]))
            elif value[0]=='fold' : N_FOLD=np.abs(int(value[1]))
            elif value[0]=='softmax' : SOFTMAX=int(value[1])
            elif value[0]=='mode' : MODE=value[1]
            elif value[0]=='test' : path_test=value[1]
            elif value[0]=='print' : PRINT=int(value[1])


train= load_data(path_train)
print('Train: ' +str(len(train))+' records')

#train=train.sample(100)

# Model configs
SAVE_CHECKPOINTS_STEPS = ((len(train)/N_FOLD)*(N_FOLD-1))-10 #if you wish to finetune a model on a larger dataset, use larger interval
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 100000

run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

label_list = ['0', '1', '2']
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

if(MODE=='train'):
    cross_validation(train)
else:
    test_estimator(train, path_test+'/')