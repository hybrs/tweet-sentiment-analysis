## Sentiment Analysis on Tweets:
### Message Polarity Classification | NLT Master Course @ UniPi
#### Amendola M., Cornacchia G. and Salinas M.L.
<hr>

In repository there is all the code used to run the experiments in our [project report](link/al/report). We tackle the message polarity classification task, that is given a message decide whether it expresses negative, neutral or positive sentiment. We developed and validated the CNN from [Zhang and Wallace, 2015](https://arxiv.org/pdf/1510.03820.pdf) and compared the performance of this system with a new method of pre-training language representations, called [BERT](https://github.com/google-research/bert), which obtains state-of-the-art results on a wide array of NLP tasks. We performed 3 repetition of 5-fold-cv for each experiment.

All our experiments were run on a linux server with an nVIDIA Tesla K40 accelerated GPU, kindly provided to us by Professor Giuseppe Attardi @ UniPi.

The root of the project contains:

 - the python script `run_cnn.py`, that implements the CNN and allow the user to choose between Cross-Validation and Test mode. See [**Scripts**](#sec_scripts) and [**Invocation**](#sec_invocation) for further details. 
 - the python script `run_bertft.py`, the same as `run_cnn.py` but implements both the fine-tuned BERT systems involved in our analysis. 
 - the Jupyter Notebook `Data_Cleaning.ipynb`, contains the data pre-processing pipeline and the functions to export data, labels and the embedding matrix needed in the lookup layer of CNN the model. (*the code in this notebook has to be executed BEFORE the python scripts*)
 - the python script `run_classifier_.py`, adapted version of the orginal script in the BERT repo.
 - the folder `cv_result`, output folder for our scripts in cv mode.
 - the folder `results_test`, output folder for our script in test mode.
 - this README

### Requirements 
Code is written in Python (3.6.8) and requires Keras (2.2.4), Tensorflow (1.13.1) and [tweet-preprocessor](https://pypi.org/project/tweet-preprocessor/) (1.3.1).

Before running the scripts make sure that your data has been preprocessed as illustrated in `Data_Cleaning.ipynb`.

<a id="sec_scripts"></a>
### Scripts

For validating and assessing the risk of CNN we use `run_cnn.py`, while for BERT `run_bertft.py`.

#### Script for CNN

#### Script for BERT

To execute the script you must:
* download Bert repository and BERT-Base Uncased from [google-research](https://github.com/google-research/bert);
* put module `run_bertft.py` and `run_classifier_.py` into Bert directory. The latter is a modified version forked 

Example of script esecution: 

```
python run_bertft.py mode=test seq_len=50 epochs=3 reps=2 fold=10
```


Example of output of our script in cv mode: for each fold we print validation scores and scores for each class.

```
....

*******[ 15 / 15 ]*******
....
***** Scores for each class *****
Accuracy: 0.6923258003766478
Recall:  [0.65602322 0.62883087 0.76850306]
F1:  [0.63261022 0.65291691 0.75197386]
***** Averaged Scores *****
Accuracy: 0.6923258003766478
Recall:  0.6844523855748061 ---- 0.6844523855748061
F1:  0.69229204013095
=========== REPS 3 RESULTS ===========
Accuracy: 0.699482109227872
Recall:  0.6884577812392527
F1:  0.6911314647836555
```



#### Dataset 
All dataset (train and test) must be .tsv dataset where the last column is the tweet column and the penultimate column is the label column, as in the below example.

| ID  | Label   | Tweet   |
|:--: |:--------: |:---------------------:  |
| 32  | Positive  | It's a beautiful day!   |

#### Parameters
The script has some parameter with default value. 
The table below contains all the parameters you can change.

|  <center>  Name   </center>   | <center>  Values   </center>  |                                                      <center>   Description </center>                                                       |    <center>    Default value    </center>       |
|:----------: |:----------: |:------------------------------------------------------------------------------------------------------------------------: |:---------------------------:  |
|   <center> mode   </center>   |<center>train<br>test </center>  | <center>if train: the script runs cross_validation <br>  else: the script runs the tests                                      |       <center>     train </center>              |
|   <center> train   </center>  | <center>   path  </center>    | <center>path of the train dataset </center>                                                                                                 |  <center>  ./data/BERT_data/train/tweet_train_df.tsv   </center>    |
|  <center>  test   </center>   |   <center> path  </center>    | <center>path of the directory that contains test datasets  </center>                                                                        |        <center>    ./data/BERT_data/test   </center>          |
|  <center> softmax  </center>  |<center>0<br>1   </center>   | <center>if 1: the script runs BERT fine-tuning with softmax layer <br> else: the script run BERT fine-tuning with CNN </center>   |          <center>    1    </center>           |
|<center> batch_size </center>  | <center>  int > 0  </center>  | <center>batch size   </center>                                                                                                              |            <center>  32    </center>          |
|  <center> seq_len </center>   | <center>  int > 0   </center>| <center>sequence length</center>                                                                                                             |             <center> 40  </center>            |
|  <center> epochs  </center>   | <center>  int > 0  </center>  | <center>number of epochs  </center>                                                                                                         |           <center>   2   </center>            |
|  <center>  reps   </center>   | <center>  int > 0  </center>  | <center>number of repetition of cross-validation   </center>                                                                                |          <center>    3  </center>             |
|   <center> fold   </center>   | <center>  int > 0  </center>  | <center>number of fold for cross-validation </center>                                                                                       |          <center>    5   </center>            |
|  <center> print  </center>  |<center>0<br>1   </center>   | <center>if 1: at each step prints the recall and f1 scores for each class  </center>  |          <center>    0    </center> 
    |



<a id="sec_invocation"></a>
### Invocation
```
usage: run_[SCRIPT_SUFF].py [-mode] [-bBATCH_SIZE] [-eEPOCHS] [-kKERNEL_SIZE(S)] [-nN_FILTER(S)]
                            [-aACTIVATION] [-dDROPOUT] [-xEMBEDDING_SUFF]
                            model input

CNN sentence/tweet classifier.

positional arguments:
  model                 model file (default mr)
  input                 train/test file in SemEval twitter format

optional arguments:
  -h, --help            show this help message and exit
  -train                train model
  -lower                whether to lowercase text
  -filters FILTERS      n[,n]* (default 3,4,5)
  -vectors VECTORS      word2vec embeddings file (random values if missing)
  -dropout DROPOUT      dropout probability (default 0.5)

.......


```

VECTORS is the word2vec binary file (e.g. `GoogleNews-vectors-negative300.bin` file),
clean, if present, text is lower-cased.

input contain sentences from the corpus in SemEval format, i.e. one sentence
per line, tab separated values:

```
ID	UID	text	label
```

Also a sentence classifier is present, that does cross validation on the training set for evaluating the performance of the classifier, like in the original code:
```
	other code
```

### Hyperparameters
Ye Zhang has written a [very nice paper](http://arxiv.org/abs/1510.03820) doing an extensive analysis of model variants (e.g. filter widths, k-max pooling, word2vec vs Glove, etc.) and their effect on performance.

### SemEval-2017 SubTask 4A: Sentiment Analysis in Twitter

Experiment using test data from reruns of SemEval from 2013 up to 2017, running `run_bertft.py` option *parameters* achieves state of the art scores across multiple test sets. *(e.g as reported in [Rosenthal et al., 2017](http://alt.qcri.org/semeval2017/task4/data/uploads/semeval2017-task4.pdf) in 2017 the top f1-score was 0.00000000, our system achives 0.00000000)*.

