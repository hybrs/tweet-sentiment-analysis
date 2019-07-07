## Sentiment Analysis on Tweets:
### Message Polarity Classification | HLT Master Course @ UniPi
#### Amendola M., Cornacchia G. and Salinas M.L.
<hr>

In repository there is all the code used to run the experiments in our [project report]([HLT]Sentiment_Analysis_on_Tweets.pdf). We tackle the message polarity classification task, that is given a message decide whether it expresses negative, neutral or positive sentiment. We developed and validated the CNN from [Zhang and Wallace, 2015](https://arxiv.org/pdf/1510.03820.pdf) and compared the performance of this system with a new method of pre-training language representations, called [BERT](https://github.com/google-research/bert), which obtains state-of-the-art results on a wide array of NLP tasks. We performed 3 repetition of 5-fold-cv for each experiment.

All our experiments were run on a linux server with an nVIDIA Tesla K40 accelerated GPU, kindly provided to us by Professor Giuseppe Attardi @ UniPi.

The root of the project contains:

 - the python script `run_cnn.py`, that implements the CNN and allow the user to choose between Cross-Validation and Test mode. See [**Scripts**](#sec_scripts) for further details. 
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

Once you've created your `train_data`, `train_labels` and `embedding_matrix*` files using the data pre-proc pipeline in `Data_Cleaning.ipynb`, and put them in the `data` folder, you can run the script in this way:

```
python run_cnn.py mtest k5,5 n100,100
```
the following table provides additional information on the parameters.

| <center>Name  | <center>Values  |<center> Description   | <center>Default Value   |
|:----: |:-------------:  |-------------------------------------------------------------------------------------  |------------------------------ |
| <center>b</center>  | <center>int > 0</center>  | <center> batch size </center>   | <center> 32 </center>   |
| <center>e </center> | <center>int > 0</center>  | <center> number of epochs </center>   | <center>2</center>  |
| <center>k </center> | <center>tuple </center> | <center> seq of int comma separated that specifies kernels size </center>   | <center>2,3,4</center>  |
| <center>n </center> | <center>tuple </center> | <center> seq of int comma separated that specifies filters size </center>   | <center>100,100,100</center>  |
| <center>x </center> | <center>string</center>   | <center> suffix of the matrix build with the data cleaning pipeline </center>   | <center>TW200</center>  |
| <center>a </center> | <center>string </center>  | <center> activation function </center>  | <center>relu</center>   |
| <center>d</center>  | <center>float >= 0</center>   | <center> dropout </center>  | <center>0.0</center>  |
| <center>m   </center>| <center>cv <br> test </center> | <center> if cv: script runs cross-validation <br> else: script runs tests</center>  | <center> cv </center>   |

Example of output of the script in test mode: verbose mode is on during train and we print scores, for both train and test set, for the single classes(*negative, neutral, positive*) and averaged.

```
Epoch 1/2
21240/21240 [==============================] - 140s 7ms/step - loss: 0.8765 - categorical_accuracy: 0.5734
Epoch 2/2
21240/21240 [==============================] - 154s 7ms/step - loss: 0.5440 - categorical_accuracy: 0.7769
Scores on training
21240/21240 [==============================] - 16s 765us/step
Accuracy:  0.953436911465309
Mavg_recall:  0.9510301922854709
F1-score:  0.9475169784506192
Class F1 [0.9316843345111896, 0.9516968325791855, 0.9633496223900488]
Start TEST
**************************
TEST:  2013
3547/3547 [==============================] - 2s 678us/step
Accuracy:  0.41217930642255163
Mavg_recall:  0.34760255236915666
F1-score:  0.2617124634916851
Class F1 [0.13157894736842107, 0.4998584772148316, 0.39184597961494905]
```

#### Script for BERT

To execute the script you must:
* download Bert repository and BERT-Base Uncased from [google-research](https://github.com/google-research/bert);
* put module `run_bertft.py` and `run_classifier_.py` into Bert directory. The latter is a modified version forked 
 
All dataset (train and test) must be .tsv dataset where the last column is the tweet column and the penultimate column is the label column, as in the below example.

| ID  | Label   | Tweet   |
|:--: |:--------: |:---------------------:  |
| 32  | Positive  | It's a beautiful day!   |


Example of script execution: 

```
python run_bertft.py mode=test seq_len=50 epochs=3 reps=2 fold=10
```

The script has some parameter with default value. The table below contains all the parameters you can change.

|  <center>  Name   </center>   | <center>  Values   </center>  |                                                      <center>   Description </center>                                                       |    <center>    Default value    </center>       |
|:----------: |:----------: |:------------------------------------------------------------------------------------------------------------------------: |:---------------------------:  |
|   <center> mode   </center>   |<center>train<br>test </center>  | <center>if train: the script runs cross_validation <br>  else: the script runs the tests                                      |       <center>     train </center>              |
|   <center> train   </center>  | <center>   path  </center>    | <center>path of the train dataset </center>                                                                                                 |  <center>  ./data/BERT_data/train/tweet_train_df.tsv   </center>    |
|  <center>  test   </center>   |   <center> path  </center>    | <center>path of the directory that contains test datasets  </center>                                                                        |        <center>    ./data/BERT_data/test   </center>          |
|  <center> softmax  </center>  |<center>0<br>1   </center>   | <center>if 1 run BERT fine-tuning with softmax layer <br> else run BERT fine-tuning with CNN </center>   |          <center>    1    </center>           |
|<center> batch_size </center>  | <center>  int > 0  </center>  | <center>batch size   </center>                                                                                                              |            <center>  32    </center>          |
|  <center> seq_len </center>   | <center>  int > 0   </center>| <center>sequence length</center>                                                                                                             |             <center> 40  </center>            |
|  <center> epochs  </center>   | <center>  int > 0  </center>  | <center>number of epochs  </center>                                                                                                         |           <center>   2   </center>            |
|  <center>  reps   </center>   | <center>  int > 0  </center>  | <center>number of repetition of cross-validation   </center>                                                                                |          <center>    3  </center>             |
|   <center> fold   </center>   | <center>  int > 0  </center>  | <center>number of fold for cross-validation </center>                                                                                       |          <center>    5   </center>            |
|  <center> print  </center>  |<center>0<br>1   </center>   | <center>if 1 prints the recall and f1 scores for each class  </center>  |          <center>    0    </center> |

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

### CNN Hyperparameters
Ye Zhang has written a [very nice paper](http://arxiv.org/abs/1510.03820) doing an extensive analysis of model variants (e.g. filter widths, k-max pooling, word2vec vs Glove, etc.) and their effect on performance.

### SemEval-2017 SubTask 4A: Sentiment Analysis in Twitter

Experiment using test data from reruns of SemEval from 2013 up to 2017, running `run_bertft.py` option *mode = test* achieves state of the art scores across multiple test sets. *(e.g as reported in [Rosenthal et al., 2017](http://alt.qcri.org/semeval2017/task4/data/uploads/semeval2017-task4.pdf) in 2017 the top f1-score was 0.685, our system achives 0.694)*.

