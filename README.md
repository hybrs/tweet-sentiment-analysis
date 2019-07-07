## Sentiment Analysis on Tweets:
### Message Polarity Classification | NLP Master Course @ UniPi
#### Amendola M., Cornacchia G. and Salinas M.L.
<hr>

In repository there is all the code used to run the experiments in our [project report](link/al/report). We tackle the message polarity classification task, that is given a message decide whether it expresses negative, neutral or positive sentiment. We developed and validated the CNN from [Zhang and Wallace, 2015](https://arxiv.org/pdf/1510.03820.pdf) and compared the performance of this system with a new method of pre-training language representations, called [BERT](https://github.com/google-research/bert), which obtains state-of-the-art results on a wide array of NLP tasks.

All our experiments were run on a linux server with an nVIDIA Tesla K40 accelerated GPU, that kindly provided to us by Professor Giuseppe Attardi @ UniPi.

The root of the project contains:

 - the python script `run_cnn.py`, that implements the CNN and allow the user to choose between Cross-Validation and Test mode. See [**Scripts**](#sec:scripts) and [**Invocation**](#sec:invocation) for further details. 
 - the python script `run_bertft.py`, the same as `run_cnn.py` but implements both the fine-tuned BERT systems involved in our analysis. 
 - the Jupyter Notebook `Data_Cleaning.ipynb`, contains the data pre-processing pipeline. (*the code in this notebook has to be executed BEFORE the python scripts*)
 - altri script?
 - the folder `cv_result`, output folder for our scripts in cv mode.
 - the folder `results_test`, output folder for our script in test mode.

### Requirements 
Code is written in Python (3.6.8) and requires Keras (2.2.4), Tensorflow (1.13.1) and [tweet-preprocessor](https://pypi.org/project/tweet-preprocessor/) (1.3.1).

Before running the scripts make sure that your data has been preprocessed as illustrated in `Data_Cleaning.ipynb`.

### Scripts

For the CNN `CNN/run_cnn.py` for BERT `BERT/run_bertft.py`

<a id="sec:invocation"></a>
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

