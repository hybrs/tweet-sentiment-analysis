## Sentiment Analysis on Tweets:
### Message Polarity Classification | NLP Master Course @ UniPi
#### Amendola M., Cornacchia G. and Salinas M.L.
<hr>

In repository there is all the code used to run the experiments in our [project report](link/al/report). We tackle the message polarity classification task, that is given a message decide whether it expresses negative, neutral or positive sentiment. We developed and validated the CNN from [Zhang and Wallace, 2015](https://arxiv.org/pdf/1510.03820.pdf) and compared the performance of this system with a new method of pre-training language representations, called [BERT](https://github.com/google-research/bert), which obtains state-of-the-art results on a wide array of NLP tasks.

All our experiments were run on a linux server with an nVIDIA Tesla K40 accelerated GPU, that kindly provided to us by Professor Giuseppe Attardi @ UniPi.

The root of the project contains:

	-  the python script `run_cnn.py`, that implements the CNN and allow the user to choose between Cross-Validation and Test mode. See [**Scripts**](#sec:scripts) and [**Invocation**](#sec:invocation) for further details. 
	- the python script `run_bertft.py`, the same as `run_cnn.py` but implements both the fine-tuned BERT systems involved in our analysis. 
	- the Jupyter Notebook `Data_Cleaning.ipynb`, 
	- altri script?
	- the folder `cv_result`, output folder for our scripts in cv mode.
	- the folder `results_test`, output folder for our script in test mode.

### Requirements 
Code is written in Python (3.6.8) and requires Keras (2.2.4), Tensorflow (1.13.1) and [tweet-preprocessor](https://pypi.org/project/tweet-preprocessor/) (1.3.1).

Before running the script

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

### Running the models (CPU)
Example commands:

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 ./conv_net_sentence.py -vectors w2v_file mr.w2v mr.tsv
```

This will run the CNN classifier using the embeddings from file w2v_file and
the sentences in the TSV file mr.tsv and saving the model to file mr.w2v.

### Using the GPU
GPU will result in a good 10x to 20x speed-up, so it is highly recommended. 
To use the GPU, simply change `device=cpu` to `device=gpu` (or whichever gpu you are using).
For example:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec
```

### Example output
CPU output:
```
epoch: 1, training time: 219.72 secs, train perf: 81.79 %, val perf: 79.26 %
epoch: 2, training time: 219.55 secs, train perf: 82.64 %, val perf: 76.84 %
epoch: 3, training time: 219.54 secs, train perf: 92.06 %, val perf: 80.95 %
```
GPU output:
```
epoch: 1, training time: 16.49 secs, train perf: 81.80 %, val perf: 78.32 %
epoch: 2, training time: 16.12 secs, train perf: 82.53 %, val perf: 76.74 %
epoch: 3, training time: 16.16 secs, train perf: 91.87 %, val perf: 81.37 %
```

### Other Implementations
#### TensorFlow
[Denny Britz](http://www.wildml.com) has an implementation of the model in TensorFlow:

https://github.com/dennybritz/cnn-text-classification-tf

He also wrote a [nice tutorial](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow) on it, as well as a general tutorial on [CNNs for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp).

### Hyperparameters
Ye Zhang has written a [very nice paper](http://arxiv.org/abs/1510.03820) doing an extensive analysis of model variants (e.g. filter widths, k-max pooling, word2vec vs Glove, etc.) and their effect on performance.

### SemEval-2015 Task 10: Sentiment Analysis in Twitter

Experiments using the data from SemEval-15, using option --filters 7,7,7, achieves an accuracy
of 67.28%, better than the top official score (64.84%), as reported in:
Sara Rosenthal et al. [SemEval-2015 Task 10: Sentiment Analysis in Twitter](http://alt.qcri.org/semeval2015/cdrom/pdf/SemEval078.pdf).

