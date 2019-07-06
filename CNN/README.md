# Usage:
python Run_cnn.py

# Flags:
 - b = batch_size
 - e = epochs
 - k = kernel_size
 - n = number of filters
 - x = embeddings
 - a = activation function
 - d = dropout
 - m = mode (cv/test)

# Example:
python Run_cnn.py k5,5 n100,100 mtest

# Files:
 - Train data: data/CNN/train/train_{data,labels}
 - Test data: data/CNN/test/test_{data,labels}{2013,2013sms,2014,2014sar,2014livej,2015,2016,2017}

# Mode:
This script has two mode Cross-Validation (mcv) and Test (mtest).

In cross-validation mode the models performs a 5 cv for 3 repetitions.
In test mode the modoles performs a Train, Scores(Train) and Scores(Tests).

# Result:
All the results are stored in the folder resultsCNN.