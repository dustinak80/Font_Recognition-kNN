# Font_Recognition-kNN

This kNN algorithm was made for predicting three types of font: Calibri, Courier, and Times. The data set was used from UC Irvine 
repository and was reduced to those three fonts with 

1) Bring in the data. (kNN.py)
2) Run PCA for feature reduction. (kNN.py)
- correlation matrix
- eigenvalues from correlation matrix
- ratio of variance calculated from eigenvalues. Reduced features from 400 to 79 while keeping 90% of explained variance.
- project data into new dimension
3) Test/Train split - manually (kNN.py)
4) kNN algorithm for training (kNN.py)
5) kNN algorithm for predicting (kNN.py)
6) Confusion matrix and analysis of predictions (kNN.py)
7 Report on algorithm design and use (kNN.pdf)
