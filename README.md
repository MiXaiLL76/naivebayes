# Description
Sklearn Naive Bayes interpretation for GO.
The project was implemented as part of a service that works with GaussianNB.

# Install
```
go get github.com/MiXaiLL76/naivebayes
```

# Example  
**TRAIN** in Python [train.ipynb](examples/train.ipynb)  
**TEST**  in Golang [examples/main.go](examples/main.go)  

# Implementing Naive Bayes Classes

## GaussianNB [sklearn](https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes)

| Method | Description | Status |
|--------|-------------|--------|
|**fit**(X, y[, sample_weight])|Fit Gaussian Naive Bayes according to X, y| **W.I.P.** |
|**get_weight**() | Get weight for this estimator. | **✓ DONE**|
|**set_weight**(weight) | Set the weight of this estimator.| **✓ DONE**|
|**predict**(X) | Perform classification on an array of test vectors X. | **✓ DONE**|
|**predict_log_proba**(X)|Return log-probability estimates for the test vector X.|**✓ DONE**|
|**predict_proba**(X)|Return probability estimates for the test vector X.|**✓ DONE**|
|**score**(X, y[, sample_weight])|Return the mean accuracy on the given test data and labels.| **✓ DONE** |  
  

## Utility

| Method | Description | Status |
|--------|-------------|--------|
|**argmax**(array []float64)|Returns the indices of the maximum values| **✓ DONE**|
|**logsumexp**(array []float64)|Compute the log of the sum of exponentials of input elements.| **✓ DONE**|
|**getShape**(array [][]float64)|Return the shape of an array.| **✓ DONE**|
|**AccuracyScore**(y_true, y_pred)|Accuracy classification score.| **✓ DONE**|
