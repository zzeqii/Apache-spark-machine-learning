# Spark Machine Learning Application

## Group members

- Nicholas Moore 
- Ziqi Xu 
- Ang Li 

## Dataset

The task contains three stages. All stages use the MNIST data set of handwritten
digits(http://yann.lecun.com/exdb/mnist/). The original data set contains four
files to store training image, training label, test image and test label. The files have been converted to csv format and are stored in School of IT Hadoop cluster under /share/MNIST(http://soit-hdp-pro-1.ucc.usyd.edu.au:50070/explorer.html/share/MNIST )

## Stage One: KNN Classifier

Developing a KNN classifier from the scratch to classify 10,000 test images.

## Stage Two: Performance 

Running KNN classifier with different combinations of hyperparameters and execution properties

- reduced dimensions d: 50-100
- neareaset neighbour k: 5-10
- num-executors : 1,4,8
- executor-cores : 1,2
- total combinations: 12

## Stage Three: Spark Classifier Exploration

- logistic regression
- random forest 
