
# coding: utf-8

# In[1]:


# Set up spark context

# import findspark
# findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName("Python Spark KNN Test")     .getOrCreate()


# In[2]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id 
from math import sqrt
from pyspark.sql.functions import sqrt

# Data 
train_data_labels = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv'
test_data_labels = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv'

def get_vector(data, col_name):
    assembler = VectorAssembler(inputCols=data.columns, outputCol=col_name)
    return assembler.transform(data).select(col_name)    

# Train Data
train = spark.read.csv(train_data_labels, header=False, inferSchema="true")
train_labels = get_vector(train.select('_c0'), 'train_labels')
train_features = get_vector(train.drop('_c0'), 'features')

# Test Data
test = spark.read.csv(test_data_labels, header=False, inferSchema="true")
test_labels = get_vector(test.select('_c0'), 'test_labels')
test_features = get_vector(test.drop('_c0'), 'features')


# In[3]:


from pyspark.ml.feature import PCA

pca = PCA(k=50, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(train_features)

# Apply PCA to train / test features
train_features_pca = pca_model.transform(train_features).select("pca_features")
test_features_pca = pca_model.transform(test_features).select("pca_features")


# In[4]:


# Rename pca feature column values
train_features_pca = train_features_pca.withColumnRenamed("pca_features", "train_features")
test_features_pca = test_features_pca.withColumnRenamed("pca_features", "test_features")


# In[5]:


# Develop a combined dataframe for all data

def combine_features_labels(feature_vector, label_vector, kind='train'):
    features = feature_vector.withColumn('{}_id'.format(kind), monotonically_increasing_id())
    labels = label_vector.withColumn('{}_id'.format(kind), monotonically_increasing_id())
    data = features.join(labels, '{}_id'.format(kind))
    return data

# Create combined train / test data
train_data = combine_features_labels(train_features_pca, train_labels, 'train')
test_data = combine_features_labels(test_features_pca, test_labels, 'test')


# In[6]:


train_features = sc.broadcast(train_data.rdd.map(lambda x: x.train_features).collect())
train_labels = sc.broadcast(train_data.rdd.map(lambda x: x.train_labels).collect())


# In[7]:


import numpy as np
import operator
import collections

K = 10

def make_prediction(test_feature, train_feature, train_labels, K = 3):
    similarity = ((test_feature - train_feature) ** 2).sum(axis = 1) ** 5
    k_neighbors = train_labels[np.argpartition(similarity, K)[:K]]
    predicted_label = collections.Counter(k_neighbors.ravel()).most_common()[0][0]
    return float(predicted_label)

#Calculate compponents
labels = np.array(train_labels.value)
features = np.array(train_features.value)

#KNN Prediction
predictions = test_data.rdd.map(lambda x: (x.test_id, make_prediction(np.array(x.test_features), features, labels, K = 5))).collect()

# Create dataframe of prediction resutlts
prediction_data = sc.parallelize(predictions).toDF(['test_id', 'prediction'])


# In[8]:


#prediction_data.show(3)


# In[9]:


# Input dataframes to calculate summary statistics
test_data.show(3), prediction_data.show(3)


# In[10]:


from pyspark.mllib.evaluation import MulticlassMetrics

# Calculation of statistics

def summary_statistics(metrics):
    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print('Precision\tRecall\tF-Score')
    print('{}\t{}\t{}'.format(metrics.precision(), metrics.recall(), metrics.fMeasure()))
    

def label_statistics(metrics, labels):
    print('Class\tPrecision\tRecall\tF-Score')
    for label in sorted(labels):
        print('{}\t{}\t{}\t{}'.format(label, 
                                      round(metrics.precision(label), 3), 
                                      round(metrics.recall(label), 3), 
                                      round(metrics.fMeasure(label), 3)))

        
def statistics(test_data, prediction_data):
    # Compute raw scores on the test set
    prediction_and_labels = test_data.join(prediction_data, 'test_id').rdd                             .map(lambda x: (float(x.prediction[0]), float(x.test_labels[0])))

    # Instantiate metrics object
    metrics = MulticlassMetrics(prediction_and_labels)

    # Overall statistics
    print("Summary Statistics\n")
    summary_statistics(metrics)

    # Statistics by class
    print("\nClass Summary Statistics\n")
    label_statistics(metrics, labels)
      

statistics(test_data, prediction_data)

