import operator
import os
import collections
import numpy as np
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import monotonically_increasing_id

from pyspark.sql import SparkSession

# Initialize spark session
spark = SparkSession.builder \
    .appName("comp5349-assignment-two-a2-s1") \
    .getOrCreate()

sc = spark.sparkContext

# CONFIG --- START ---

INPUT_TRAIN_DATA = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv'
INPUT_TEST_DATA = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv'

FILE_FLAG = '-test'
CWD = os.getcwd()
OUTPUT_DIR = '{}'.format(CWD)

K_NEAREST_NEIGHBORS = 5

# CONFIG --- END ---

# Data
train_data_labels = INPUT_TRAIN_DATA
test_data_labels = INPUT_TEST_DATA


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

from pyspark.ml.feature import PCA

pca = PCA(k=50, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(train_features)

# Apply PCA to train / test features
train_features_pca = pca_model.transform(train_features).select("pca_features")
test_features_pca = pca_model.transform(test_features).select("pca_features")

# Rename pca feature column values
train_features_pca = train_features_pca.withColumnRenamed(
    "pca_features", "train_features")
test_features_pca = test_features_pca.withColumnRenamed(
    "pca_features", "test_features")

# Rename pca feature column values
train_features_pca = train_features_pca.withColumnRenamed(
    "pca_features", "train_features")
test_features_pca = test_features_pca.withColumnRenamed(
    "pca_features", "test_features")


def combine_features_labels(feature_vector, label_vector, kind='train'):
    features = feature_vector.withColumn(
        '{}_id'.format(kind), monotonically_increasing_id())
    labels = label_vector.withColumn(
        '{}_id'.format(kind), monotonically_increasing_id())
    data = features.join(labels, '{}_id'.format(kind))
    return data


# Create combined train / test data
train_data = combine_features_labels(train_features_pca, train_labels, 'train')
test_data = combine_features_labels(test_features_pca, test_labels, 'test')

train_features = sc.broadcast(train_data.rdd.map(
    lambda x: x.train_features).collect())
train_labels = sc.broadcast(train_data.rdd.map(
    lambda x: x.train_labels).collect())

import numpy as np
import operator
import collections

import numpy as np
import operator
import collections

K = 10


def make_prediction(test_feature, train_feature, train_labels, K=3):
    similarity = ((test_feature - train_feature) ** 2).sum(axis=1) ** 5
    k_neighbors = train_labels[np.argpartition(similarity, K)[:K]]
    predicted_label = collections.Counter(
        k_neighbors.ravel()).most_common()[0][0]
    # return float(predicted_label)
    return Vectors.dense(predicted_label)


# Calculate compponents
labels = np.array(train_labels.value)
features = np.array(train_features.value)

# KNN Prediction
predictions = test_data.rdd.map(lambda x: (x.test_id, make_prediction(
    np.array(x.test_features), features, labels, K=5))).collect()


def prepare_data(actual_data, prediction_data, on='test_id'):
    return actual_data.join(prediction_data, on).rdd \
        .map(lambda x: (float(x.prediction[0]), float(x.test_labels[0])))


def overall_report(actual_data, prediction_data):
    # Calculate actual / predicted labels in rdd from
    prediction_and_labels = prepare_data(actual_data, prediction_data)

    # Calculate actual / predicted labels in rdd from
    metrics = MulticlassMetrics(prediction_and_labels)

    # Calculate overall level metrics
    # print('Precision:', metrics.precision(), type(metrics.precision()))
    return sc.parallelize([(Vectors.dense(metrics.precision()),
                            Vectors.dense(metrics.recall()),
                            Vectors.dense(metrics.fMeasure()))]).toDF(['Precision', 'Recall', 'F-Score'])


def classification_report(actual_data, prediction_data):
    # Calculate actual / predicted labels in rdd from
    prediction_and_labels = prepare_data(actual_data, prediction_data)

    # Calculate calculate class level metrics
    metrics = MulticlassMetrics(prediction_and_labels)
    classes = set(actual_data.rdd.map(lambda x: x.test_labels[0]).collect())
    results = [(Vectors.dense(float(c)),
                Vectors.dense(round(metrics.precision(c), 3)),
                Vectors.dense(round(metrics.recall(c), 3)),
                Vectors.dense(round(metrics.fMeasure(c), 3))) for c in sorted(classes)]
    return sc.parallelize(results).toDF(['Class', 'Precision', 'Recall', 'F-Score'])


# Create dataframe of prediction resutlts
prediction_data = sc.parallelize(
    predictions).toDF(['test_id', 'prediction'])

# Calculate overall summary statistics
overall_metrics = overall_report(test_data, prediction_data)
overall_metrics.show()

# # Calculate class level statistics
classification_metrics = classification_report(test_data, prediction_data)
classification_metrics.show()

# Save statistics to CSV
# overall_metrics.toPandas().to_csv(
#     '{0}/overall_metrics{1}.csv'.format(OUTPUT_DIR, FILE_FLAG), index=None)
# classification_metrics.toPandas().to_csv(
#     '{0}/classification_report{1}.csv'.format(OUTPUT_DIR, FILE_FLAG), index=None)

print('Complete')  # . Output saved to {}'.format(OUTPUT_DIR))
