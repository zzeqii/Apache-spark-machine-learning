# coding: utf-8

import operator
import os
import collections
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
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

FILE_FLAG = '-cross-test'
CWD = os.getcwd()
OUTPUT_DIR = '{}'.format(CWD)

K_NEAREST_NEIGHBORS = 5

# CONFIG --- END ---


def get_vector(data, col_name):
    assembler = VectorAssembler(inputCols=data.columns, outputCol=col_name)
    return assembler.transform(data).select(col_name)


def combine_features_labels(feature_vector, label_vector, kind='train'):
    features = feature_vector.withColumn(
        '{}_id'.format(kind), monotonically_increasing_id())
    labels = label_vector.withColumn(
        '{}_id'.format(kind), monotonically_increasing_id())
    data = features.join(labels, '{}_id'.format(kind))
    return data


def get_preprocessed_data(input_train, input_test):
    # Train Data
    train = spark.read.csv(input_train, header=False, inferSchema="true")
    train_labels = get_vector(train.select('_c0'), 'train_label')
    train_features = get_vector(train.drop('_c0'), 'feature')

    # Test Data
    test = spark.read.csv(input_test, header=False, inferSchema="true")
    test_labels = get_vector(test.select('_c0'), 'test_label')
    test_features = get_vector(test.drop('_c0'), 'feature')

    # Compute PCA
    pca = PCA(k=50, inputCol="feature", outputCol="pca_feature")
    pca_model = pca.fit(train_features)

    # Apply PCA to train / test features
    train_features_pca = pca_model.transform(
        train_features).select("pca_feature")
    test_features_pca = pca_model.transform(
        test_features).select("pca_feature")

    # Rename pca feature column values
    train_features_pca = train_features_pca.withColumnRenamed(
        "pca_feature", "train_feature")
    test_features_pca = test_features_pca.withColumnRenamed(
        "pca_feature", "test_feature")

    # Create combined train / test data
    train_data = combine_features_labels(
        train_features_pca, train_labels, 'train')
    test_data = combine_features_labels(
        test_features_pca, test_labels, 'test')

    return train_data, test_data


def make_prediction(train_data, test_data, K=3):
    # Find all cartesian pairs of train and test data
    cross = test_data.crossJoin(train_data)

    # Define UDF to calculate euclidean distance
    distance_udf = F.udf(lambda x, y: float(
        (np.array(x - y) ** 2).sum() ** 0.5), FloatType())

    # Calculate similarity with euclidean distance
    similarity = cross.withColumn(
        'distance', distance_udf(F.col('test_feature'), F.col('train_feature')))

    # Define window to sort distance for each test instance
    window = Window.partitionBy(similarity.test_id).orderBy(
        similarity.distance.asc())

    # Calculate K nearest neighbors
    k_neighbors = similarity.select(
        '*', rank().over(window).alias('rank')).filter(col('rank') <= K)

    # Define function to perform majority vote
    def majority_vote(neighbors):
        return Vectors.dense(collections.Counter([x[1] for x in neighbors]).most_common()[0][0])

    # Compute predicted label by majority vote
    predicted_labels = k_neighbors.rdd \
        .map(lambda x: (x.test_id, (x.distance, x.train_label[0]))) \
        .groupByKey() \
        .map(lambda x: (x[0], majority_vote(list(x[1])))).collect()

    # Return result
    return predicted_labels


# def prepare_data(actual_data, prediction_data, on='test_id'):
#     return actual_data.join(prediction_data, on).rdd \
#         .map(lambda x: (float(x.prediction[0]), float(x.test_label[0])))


# def overall_report(actual_data, prediction_data):
#     # Calculate actual / predicted labels in rdd from
#     prediction_and_labels = prepare_data(actual_data, prediction_data)

#     # Calculate actual / predicted labels in rdd from
#     metrics = MulticlassMetrics(prediction_and_labels)

#     # Calculate overall level metrics
#     return sc.parallelize([(Vectors.dense(round(metrics.precision(), 3)),
#                             Vectors.dense(round(metrics.recall(), 3)),
#                             Vectors.dense(round(metrics.fMeasure(), 3)))]).toDF(['Precision', 'Recall', 'F-Score'])


# def classification_report(actual_data, prediction_data):
#     # Calculate actual / predicted labels in rdd from
#     prediction_and_labels = prepare_data(actual_data, prediction_data)

#     # Calculate calculate class level metrics
#     metrics = MulticlassMetrics(prediction_and_labels)
#     classes = set(actual_data.rdd.map(lambda x: x.test_label[0]).collect())
#     results = [(Vectors.dense(float(c)),
#                 Vectors.dense(round(metrics.precision(c), 3)),
#                 Vectors.dense(round(metrics.recall(c), 3)),
#                 Vectors.dense(round(metrics.fMeasure(c), 3))) for c in sorted(classes)]
#     return sc.parallelize(results).toDF(['Class', 'Precision', 'Recall', 'F-Score'])


train_data, test_data = get_preprocessed_data(
    INPUT_TRAIN_DATA, INPUT_TEST_DATA)

# Compute KNN predictions
predictions = make_prediction(train_data, test_data, K=K_NEAREST_NEIGHBORS)

# Create dataframe of prediction resutlts
prediction_data = sc.parallelize(
    predictions).toDF(['test_id', 'prediction'])
prediction_data.show(5)
# Calculate overall summary statistics
# overall_metrics = overall_report(test_data, prediction_data)
# overall_metrics.show()

# # Calculate class level statistics
# classification_metrics = classification_report(test_data, prediction_data)
# classification_metrics.show()

# Save statistics to CSV
# overall_metrics.toPandas().to_csv(
#     '{0}/overall_metrics{1}.csv'.format(OUTPUT_DIR, FILE_FLAG), index=None)
# classification_metrics.toPandas().to_csv(
#     '{0}/classification_report{1}.csv'.format(OUTPUT_DIR, FILE_FLAG), index=None)

print('Complete.')  # Output saved to {}'.format(OUTPUT_DIR))
