# coding: utf-8

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
    test_data = combine_features_labels(test_features_pca, test_labels, 'test')

    return train_data, test_data


def make_prediction(test_feature, train_features, train_lables, K=3):
    # Calulate components
    train_features = np.array(train_features.value)
    train_lables = np.array(train_lables.value)
    test_feature = np.array(test_feature)

    # Calculate similarity with euclidean distance
    similarity = ((test_feature - train_features) ** 2).sum(axis=1) ** 0.5

    # Calculate K nearest neighbors
    k_neighbors = train_lables[np.argpartition(similarity, K)[:K]]

    # Compute predicted label by majority vote
    predicted_label = collections.Counter(
        k_neighbors.ravel()).most_common()[0][0]

    # Return result
    return Vectors.dense(predicted_label)


def prepare_data(actual_data, prediction_data, on='test_id'):
    return actual_data.join(prediction_data, on).rdd \
        .map(lambda x: (float(x.prediction[0]), float(x.test_label[0])))


def overall_report(actual_data, prediction_data):
    # Calculate actual / predicted labels in rdd from
    prediction_and_labels = prepare_data(actual_data, prediction_data)

    # Calculate actual / predicted labels in rdd from
    metrics = MulticlassMetrics(prediction_and_labels)

    # Calculate overall level metrics
    return sc.parallelize([(Vectors.dense(round(metrics.precision(), 3)),
                            Vectors.dense(round(metrics.recall(), 3)),
                            Vectors.dense(round(metrics.fMeasure(), 3)))]).toDF(['Precision', 'Recall', 'F-Score'])


def classification_report(actual_data, prediction_data):
    # Calculate actual / predicted labels in rdd from
    prediction_and_labels = prepare_data(actual_data, prediction_data)

    # Calculate calculate class level metrics
    metrics = MulticlassMetrics(prediction_and_labels)
    classes = set(actual_data.rdd.map(lambda x: x.test_label[0]).collect())
    results = [(Vectors.dense(float(c)),
                Vectors.dense(round(metrics.precision(c), 3)),
                Vectors.dense(round(metrics.recall(c), 3)),
                Vectors.dense(round(metrics.fMeasure(c), 3))) for c in sorted(classes)]
    return sc.parallelize(results).toDF(['Class', 'Precision', 'Recall', 'F-Score'])


# def main():
train_data, test_data = get_preprocessed_data(
    INPUT_TRAIN_DATA, INPUT_TEST_DATA)

# Broadcast training features and examples
train_features = sc.broadcast(train_data.rdd.map(
    lambda x: x.train_feature).collect())
train_labels = sc.broadcast(train_data.rdd.map(
    lambda x: x.train_label).collect())

# Compute KNN predictions
predictions = test_data.rdd.map(
    lambda x: (x.test_id, make_prediction(
        x.test_feature, train_features, train_labels, K=K_NEAREST_NEIGHBORS))).collect()

# Create dataframe of prediction resutlts
prediction_data = sc.parallelize(
    predictions).toDF(['test_id', 'prediction'])

# Calculate overall summary statistics
overall_metrics = overall_report(test_data, prediction_data)
overall_metrics.show()

# Calculate class level statistics
classification_metrics = classification_report(test_data, prediction_data)
classification_metrics.show()

# Save statistics to CSV
overall_metrics.toPandas().to_csv(
    '{0}/overall_metrics{1}.csv'.format(OUTPUT_DIR, FILE_FLAG), index=None)
classification_metrics.toPandas().to_csv(
    '{0}/classification_report{1}.csv'.format(OUTPUT_DIR, FILE_FLAG), index=None)

print('Complete. Output saved to {}'.format(OUTPUT_DIR))


# if __name__ == '__main__':
#     main()
