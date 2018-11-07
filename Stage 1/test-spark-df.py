import operator
import os
import collections
import numpy as np
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType

from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.functions import row_number

from pyspark.sql import SparkSession

# Initialize spark session
spark = SparkSession.builder \
    .appName("comp5349-assignment-two-a2-s1") \
    .getOrCreate()

sc = spark.sparkContext

# CONFIG --- START ---

INPUT_TRAIN_DATA = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv'
INPUT_TEST_DATA = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv'

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


def prepare_data(actual_data, prediction_data, on='test_id'):
    return actual_data.join(prediction_data, on).rdd \
        .map(lambda x: (float(x.prediction[0]), float(x.test_label[0])))


def overall_report(actual_data, prediction_data):
    # Calculate actual / predicted labels in rdd from
    prediction_and_labels = prepare_data(actual_data, prediction_data)

    # Calculate actual / predicted labels in rdd from
    metrics = MulticlassMetrics(prediction_and_labels)

    # Calculate overall level metrics
    # print('Precision:', metrics.precision(), type(metrics.precision()))
    return sc.parallelize([(Vectors.dense(metrics.accuracy),
                            Vectors.dense(metrics.precision()),
                            Vectors.dense(metrics.recall()),
                            Vectors.dense(metrics.fMeasure()))]).toDF(['Accuracy', 'Precision', 'Recall', 'F - Score'])


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


def euclidean_distance(x, y):
    return (np.array(x - y) ** 2).sum() ** 0.5


# Train Data
train = spark.read.csv(INPUT_TRAIN_DATA, header=False, inferSchema="true")
train_labels = get_vector(train.select('_c0'), 'train_label')
train_features = get_vector(train.drop('_c0'), 'feature')

# Test Data
test = spark.read.csv(INPUT_TEST_DATA, header=False, inferSchema="true")
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

# Generate cross-join
cross = test_data.crossJoin(train_data)

# Find similarity of each pair
distance_udf = F.udf(lambda x, y: float(euclidean_distance(x, y)), FloatType())
similarity = cross.withColumn('distance', distance_udf(F.col('test_feature'), F.col(
    'train_feature'))).select(['test_id', 'train_label', 'distance'])

# Rank similarityof pairs to find k neighbors
window = Window.partitionBy(
    similarity['test_id']).orderBy(similarity['distance'].asc())
k_neighbors = similarity.select(
    '*', rank().over(window).alias('rank')).filter(col('rank') <= K_NEAREST_NEIGHBORS)

# Perform majority vote by counting test_id and train_label pairs
grouped = k_neighbors.groupBy('test_id', 'train_label').count()
window = Window.partitionBy("test_id").orderBy('count')
prediction_data = grouped.withColumn('order', row_number().over(window)) \
    .where(col('order') == 1) \
    .select(['test_id', 'train_label']) \
    .sort('test_id')

# Show prediced data
prediction_data.head(10)

# Calculate overall summary statistics
# overall_metrics = overall_report(test_data, prediction_data)
# overall_metrics.show()

# # Calculate class level statistics
# classification_metrics = classification_report(test_data, prediction_data)
# classification_metrics.show()

print('Complete')
