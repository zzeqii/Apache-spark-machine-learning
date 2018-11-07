
# coding: utf-8



# import findspark
# findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName("Python Spark LR Classifier")     .getOrCreate()

sc=spark.sparkContext




from pyspark.ml.classification import LogisticRegression



#load and parse the data file,converitn it to a DataFrame
path='hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/'
#path='hdfs://localhost:9000/user/jessicaxu/'
training_data=spark.read.csv(path+'Train-label-28x28.csv', header=False, inferSchema="true").withColumnRenamed('_c0','label')
testing_data=spark.read.csv(path+'Test-label-28x28.csv',header=False, inferSchema="true").withColumnRenamed('_c0','label')



from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id

#prepare training Data
assembler=VectorAssembler(inputCols=training_data.columns[1:],outputCol='features')
newdata=assembler.transform(training_data)
train_data=newdata.select('label','features')
train_id = train_data.withColumn(
        '{}_id'.format(train_data), monotonically_increasing_id())
training=train_id.withColumnRenamed('DataFrame[label: int, features: vector]_id','ID')
training.show(5)

#prepare testing Data
assembler_test=VectorAssembler(inputCols=testing_data.columns[1:],outputCol='features')
newdata_test=assembler_test.transform(testing_data)
test_data=newdata_test.select('label','features')
test_id = test_data.withColumn(
        '{}_id'.format(test_data), monotonically_increasing_id())
testing=test_id.withColumnRenamed('DataFrame[label: int, features: vector]_id','ID')
testing.show(5)


#apply PCA
from pyspark.ml.feature import PCA
pca = PCA(k=50, inputCol="features", outputCol="feature")
pca_train = pca.fit(training)
#Apply PCA to train / test features
train_pca = pca_train.transform(training).select("label","feature","ID")
test_pca = pca_train.transform(testing).select("label","feature","ID")
train_pca.show(5)


#define the LR model
lr = LogisticRegression(featuresCol="feature",family="multinomial")
# Fit the model
lrModel = lr.fit(train_pca)
#train the model
prediction=lrModel.transform(test_pca)
#select columns
prediction.select('label','feature','ID','probability','prediction').show(1)
#select columns
prediction.select('label','feature','ID','probability','prediction').show(1)



#Select (prediction, true label) and compute test error
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(prediction)
print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))
# Print the coefficients and intercept for multinomial logistic regression
print("Coefficients: \n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))




from pyspark.mllib.evaluation import MulticlassMetrics
def prepare_data(actual_data, prediction_data, on='ID'):
    return actual_data.join(prediction_data, on).rdd.map(lambda x: (float(x.prediction), float(x.label)))


def overall_report(actual_data, prediction_data):
    # Calculate actual / predicted labels in rdd from
    prediction_and_labels = prepare_data(actual_data, prediction_data)
    # Calculate actual / predicted labels in rdd from
    metrics = MulticlassMetrics(prediction_and_labels)
    # Calculate overall level metrics
    print('Accuracy\tPrecision\tRecall\tF-Score')
    print('{}\t{}\t{}\t{}'.format(metrics.accuracy, metrics.precision(),
                                  metrics.recall(), metrics.fMeasure()))


def classification_report(actual_data, prediction_data):
    # Calculate actual / predicted labels in rdd from
    prediction_and_labels = prepare_data(actual_data, prediction_data)
    # Calculate calculate class level metrics
    metrics = MulticlassMetrics(prediction_and_labels)
    classes = set(actual_data.rdd.map(lambda x: float(x.label)).collect())
    print('Class\tPrecision\tRecall\tF-Score')
    for c in sorted(classes):
        print('{}\t{}\t{}\t{}'.format(c,
                                      round(metrics.precision(c), 3),
                                      round(metrics.recall(c), 3),
                                      round(metrics.fMeasure(c), 3)))





#print out precision, recall, f1-score
overall_report(test_pca, prediction)
# print out precision, recall, f1-score for each class
classification_report(test_pca, prediction)

