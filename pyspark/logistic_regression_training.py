from __future__ import print_function

import time

from itertools import chain
from pyspark.context import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, \
                                  MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, \
                               RobustScaler, \
                               StringIndexer, \
                               VectorAssembler
from pyspark.ml.tuning import TrainValidationSplit, \
                              ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, \
                                  create_map, \
                                  lit, \
                                  udf
from pyspark.sql.types import FloatType


CATEGORICAL_FEATS = ['campaignId', 'platform', 'softwareVersion',
                     'sourceGameId', 'country', 'connectionType', 'deviceType']
NUMERICAL_FEATS = ['startCount', 'viewCount', 'clickCount', 'installCount',
                   'startCount1d', 'startCount7d', 'timeSinceLastStart']


def time_diff_in_minutes(dt_0, dt_1):
    if dt_0 is None:
        return 0.0
    return round((dt_1 - dt_0).total_seconds() / 60.0, 1)


if __name__ == "__main__":
    # spark = SparkSession.builder.appName('ads-ml').getOrCreate()
    sc = SparkContext()
    spark = SparkSession(sc)

    # Load data as spark data frame
    df = spark.read.options(delimiter=';') \
        .options(header=True) \
        .options(inferSchema=True) \
        .csv('gs://[bucket-name]/training_data.csv')

    time_diff_in_min_udf = udf(time_diff_in_minutes, FloatType())
    df = \
        df.withColumn('timeSinceLastStart',
                      time_diff_in_min_udf(
                          df.lastStart, df.timestamp)).drop("id",
                                                            "timestamp",
                                                            "lastStart")

    # Rename install column to label
    df = df.withColumnRenamed('install', 'label')

    # Determine Logistic Regression Weights From Class Imbalance
    class_count_df = df.groupby('label').agg({'label': 'count'})
    n_1 = class_count_df.filter(df.label == '1') \
                        .select("count(label)").collect()[0][0]
    n_0 = class_count_df.filter(df.label == '0') \
                        .select("count(label)").collect()[0][0]

    w_1 = (n_0 + n_1) / (2.0 * n_1)
    w_0 = (n_0 + n_1) / (2.0 * n_0)

    class_weights = {0: w_0, 1: w_1}

    mapping_expr = create_map([lit(x) for x in chain(*class_weights.items())])
    df = df.withColumn("weights", mapping_expr.getItem(col("label")))

    # Split training and test set
    train_df, test_df = df.randomSplit([0.8, 0.2])

    # Create training pipeline
    stages = []
    for column in CATEGORICAL_FEATS:
        str_indexer = StringIndexer(inputCol=column,
                                    outputCol=column + "Index",
                                    handleInvalid='keep')
        encoder = OneHotEncoder(inputCols=[str_indexer.getOutputCol()],
                                outputCols=[column + "Vec"],
                                handleInvalid='keep')
        stages += [str_indexer, encoder]

    assembler1 = VectorAssembler(inputCols=NUMERICAL_FEATS,
                                 outputCol="num_features")
    scaler = RobustScaler(inputCol='num_features',
                          outputCol='scaled')
    stages += [assembler1, scaler]

    assembler_inputs = [c + "Vec" for c in CATEGORICAL_FEATS] + ["scaled"]
    assembler2 = VectorAssembler(inputCols=assembler_inputs,
                                 outputCol="features")
    stages += [assembler2]

    lr = LogisticRegression(weightCol='weights')
    stages.append(lr)

    pipeline = Pipeline(stages=stages)

    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [1.0, 0.1, 0.01, 0.001]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    # Hyperparameter tuning
    t_0 = time.time()
    train_val = TrainValidationSplit(estimator=pipeline,
                                     estimatorParamMaps=param_grid,
                                     evaluator=BinaryClassificationEvaluator(
                                         metricName='areaUnderPR'),
                                     trainRatio=0.8)
    model = train_val.fit(train_df)

    print(model.bestModel.stages[-1].explainParam('regParam'))
    print(model.bestModel.stages[-1].explainParam('elasticNetParam'))
    print('Grid search took: {} seconds'.format(time.time() - t_0))

    # Model Metrics
    t_0 = time.time()
    predictions = model.transform(test_df)
    print('Model training took: {} seconds'.format(time.time() - t_0))

    evaluator = BinaryClassificationEvaluator()

    auroc = evaluator.evaluate(predictions,
                               {evaluator.metricName: "areaUnderROC"})
    auprc = evaluator.evaluate(predictions,
                               {evaluator.metricName: "areaUnderPR"})
    print("Area under ROC Curve: {:.4f}".format(auroc))
    print("Area under PR Curve: {:.4f}".format(auprc))

    evaluator = MulticlassClassificationEvaluator()
    rec = evaluator.evaluate(predictions,
                             {evaluator.metricName: "recallByLabel",
                              evaluator.metricLabel: 1})
    prec = evaluator.evaluate(predictions,
                              {evaluator.metricName: "precisionByLabel",
                               evaluator.metricLabel: 1})
    print(prec)
    print(rec)




