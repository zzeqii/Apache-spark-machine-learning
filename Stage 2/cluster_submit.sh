#!/bin/bash

spark-submit \
    --master yarn \
    --num-executors  \
    Spark_KNN_Clouster.py
    
