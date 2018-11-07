#!/bin/bash

spark-submit \
    --master yarn \
    --num-executors 8 \
    Logistic_Regression.py
    
