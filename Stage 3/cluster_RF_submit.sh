#!/bin/bash

spark-submit \
    --master yarn \
    --num-executors 8 \
    Random_Forest_Classifier.py
    
