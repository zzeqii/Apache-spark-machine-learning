#!/bin/bash

spark-submit \
    --master yarn \
    --num-executors  \
    stage-one-2.py
