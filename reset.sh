#!/bin/bash

python3 reset.py
python3 piEstimator.py -i
rm -f redis.lock
rm -f out/*
rm -f pi/*
rm job0*
