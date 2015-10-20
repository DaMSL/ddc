#!/bin/bash

python3 src/reset.py
python3 src/piEstimator.py -i
rm -f redis.lock
rm -f out/*
rm -f pi/*
rm -f sh/*
pkill redis-server
