#!/bin/bash
set -ux

../sparselda \
  -train_file nytimes.train \
  -test_file nytimes.test \
  -num_iter 100000 \
  -num_topic 1000

