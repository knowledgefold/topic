import sys
import random

if len(sys.argv) != 5:
  print '%s input test_ratio output.train output.test' % sys.argv[0]
  exit()

ratio = float(sys.argv[2]) # e.g. 10% of the entire dataset

lines = open(sys.argv[1], 'r').readlines()
n = len(lines)
print 'total = %d' % n

num_test = int(n * ratio)
num_train = n - num_test
random.shuffle(lines)
train_fp = open(sys.argv[3], 'w')
for line in lines[:num_train]:
  train_fp.write(line)
train_fp.close()
test_fp = open(sys.argv[4], 'w')
for line in lines[num_train:]:
  test_fp.write(line)
test_fp.close()

print 'num train = %d, num test = %d' % (num_train, num_test)
