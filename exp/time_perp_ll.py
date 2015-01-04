import sys

"""
Purpose: Extract time, perplexity, and loglikelihood from the log file
Input: Log file generated from `../gibbs`
Output: (stdout) Elapsed time, perplexity, and loglikelihood of each iter
"""

fin = open(sys.argv[1], 'r')

print 'Time', '\t', 'Perplexity', '\t', 'Loglikelihood'
elapsed_time = 0
perp = 0
for line in fin:
  if 'Iteration' in line:
    tok = line.split()
    iter_time = float(tok[-2])
    elapsed_time += iter_time
  if 'Perplexity' in line:
    tok = line.split()
    perp = float(tok[-1])
  if 'loglikelihood' in line:
    tok = line.split()
    ll = float(tok[-1])
    print '%5.2f\t%10.2f\t%e' % (elapsed_time, perp, ll)
