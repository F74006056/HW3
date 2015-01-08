import sys
from random import random
from operator import add
from pyspark import SparkContext

sc = SparkContext('local', 'test')
file = sc.textFile("pg5000.txt")
counts = file.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1))
result = counts.map(lambda (word, number) : (number, word)).collect()
print len(result)-1
