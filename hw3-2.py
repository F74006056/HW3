import sys
from random import random
from operator import add
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from numpy import array

sc = SparkContext('local', 'test')

# Load and parse the data
def parseRating(line):
	field = line.strip().split("::")
	return int(field[0]), int (field[1]), int (field[2])

def parseMovie(line):
	field = line.strip().split("::")
	return int(field[0]), field[1]

rating = sc.textFile("ratings.dat").map(parseRating)
movie = dict(sc.textFile("movies.dat").map(parseMovie).collect())


# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 20
lmbda = 0.1
model = ALS.train(rating, rank, numIterations, lmbda)

# Evaluate the model on training data
testdata = rating.map(lambda p: (int(p[0]), int(p[1])))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2])).collect()
rec = sorted(predictions, key=lambda x : x[1], reverse=True)
count = 0
checkList = [0]
while count < 10:
	tmp = rec.pop(0)
	temp = tmp[0][1]
	if temp not in checkList:
		print ('%d:%s') % (count+1, movie[temp])
		checkList.append(temp)
		count = count + 1

