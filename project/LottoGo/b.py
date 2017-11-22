# -*- coding: utf-8 -*- 
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import date
import csv
f = open('data.csv', 'r')
rdr = csv.reader(f)

date1 = []
year = []
month = []
day = []

firstDate = date(2002,12,7)
firstganzi = [8,8,5]
firstzizi = [6,0,9]

manseTemp =[]
manse = []
ganzi = ["갑","을","병","정","무","기","경","신","임","계"]
ganziNumber = [3,8,7,2,5,10,9,4,1,6]
zizi = ["자","축","인","묘","진","사","오","미","신","유","술","해"]
ziziNumber = [1,10,3,8,5,2,7,10,9,4,5,6]

# xDataTemp = []
# xData = []
xData = [[] for row in range(1000)]

# numberTemp =[]
# number = []
y_data=[]
number = [[] for row in range(1000)]
index=0;
# print "hello"
for line in rdr:
	date1.append(date(int(line[0].split('.')[0]),int(line[0].split('.')[1]),int(line[0].split('.')[2])))
	
	year.append(int(line[0].split('.')[0]) - 2002);
	month.append(int(line[0].split('.')[1]) + year[index]*12 - 12);
	gap = date1[index] - firstDate 
	day.append(gap.days)

	manseTemp = [(firstganzi[0] + year[index])%10,
				(firstzizi[0] + year[index])%12,
				(firstganzi[1] + month[index])%10,
				(firstzizi[1] + month[index])%12,
				(firstganzi[2] + day[index])%10,
				(firstzizi[2] + day[index])%12]
	manse.append(manseTemp)

# 	xDataTemp = [ganziNumber[(firstganzi[0] + year[index])%10],
# 				ziziNumber[(firstzizi[0] + year[index])%12],
# 				ganziNumber[(firstganzi[1] + month[index])%10],
# 				ziziNumber[(firstzizi[1] + month[index])%12],
# 				ganziNumber[(firstganzi[2] + day[index])%10],
# 				ziziNumber[(firstzizi[2] + day[index])%12]]
# 	xData.append(xDataTemp)
    
	xData[0].append(ganziNumber[(firstganzi[0] + year[index])%10])
	xData[1].append(ziziNumber[(firstzizi[0] + year[index])%12])
	xData[2].append(ganziNumber[(firstganzi[1] + month[index])%10])
	xData[3].append(ziziNumber[(firstzizi[1] + month[index])%12])
	xData[4].append(ganziNumber[(firstganzi[2] + day[index])%10])
	xData[5].append(ziziNumber[(firstzizi[2] + day[index])%12])

	number[0].append(int(line[1]))
	number[1].append(int(line[2]))
	number[2].append(int(line[3]))
	number[3].append(int(line[4]))
	number[4].append(int(line[5]))
	number[5].append(int(line[6]))

    
	index = index+1

index = 0
# for item in date1:
# 	print (ganzi[manse[index][0]] + zizi[manse[index][1]] + ganzi[manse[index][2]] + zizi[manse[index][3]]+ ganzi[manse[index][4]] + zizi[manse[index][5]])
# 	index = index + 1

# print xData[0]
f.close()   

W1 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
W2 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
W3 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
W4 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
W5 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
W6 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

#hypothesis
hypothesis = W1 * xData[0] + W2 * xData[1]+ W3 * xData[2]+ W4 * xData[3]+ W5 * xData[4]+ W6 * xData[5] + b


cost = tf.reduce_mean(tf.square(hypothesis - number[0]))

#minimize
a = tf.Variable(0.1) #alpha, learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)


# befor starting, initialize variables
init = tf.initialize_all_variables()

#launch
sess = tf.Session()
sess.run(init)


# fit the line
for step in range(40):
    sess.run(train)
    print (step, sess.run(cost), sess.run(W1), sess.run(W2),  sess.run(W3),  sess.run(W4),  sess.run(W5), sess.run(W6), sess.run(b) )


plt.plot(cost, 'ro', label='Original data')
plt.legend()
plt.show()