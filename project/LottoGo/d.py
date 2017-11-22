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

xDataTemp = []
xData = []
# xData = [[] for row in range(1000)]

numberTemp =[]
number = []
number = [[] for row in range(1000)]
index=0;
# print "hello"
def mansedate(input):
	date1=date(int(input.split('.')[0]),int(input.split('.')[1]),int(input.split('.')[2]))
	year=int(input.split('.')[0]) - 2002;
	month=int(input.split('.')[1]) + year*12 - 12;
	gap = date1 - firstDate 
	day=gap.days;

	manseTemp = [(firstganzi[0] + year)%10,
				(firstzizi[0] + year)%12,
				(firstganzi[1] + month)%10,
				(firstzizi[1] + month)%12,
				(firstganzi[2] + day)%10,
				(firstzizi[2] + day)%12]
	manse.append(manseTemp)

	xDataTemp = [ganziNumber[(firstganzi[0] + year)%10],
				ziziNumber[(firstzizi[0] + year)%12],
				ganziNumber[(firstganzi[1] + month)%10],
				ziziNumber[(firstzizi[1] + month)%12],
				ganziNumber[(firstganzi[2] + day)%10],
				ziziNumber[(firstzizi[2] + day)%12]]
	return xDataTemp

for line in rdr:
	xData.append(mansedate(line[0]))
    
	for j in range(6):
		numberTemp = []
		for k in range(45):
			numberTemp.insert(k,0)
			if(int(line[j+1])-1==k):
				numberTemp.insert(k,1)
		number[j].append(numberTemp);
	index = index+1
index = 0
f.close()   



#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# W = tf.Variable(tf.random_uniform([6, 46], -1., 1.))
# b = tf.Variable(tf.zeros([46]))
# L = tf.add(tf.matmul(X, W), b)
# L = tf.nn.relu(L)
# model = tf.nn.softmax(L)


# 첫번째 가중치의 차원은 [특성, 히든 레이어의 뉴런갯수] -> [2, 10] 으로 정합니다.
W1 = tf.Variable(tf.random_uniform([6, 10], -1., 1.))
# 두번째 가중치의 차원을 [첫번째 히든 레이어의 뉴런 갯수, 분류 갯수] -> [10, 3] 으로 정합니다.
W2 = tf.Variable(tf.random_uniform([10, 46], -1., 1.))

# 편향을 각각 각 레이어의 아웃풋 갯수로 설정합니다.
# b1 은 히든 레이어의 뉴런 갯수로, b2 는 최종 결과값 즉, 분류 갯수인 3으로 설정합니다.
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([46]))

# 신경망의 히든 레이어에 가중치 W1과 편향 b1을 적용합니다
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# 최종적인 아웃풋을 계산합니다.
# 히든레이어에 두번째 가중치 W2와 편향 b2를 적용하여 3개의 출력값을 만들어냅니다.
model = tf.add(tf.matmul(L1, W2), b2)

# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train_op = optimizer.minimize(cost)


# 텐서플로우에서 기본적으로 제공되는 크로스 엔트로피 함수를 이용해
# 복잡한 수식을 사용하지 않고도 최적화를 위한 비용 함수를 다음처럼 간단하게 적용할 수 있습니다.
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

#########
# 신경망 모델 학습
######
forRange = 20000
printPoint = 2000

init = tf.global_variables_initializer()
sess1 = tf.Session()
sess1.run(init)
for step in range(forRange):
    sess1.run(train_op, feed_dict={X: xData, Y: number[0]})
    if (step + 1) % printPoint == 0:
        print(step + 1, sess1.run(cost, feed_dict={X: xData, Y: number[0]}))
        
init = tf.global_variables_initializer()
sess2 = tf.Session()
sess2.run(init)
for step in range(forRange):
    sess2.run(train_op, feed_dict={X: xData, Y: number[1]})
    if (step + 1) % printPoint == 0:
        print(step + 1, sess2.run(cost, feed_dict={X: xData, Y: number[1]}))
        
init = tf.global_variables_initializer()
sess3 = tf.Session()
sess3.run(init)
for step in range(forRange):
    sess3.run(train_op, feed_dict={X: xData, Y: number[2]})
    if (step + 1) % printPoint == 0:
        print(step + 1, sess3.run(cost, feed_dict={X: xData, Y: number[2]}))
        
init = tf.global_variables_initializer()
sess4 = tf.Session()
sess4.run(init)
for step in range(forRange):
    sess4.run(train_op, feed_dict={X: xData, Y: number[3]})
    if (step + 1) % printPoint == 0:
        print(step + 1, sess4.run(cost, feed_dict={X: xData, Y: number[3]}))
        
init = tf.global_variables_initializer()
sess5 = tf.Session()
sess5.run(init)
for step in range(forRange):
    sess5.run(train_op, feed_dict={X: xData, Y: number[4]})
    if (step + 1) % printPoint == 0:
        print(step + 1, sess5.run(cost, feed_dict={X: xData, Y: number[4]}))
        
init = tf.global_variables_initializer()
sess6 = tf.Session()
sess6.run(init)
for step in range(forRange):
    sess6.run(train_op, feed_dict={X: xData, Y: number[5]})
    if (step + 1) % printPoint == 0:
        print(step + 1, sess6.run(cost, feed_dict={X: xData, Y: number[5]}))


#########
# 결과 확인
######

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
# print('예측값:', sess1.run(prediction, feed_dict={X: xData}))
# print('예측값:', sess2.run(prediction, feed_dict={X: xData}))
# print('예측값:', sess3.run(prediction, feed_dict={X: xData}))
# print('예측값:', sess4.run(prediction, feed_dict={X: xData}))
# print('예측값:', sess5.run(prediction, feed_dict={X: xData}))
# print('예측값:', sess6.run(prediction, feed_dict={X: xData}))

# print('실제값:', sess1.run(target, feed_dict={Y: number[0]}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess1.run(accuracy * 100, feed_dict={X: xData, Y: number[0]}))
print('정확도: %.2f' % sess2.run(accuracy * 100, feed_dict={X: xData, Y: number[1]}))
print('정확도: %.2f' % sess3.run(accuracy * 100, feed_dict={X: xData, Y: number[2]}))
print('정확도: %.2f' % sess4.run(accuracy * 100, feed_dict={X: xData, Y: number[3]}))
print('정확도: %.2f' % sess5.run(accuracy * 100, feed_dict={X: xData, Y: number[4]}))
print('정확도: %.2f' % sess6.run(accuracy * 100, feed_dict={X: xData, Y: number[5]}))


today = '2017.10.14'
nextValue = mansedate(today)
print(today, sess1.run(prediction, feed_dict={X: [nextValue]})[0]+1)
print(today, sess2.run(prediction, feed_dict={X: [nextValue]})[0]+1)
print(today, sess3.run(prediction, feed_dict={X: [nextValue]})[0]+1)
print(today, sess4.run(prediction, feed_dict={X: [nextValue]})[0]+1)
print(today, sess5.run(prediction, feed_dict={X: [nextValue]})[0]+1)
print(today, sess6.run(prediction, feed_dict={X: [nextValue]})[0]+1)
