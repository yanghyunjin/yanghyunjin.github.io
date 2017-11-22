# -*- coding: utf-8 -*- 
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

numberTemp =[]
number = []
# number = [[] for row in range(1000)]
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

	xDataTemp = [ganziNumber[(firstganzi[0] + year[index])%10],
				ziziNumber[(firstzizi[0] + year[index])%12],
				ganziNumber[(firstganzi[1] + month[index])%10],
				ziziNumber[(firstzizi[1] + month[index])%12],
				ganziNumber[(firstganzi[2] + day[index])%10],
				ziziNumber[(firstzizi[2] + day[index])%12]]
	xData.append(xDataTemp)

	numberTemp =[line[1],line[2],line[3],line[4],line[5],line[6]]
	number.append(numberTemp)
	
	index = index+1

index = 0
for item in date1:
	print (ganzi[manse[index][0]] + zizi[manse[index][1]] + ganzi[manse[index][2]] + zizi[manse[index][3]]+ ganzi[manse[index][4]] + zizi[manse[index][5]])
	index = index + 1
print xData
print number
f.close()   