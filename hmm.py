# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import random
import numpy as np
from scipy.spatial import distance
import collections

# ==============Hidden Markov Models using Viterbi Algorithm==================================
#Variable elimination algorithm on Hidden Markov Model i.e. Viterbi Algorithm
#Output: the most likely trajectory of a robot in a 10x10 grid
#Robot has a uniform prior probability over all free cells. 

#Global Variable
trans_p = collections.OrderedDict({})

#Parse data into possible states, observations
#Constructs transition probabilities. Eliminates hidden variables based on observations
#Runs viterbi to find most likely trajectory in 11 time steps
def main():
	states = []
	obs = []
	start_p = []
	emit_p = {}
	#load states grid into 2-D array
	with open("hmm-data.txt", 'r') as f:
		states = [np.array(map(int,line.split())) for line in f.readlines()[2:12]]
	states = np.array(states) #10x10 states grid
	
	#store obs into a 11x4 array
	with open("hmm-data.txt", 'r') as f:
		obs = [np.array(map(float,line.split())) for line in f.readlines()[24:35]]
	obs = np.array(obs)
	
	#constructs prior probabilities, transition probabilities, emition probabilities
	#eliminates hidden variables based on observations
	[start_p,emit_p,hidden_variables] = initialize(states, obs)

	#outputs most likely trajectory
	path = viterbi(hidden_variables, start_p, trans_p)

	display(path)

def display(probable_path):
	path = ""
	for point in probable_path:
		path += "(" + str(int(point)/10) + "," + str(int(point)%10) + ")" + "->"

	path = path[:-2] #deleting the last arrow
	print "The most likely trajectory of the robot for 11 time-steps:\n", path

#Uses the prior probabilities(start_p), coordinates(states), transition probabilities(trans_p) to go from 
#one state (t) to another.
#find the coordinate having maximum probability
#follow the prev pointers from that max probability coordinate until reached the timestep 1.
#path is formed by following the pointers
def viterbi(states, start_p, trans_p):
	#array of dictionaries to store probabilities.
	#will have 11 dictionaries for each timestep (actual states)
	#last dictionary will have the maximum probability
	#each dictionary has all hidden variables (coordinates) as keys
	#each coordinate has attributes, probability and prev pointer.
	#prev pointer has the coordinate from which we arrived to current coordinate
	V = [{}] 

	#initialze first timestep/first state X1, with prior probabilities
	#assign prior probability to each coordinate.
	#as first timestep so no prev pointer
	for point in states[0]:
		V[0][point] = {"prob": start_p[point], "prev": None}

	#runs for 10 timesteps
	for t in range(1, 11):
		V.append({})
		#get probabilities of coordinates of current state by product of point's transition probability
		#and probability of that point in prev state
		#save pointer prev to the coordinate we came from
		#if we already calculated that point's probability than calculate again and save the max of two
		#also update the prev pointer to the max probability's prev coordinate
		for st in states[t-1]:
			if st in trans_p[st]:
				for state in trans_p[st]:
					if state in states[t]:
						if state not in V[t]:
							V[t][state] = {"prob": 0.0, "prev": None}
							V[t][state]["prob"] = V[t-1][st]["prob"]*trans_p[state][st]
							V[t][state]["prev"] = st
						else:
							if V[t-1][st]["prob"]*trans_p[state][st] > V[t][state]["prob"]:
								V[t][state]["prob"] = V[t-1][st]["prob"]*trans_p[state][st]
								V[t][state]["prev"] = st

	#find the max probability coordinate which should be in last timestep
	max_point = ''
	max_prob = -1
	for point in V[10]:
		if V[10][point]["prob"] > max_prob:
			max_prob = V[10][point]["prob"]
			max_point = point

	#follow the prev pointer from max_point till the last timestep
	probable_path = []
	probable_path.append(max_point)
	for i in range(10,0,-1):
		max_point = V[i][max_point]["prev"]
		probable_path.insert(0, max_point)

	return probable_path
	
def initialize(states, obs):
	start_p = {}
	emit_p = {}
	n = len(states)
	for i in range(n):
		for j in range(n):
			key = str(i)+str(j)
			neighbors(i,j,states)
			if key not in start_p:
				#Create start_p
				if(states[i][j] == 1):
					start_p[str(i)+str(j)] = 1.0/87.0 #has uniform probability, free cells are 87
				else:
					start_p[str(i)+str(j)] = 0.0

	#eliminate hidden variables based on initial observations given in txt file
	[emit_p, hidden_variables] = eliminate_hidden_vars(states, obs)

	return [start_p,emit_p,hidden_variables]

#calculates euclidean distance of each coordinate from all 4 towers
#find the noisy range from each tower
#if initial observations are in those calculated range than keep those coordinate
#otherwise eliminate it as it is not in the initial noisy distances from towers
#observations decreases the hidden variables
def eliminate_hidden_vars(states, obs):
	emit_p = {}
	hidden_variables = {}
	d1 = 0.0
	d2 = 0.0
	d3 = 0.0
	d4 = 0.0

	for init in range(11):
		hidden_variables[init] = []
	#calculation of range formula: 0.7d..1.3d
	for i in range(len(states)):
		for j in range(len(states)):
			if states[i][j] != 0:
				[d1,d2,d3,d4] = eucl_dist(np.array((i,j)))
				range1 =  np.around(np.arange(0.7*d1, 1.3*d1, 0.1), decimals=1)
				range2 =  np.around(np.arange(0.7*d2, 1.3*d2, 0.1), decimals=1)
				range3 =  np.around(np.arange(0.7*d3, 1.3*d3, 0.1), decimals=1)
				range4 =  np.around(np.arange(0.7*d4, 1.3*d4, 0.1), decimals=1)

				#eliminate states based on noisy distances to towers
				val = str(i)+str(j)
				for row in range(11):
					if obs[row][0] in range1 and obs[row][1] in range2 and obs[row][2] in range3 and obs[row][3] in range4:
						hidden_variables[row].append(val)
			else:
				range1 =  np.array([])
				range2 =  np.array([])
				range3 =  np.array([])
				range4 =  np.array([])


			key = str(i)+str(j)
			if key not in emit_p:
				emit_p[key] = {'t1' : 0.0, 't2': 0.0, 't3' : 0.0, 't4' : 0.0}

			if len(range1) != 0:
				emit_p[key]['t1'] = float(1.0/len(range1))
			if len(range2) != 0:
				emit_p[key]['t2'] = float(1.0/len(range2))
			if len(range3) != 0:
				emit_p[key]['t3'] = float(1.0/len(range3))
			if len(range4) != 0:
				emit_p[key]['t4'] = float(1.0/len(range4))

	return [emit_p, hidden_variables]

def eucl_dist(coord):
	tower1 = np.array((0,0))
	tower2 = np.array((0,9))
	tower3 = np.array((9,0))
	tower4 = np.array((9,9))

	d1 = distance.euclidean(tower1,coord)
	d2 = distance.euclidean(tower2,coord)
	d3 = distance.euclidean(tower3,coord)
	d4 = distance.euclidean(tower4,coord)

	return [d1,d2,d3,d4]

#Observed states: Up, down, left, right
#Transition probabilites: 0.25
# Noise to towers	
#Output 
#def viterbiAlgo():
def neighbors(i, j, world):
	global trans_p
	total = 0.0
	left = False
	up = False
	right = False
	down = False

	n = len(world) - 1
	#Left
	if j - 1 >= 0:
		if world[i][j-1] == 1:
			left = True
			total += 1.0
	#Up
	if i - 1 >= 0:
		if world[i-1][j] == 1:
			up = True
			total += 1.0
	#Right
	if j + 1 <= n:
		if world[i][j+1] == 1:
			right = True
			total += 1.0
	#Down
	if  i + 1 <= n:
		if world[i+1][j] == 1:
			down = True
			total += 1.0
	
	key = str(i) + str(j)
	if key not in trans_p:
		trans_p[key] = collections.OrderedDict({})


	for row in range(len(world)):
		for col in range(len(world)):
			vals = str(row) + str(col)
			if vals not in trans_p[key]:
				trans_p[key][vals] = 0.0
	
	if(left):
		trans_p[key][str(i) + str(j-1)] = (float(1.0/total))
	if(up):
		trans_p[key][str(i-1) + str(j)] = (float(1.0/total))
	if(right):
		trans_p[key][str(i) + str(j+1)] = 9.9
		trans_p[key][str(i) + str(j+1)] = (float(1.0/total))
	if(down):
		trans_p[key][str(i+1) + str(j)] = (float(1.0/total))
	
if __name__ == "__main__":
	main()