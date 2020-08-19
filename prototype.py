import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from scipy.spatial import distance

### FUNCTIONS

# function to move people
def MovePeople(people_pos, num_people, D, t):
    delxy = np.random.randn(num_people, 2) * math.sqrt(2 * D * t)
    return(people_pos + delxy)


### PARAMETERS 

# diffusion coefficient
D = 50 # 1 = 1 meter

# time scale
t = 0.1 # 1 = 1 day

#number of people, number of clusters
num_people = int(sys.argv[1])
num_clusters = int(sys.argv[2])
people_per_cluster = int(num_people / num_clusters)

#recovery rate
recoveryrates = int(sys.argv[3]) + np.random.randn(num_people)

#rate of immunity loss
imm_loss_rate = int(sys.argv[4]) + np.random.randn(num_people)

#distance from 0,0 that the center of clusters can generate
radius = float(sys.argv[5])

#dispersion factor for clusters - higher means the cluster will be more dispersed/spread out
pop_disp = float(sys.argv[6])

#threshold for infecting people
threshold = float(sys.argv[7])

#make arrays to prevent clusters from generating on top of each other
center_xlist = np.zeros(num_clusters)
center_ylist = np.zeros(num_clusters)

for i in range(num_clusters):
    vars()["pt"+str(i)] = pop_disp * np.random.randn(people_per_cluster, 2)

    #check for existing clusters, stay away from them
    cx = random.uniform(-(radius * 0.9), (radius * 0.9))
    cy = random.uniform(-(radius * 0.9), (radius * 0.9))

    while (abs(cx - center_xlist) < radius / 10).any() or (abs(cy - center_ylist) < radius / 10).any():
        cy = random.uniform(-radius, radius)
        cx = random.uniform(-radius, radius)

    center_xlist[i] = cx
    center_ylist[i] = cy


    vars()["pt"+str(i)][:, 0] += cx
    vars()["pt"+str(i)][:, 1] += cy
    if i == 0:
        people_pos = pt0
    else:
        people_pos = np.concatenate((people_pos, vars()["pt" + str(i)]))


#create the vector for infected people
infected_vector = np.zeros(num_people)
infected_vector[0] = 1

#get the position of where it all started
epicenter = people_pos[0]

#create the vector for time to recovery
recovery_vector = np.zeros(num_people)

#create the vector for immune people
immunity_vector = np.zeros(num_people)

#create the vector for immunity loss
imm_loss_vector = np.zeros(num_people)

#compile a list of cases versus iterations
over_time = []

#these are the people who are infected
infected = np.where(infected_vector==1)[0]

#the number of people who are infected
num_infected = np.sum(infected)

#initialize the plot
plt.figure(figsize = (7, 7))

### RUN SIMULATION
runs = 0

while (num_people - num_infected) > 5: #if enough people get sick, end the simulation
    plt.axis([-radius * 2, radius * 2, -radius * 2, radius * 2])
    #if enough people recover, end the simulation
    if (num_infected < 5 and runs > 100):
        break

    # move your people once per step
    people_pos = MovePeople(people_pos, num_people, D, t)

    # plotting code!
    #plot healthy
    plt.scatter(x = people_pos[np.where(infected_vector==0), 0], 
                y = people_pos[np.where(infected_vector==0), 1], s=10, color='blue', label="Healthy")
    #plot sick
    plt.scatter(x = people_pos[np.where(infected_vector==1), 0], 
                y = people_pos[np.where(infected_vector==1), 1], s=10, color='red', label="Sick/Carrier")
    #plot immune
    plt.scatter(x = people_pos[np.where(immunity_vector==1), 0], 
                y = people_pos[np.where(immunity_vector==1), 1], s=10, color='green', label="Immune")
    #plot epicenter
    plt.scatter(x = epicenter[0],
                y = epicenter[1], s=10, color='purple')

    plt.legend()
    plt.grid(True)

    plt.pause(0.0001)

    plt.cla()

    #get the indices of those who are infected
    infected = np.where(infected_vector==1)
    infected = infected[0]

    #get the number of people infected
    num_infected = np.sum(infected_vector)

    #create an array of infected people
    infected_array = people_pos[np.where(infected_vector==1)]

    #the reason I didn't create a "healthy array" here is a bit complicated - if I did, I'd end up losing
    #the index of each individual person, which would be more difficult to deal with. This itself runs 
    #relatively fast, so I don't mind it
    distances = distance.cdist(infected_array, people_pos)
    infected_vector[np.where(distances < threshold)[1]] = 1
    infected_vector[np.where(immunity_vector == 1)] = 0

    #deal with currently infected people, recovering people, add immunity
    recovery_vector[np.where(infected_vector==1)] += t
    infected_vector[np.where((recovery_vector - recoveryrates) > 0)] = 0
    #infected_vector[np.where(recovery_vector > recoveryrate)] = 0
    immunity_vector[np.where((recovery_vector - recoveryrates) > 0)] = 1
    #immunity_vector[np.where(recovery_vector > recoveryrate)] = 1
    recovery_vector[np.where((recovery_vector - recoveryrates) > 0)] = 0
    #recovery_vector[np.where(recovery_vector > recoveryrate)] = 0

    imm_loss_vector[np.where(immunity_vector == 1)] += t
    immunity_vector[np.where((imm_loss_vector - imm_loss_rate) > 0)] = 0
    imm_loss_vector[np.where((imm_loss_vector - imm_loss_rate) > 0)] = 0

    over_time += [num_infected]

    runs += t
    print(runs)

plt.close()

plt.figure(figsize = (7, 7))
plt.plot(np.arange(0, runs, t), over_time)
plt.xlabel("Time (Days)")
plt.ylabel("Number of Cases")
plt.show()




