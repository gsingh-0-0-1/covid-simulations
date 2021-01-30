#* Copyright (C) Gurmehar Singh and Fabian E. Ortega 2020 - All Rights Reserved
#* Unauthorized copying or distribution of this file, via any medium is strictly prohibited
#* Proprietary and confidential
#* Written by Gurmehar Singh <gurmehar@gmail.com> and Fabian E. Ortega
#*/



import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from scipy.spatial import distance
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
### FUNCTIONS

# function to move people
def MovePeople(people_pos, num_people, D, t):
    #since D is a vector of length num_people, we need to format it
    #so that we can element-wise multiply it by the positions
    n = np.zeros((*np.shape(D), 2))
    n[:, 0] = D
    n[:, 1] = D
    D = n

    delxy = np.random.randn(num_people, 2) * ((2 * D * t) ** (1/2))
    people_pos[:, 1:3] = people_pos[:, 1:3] + delxy #modify the locations of people
    return(people_pos)

#function that we want to fit
def func(x, b):
    return b ** np.array(x)

def CommutePeopleToWork(people):

    #check if people have "reached work", within ~100 meters
    #the * 1 converts from True/False to 0/1
    x_is_close = 1 * abs(people[:, 1] - people[:, 12]) < 70
    y_is_close = 1 * abs(people[:, 2] - people[:, 13]) < 70
    dist_is_close = ((people[:, 1] - people[:, 12]) ** 2)
    dist_is_close += ((people[:, 2] - people[:, 13]) ** 2)
    dist_is_close = dist_is_close ** (1/2)
    dist_is_close = 1 * (dist_is_close < 100)
    dist_is_close = 1 - dist_is_close
    #invert the values. if the person is too close, we need to multiply their delta by 0
    x_is_close = 1 - x_is_close
    y_is_close = 1 - y_is_close

    deltafactor = 1/2
    #modify the x coordinates of people going to work

    #find the difference between commute target and current x position
    delta_x = people[:, 12] - people[:, 1]

    #modify the delta
    delta_x = delta_x * deltafactor

    #modify the people array where the 17th index (going_work) is equal to 1
    people[:, 1] = people[:, 1] + (delta_x * people[:, 9] * people[:, 17] * dist_is_close)


    #modify the y coordinates of people going to work

    #find the difference between commute target and current y position
    delta_y = people[:, 13] - people[:, 2]

    #modify the delta
    delta_y = delta_y * deltafactor

    #modify the people array where the 17th index (going_work) is equal to 1
    people[:, 2] = people[:, 2] + (delta_y * people[:, 9] * people[:, 17] * dist_is_close)


    '''#check if people have "reached work", within ~100 meters
    x_is_close = abs(people[:, 1] - people[:, 12]) < 70
    y_is_close = abs(people[:, 2] - people[:, 13]) < 70

    #set to at_work for those close enough
    people[:, 15][np.where(np.logical_and(x_is_close, y_is_close))] = 1
    #remove to going_to_work tag from those close enough
    #people[:, 17] = np.where(np.logical_and(x_is_close, y_is_close), 0, 1*people[:, 9])
    people[:, 17][np.where(np.logical_and(x_is_close, y_is_close))] = 0

    #add to the location_stay_clock for those who are at work
    #people[:, 19] = np.where(people[:, 15] == 1, people[:, 19]+1, people[:, 19])
    people[:, 19][np.where(people[:, 15] == 1)] += t

    #switch to going_home and reset the clock if stay time has been exceeded
    people[:, 15][np.where(people[:, 19] > people[:, 18])] = 0
    people[:, 16][np.where(people[:, 19] > people[:, 18])] = 1
    people[:, 17][np.where(people[:, 19] > people[:, 18])] = 0
    people[:, 19][np.where(people[:, 19] > people[:, 18])] = 0'''

    if time % 24 <= workdaylength:
        people[:, 17][np.where(people[:, 9] == 1)] = 1
        people[:, 16][np.where(people[:, 9] == 1)] = 0



    return people

def CommutePeopleToHome(people):
    deltafactor = 1/2

    #find the difference between home and current x position
    delta_x = people[:, 10] - people[:, 1]

    #modify the x coordinates of people going home
    delta_x = delta_x * deltafactor

    #modify the people array where the 16th index (going_home) is equal to 1
    people[:, 1] = people[:, 1] + (delta_x * people[:, 9] * people[:, 16])
    

    #find the difference between home and current y position
    delta_y = people[:, 11] - people[:, 2]

    #modify the x coordinates of people going home
    delta_y = delta_y * deltafactor

    #modify the people array where the 16th index (going_home) is equal to 1
    people[:, 2] = people[:, 2] + (delta_y * people[:, 9] * people[:, 16])


    #check if people have "reached home", within ~15 meters
    '''x_is_close = abs(people[:, 1] - people[:, 10]) < 10
    y_is_close = abs(people[:, 2] - people[:, 11]) < 10

    #set to at_home for those close enough
    people[:, 14][np.where(np.logical_and(x_is_close, y_is_close))] = 1
    #remove to going_to_home tag from those close enough
    people[:, 16][np.where(np.logical_and(x_is_close, y_is_close))] = 0

    #add to the location_stay_clock for those who are at home
    #people[:, 19] = np.where(people[:, 14] == 1, people[:, 19]+1, people[:, 19])
    people[:, 21][np.where(people[:, 14] == 1)] += t

    #switch to going_work and reset the clock if stay time has been exceeded
    people[:, 14][np.where(people[:, 21] > people[:, 20])] = 0
    people[:, 16][np.where(people[:, 21] > people[:, 20])] = 0
    people[:, 17][np.where(people[:, 21] > people[:, 20])] = 1
    people[:, 21][np.where(people[:, 21] > people[:, 20])] = 0'''

    if time % 24 >= workdaylength:
        people[:, 17][np.where(people[:, 9] == 1)] = 0
        people[:, 16][np.where(people[:, 9] == 1)] = 1

    return people


def set_homes(people):
    p_ID = 0 #keep track of which person we are assigning to a home
    people[:, 0] = range(0, num_people) #set the ID's
    #create the radii that people will live at (away from the center / workplace)
    r = np.random.random_sample((num_people)) * (graph_radius / 2) + (graph_radius * 0.4) 
    theta = np.random.random_sample((num_people)) * np.pi * 2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    #set home positions
    people[:, 10] = x
    people[:, 11] = y

    #set current x,y
    people[:, 1] = x
    people[:, 2] = y

    #create commuters
    c = np.random.random_integers(low=0, high=num_people-1, size=int(num_people/2))
    people[c, 9] = 1

    return people
    '''for x in range(column_start, column_end, step):
        for y in range(row_start, row_end, step):
            if p_ID > people[-1, 0]:    #check if the person ID has gone too high
                return people           #and then break

            people[p_ID][10] = x #set the home x coordinate for each person
            people[p_ID][11] = y #set the home y coordinate for each person

            people[p_ID][1] = x #set their actual location as well (x)
            people[p_ID][2] = y #set their actual location as well (y)

            if random.randint(1, 100) > 50:
                people[p_ID][9] = 1 #make ~50% of the people commuters

            p_ID += 1 #move to the next person'''



### PARAMETERS 

# diffusion coefficient varies based on the time of day and the social compliance
def calculateDiffusion(current_time, soc_c):
    if current_time < 5:
        D = np.zeros(num_people)
    else:
        main_func = np.sin( np.pi * (current_time - 5) / 19 )
        D = np.random.randn(num_people) + 50 * (1 - soc_c / 100) * (main_func ** 4)
        D[np.where(D < 0)] *= -1 #make all values positive

    return D

#D = (1 - social_compliance / 100) * 50 # 1 = 1 meter

# time scale
t = 0.25 # 1 = 1 hour

graph_radius = 1000 #make it 2km in diameter

pop_density = 500 #people per square kilometer

#use the formula for setting homes to calculate "live-able" area
living_area = np.pi * (graph_radius/1000) * (graph_radius/1000) 
living_area = living_area - (np.pi * (0.4 * (graph_radius/1000)) * (0.4 * (graph_radius/1000)))

num_people = living_area * pop_density
#ensure that this is an even number
num_people = int(num_people/2) * 2

#list of attributes we need
attributes = ["0: ID", 
                "1: loc_x",
                "2: loc_y", 
                "3: sick", 
                "4: recovery_time", 
                "5: recovery_clock", 
                "6: immune", 
                "7: immunity_loss_time", 
                "8: immunity_clock",
                "9: commuter",
                "10: home_x",
                "11: home_y",
                "12: commute_target_x",
                "13: commute_target_y",
                "14: at_home",
                "15: at_work",
                "16: going_home",
                "17: going_work",
                "18: work_stay_time",
                "19: work_stay_clock",
                "20: home_stay_time",
                "21: home_stay_clock"]

workdaylength = float(sys.argv[1])

mean_social_compliance = float(sys.argv[2])

social_compliance = np.random.randn(num_people) + mean_social_compliance

social_compliance[np.where(social_compliance > 100)] = 100
social_compliance[np.where(social_compliance < 0)] = 0

num_attributes = len(attributes)

#create the array
people = np.zeros((num_people, num_attributes))

#create the ID's
people[:, 0] = range(0, num_people)

#set the stay time to a normal distribution with a mean of 8 hours
#the * 0.01 is meant to constrain the workday a bit more - to keep things
#more regular
#people[:, 18] = 0.01 * np.random.randn(num_people) + 8

#set the home stay time to the remaining part of the day
#people[:, 20] = 24 - people[:, 18]
home_stay = 16

#create the recovery times for people
people[:, 4] = np.random.randn(num_people) + (28 * 24/t) #28 days

#create the immunity loss times for people
people[:, 7] = np.random.randn(num_people) + (75 * 24/t) #75 days

#set the home locations of all the people

#we want the homes to be separate from the target commute location
#let's put them on the left side of the graph
#we can put them in rows and columns
column_start = -graph_radius
column_end = 0

row_start = -graph_radius
row_end = graph_radius

step = int(graph_radius/10)

people = set_homes(people)

#set one person who is a commuter to be infected
people[:, 3][random.choice(np.where(people[:, 9] == 1)[0])] = 1


#initialize the plot
plt.figure(figsize = (15, 7))
plt.suptitle("Mean Social Compliance: "+str(mean_social_compliance)+"%, Workday Length: "+str(workdaylength)+"h")

### RUN SIMULATION
runs = 0
runs_to_do = int(float(sys.argv[3]) / 0.25)
time = 0
#different clock for "time of day" - makes measurement a bit easier internally
time_of_day = 9 #start at 9 am

goingtowork = True

threshold = np.array(1.5) #1.5 meters

commute_cycle_started = False

#create lists to collect data
time_intervals = []
over_time_active = []
infected_per_run = []
r_effective = []

infected_by_commuters = []
infected_by_residents = []

for run in range(runs_to_do):

    D = calculateDiffusion(time_of_day, social_compliance)

    display_time_hour = int(time_of_day) % 24
    display_time_minute = 60 * (time_of_day - display_time_hour)
    display_time_minute = int(display_time_minute)

    if len(str(display_time_hour)) == 1:
        display_time_hour = "0" + str(display_time_hour)
    if len(str(display_time_minute)) == 1:
        display_time_minute = "0" + str(display_time_minute)
    display_time = str(display_time_hour) + ":" + str(display_time_minute)

    if not commute_cycle_started: #make people go to work 
        people[:, 17][np.where(people[:, 9] == 1)] = 1 #make sure we only set going_work to true for commuters
        commute_cycle_started = True

    plt.subplot(1, 3, 1)

    plt.text(-graph_radius*1.5, graph_radius*1.4, "Time: "+str(display_time), fontsize=12,
        verticalalignment='top', horizontalalignment='left')
    plt.axis([-graph_radius*1.5, graph_radius*1.5, -graph_radius*1.5, graph_radius*1.5]);
    # move your people once per step
    people = MovePeople(people, num_people, D, t)
    people = CommutePeopleToWork(people)
    people = CommutePeopleToHome(people)

    #plot healthy people
    plt.scatter(x = people[:, 1][np.where(people[:, 3] == 0)],
            y = people[:, 2][np.where(people[:, 3] == 0)],
            s = 10,
            color = 'blue')
    #plot sick people
    plt.scatter(x = people[:, 1][np.where(people[:, 3] == 1)],
            y = people[:, 2][np.where(people[:, 3] == 1)],
            s = 10,
            color = 'red')    
    #plot immune people
    plt.scatter(x = people[:, 1][np.where(people[:, 6] == 1)],
            y = people[:, 2][np.where(people[:, 6] == 1)],
            s = 10,
            color = 'green')
    plt.grid(True)
    plt.pause(1)
    plt.cla()

    infected_array = people[:, 1:3][np.where(people[:, 3] == 1)]

    #create an array of just the positions of people
    positions = np.zeros((num_people, 2))
    positions[:, 0] = people[:, 1]
    positions[:, 1] = people[:, 2]

    #infect those who are close enough and currently healthy
    distances = distance.cdist(infected_array, positions)
    #this is what the "intersect1d" is for - it makes sure we are only identifying people who are both
    #within the threshold and currently healthy

    #now, we add to the cases gained for this run
    infected_per_run += [len(np.intersect1d(np.where(distances < threshold)[1], np.where(people[:, 3] == 0)))]

    #now we infect those close enough
    people[:, 3][np.intersect1d(np.where(distances < threshold)[1], np.where(people[:, 3] == 0))] = 1
    people[:, 3][np.where(people[:, 6] == 1)] = 0 #heal immune people instantly


    #note for any code readers - we have immunity/recovery mechanics, but these are relatively obsolete
    #now that we are specifically looking at short-term outbreak cases. these were crucial in earlier
    #versions of the simulation where we looked at the spread over many weeks/months, but don't come
    #into play that much anymore. though, if you let the simulation run long enough, you might see some
    #of the effects. (long here means something like a month in-simulation time, so the third parameter 
    #you use would be something like 700 or 800 hours)

    #recovery code
    people[:, 5][np.where(people[:, 3] == 1)] += t #add to clock for those who are sick
    people[:, 3][np.where(people[:, 5] > people[:, 4])] = 0 #remove sick tag
    people[:, 6][np.where(people[:, 5] > people[:, 4])] = 1 #add immune tag
    people[:, 5][np.where(people[:, 5] > people[:, 4])] = 0 #reset the recovery clock

    #immunity code
    people[:, 8][np.where(people[:, 6] == 1)] += t #add to clock for those who are immune
    people[:, 6][np.where(people[:, 8] > people[:, 7])] = 0 #remove immune tag
    people[:, 8][np.where(people[:, 8] > people[:, 7])] = 0 #reset immunity clock

    #update the data lists
    time_intervals += [time]
    over_time_active += [np.sum(people[:, 3])]

    #graph the over_time data
    plt.subplot(1, 3, 2)
    plt.cla()
    total = plt.plot(time_intervals, over_time_active, color="blue", label="Total Cases")

    if runs > 1:
        #model = np.polyfit(time_intervals, over_time_active, 2)
        if time <= workdaylength:
            popt, pcov = curve_fit(func, np.array(time_intervals), over_time_active)

            plt.plot(np.array(time_intervals), func(np.array(time_intervals), *popt), color="red", label="Model")
        else:
            plt.plot(np.array(time_intervals[:int(workdaylength/t)]), func(np.array(time_intervals[:int(workdaylength/t)]), *popt), color="red", label="Model")

    daily = plt.plot(time_intervals, infected_per_run, color="green", label="Increase in Cases")
    plt.ylabel("Cases")
    plt.xlabel("Time (h)")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.cla()
    #get the people infected this run, and the total infected last run
    try:
        r_effective += [infected_per_run[-1]/over_time_active[-2]]
    except IndexError: #if we haven't even passed the first couple of runs, ignore and add 0
        r_effective += [0]
    plt.plot(time_intervals, r_effective, color="red")
    plt.ylabel("Basic Reproduction Number")
    plt.xlabel("Time (h)")

    runs += 1
    time += t
    time_of_day += t
    time_of_day = time_of_day % 24


print("Active cases at the end of " + str(t * runs_to_do) + " hours: " + str(over_time_active[-1]) + "\t", end='')
print("Runaway Factor: ", end='')
print(*popt)

'''plt.close()

plt.plot(over_time)
plt.xlabel("Time")
plt.ylabel("Number of Cases")
plt.show()'''
