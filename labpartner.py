import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from copy import deepcopy

#This code is meant to ease the task of data collecting. The idea is, when called upon
#it asks for a variable name, say A. Then it asks for its error and finally for 
#every value you are to measure of it. When all data is taken, as data is stored in
#numpy arrays you can compute functions of every measurement, as someone would do in 
#excel. This way one can have several well organised data inputs and plotted with 
#matplotlib. My final goal is to be able to export tables to latex format and graphs
#to tikz format. 

#A lab session normally has several experiments, and every experiment has several variables 
#in it. 
class Variable:
    def __init__(self):
        self.values = np.array([]) #The list upon which the measurements of the variable are stored
        self.errors = np.array([]) #The list of the errors of each measurement

    def measurement(self):
        print("Type repeat to repeat, and nothing to stop")

        measurements = deepcopy(self.values)
        while True:
            measure = input("\nIntroduce un valor:  ")
            if measure == "repeat":
                measurements = deepcopy(self.values)
            elif measure == "":
                break
                
            np.append(measurements, float(measure))

        np.append(self.values, measurements)

class new_experiment:
    def __init__(self):
        self.variables = []  #When a new_experiment instance is created it asks how many variables
        for i in count(1):
            name = input("Variable {} name: ".format(i)) #The name of the variable
            if name == "": #If no input, the count stops
                break
            variable = Variable() #variable is a Variable instance
            variable.name = name #Save the variable name 
            self.variables.append(variable) #Append that variable to the variables list
        self.N = len(self.variables) #Number of variables in the system
        self.variable_names = [p.name for p in self.variables] #A list containing the names of the variables. 

    def run_experiment(self):  #When an experiment is ran, we only measure one
        print("The variables are {}, which one are you going to measure?".format(", ".join(self.variable_names)))
        var_name = input("")
        dependent_var = input("And which one is the dependent variable? ")
        var = self.variable_names.index(var_name)
        self.variables[var].measurement()
        N = len(self.variables[var].values)
        print(N)
        dep_var = self.variable_names.index(dependent_var)
        
        min_value = float(input("Minimum value of the dependent var: "))
        max_value = float(input("Maximum value of the dependent var: "))

        print([p.values for p in self.variables])
        self.variables[dependent_var].values = np.linspace(min_value, max_value, num=N)
    

print(np.linspace.__doc__)
experiment = new_experiment()
experiment.run_experiment()

