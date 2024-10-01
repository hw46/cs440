# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4
    W=np.zeros(train_set.shape[1])
    b=0
    for i in range(max_iter):
        for j in range(len(train_set)):
            y=np.dot(W,np.transpose(train_set[j]))+b
            if y>0 and train_labels[j]==0:
                W-=train_set[j]
                b-=1
            if y<=0 and train_labels[j]==1:
                W+=train_set[j]
                b+=1
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    W,b=trainPerceptron(train_set,train_labels,max_iter)
    developmentData=[]
    for i in range(len(dev_set)):
        y=np.dot(W,np.transpose(dev_set[i]))+b
        if y>0:
            developmentData.append(1)
        else:
            developmentData.append(0)
    return developmentData