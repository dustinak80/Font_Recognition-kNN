# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 07:43:28 2019

@author: Dustin
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
import time

t0=time.time()
fsz=(12,8)

def get_data(file):
    """Create a function for loading the csv and editing the file the way we want
    for this excel format
    """
    
    import pandas as pd
    
    #Import csv as a Data Frame, drop columns do now want, drop rows with value na
    data = pd.read_csv(file)
    data = data.drop(['fontVariant', 'm_label',  'orientation', 'm_top', 'm_left', 'originalH', 'originalW', 'h', 'w' ], axis = 1)
    data.dropna()
    print(data.shape)
    
    data = data.loc[(data.strength ==.4) & (data.italic == 0)]
        
    return data

#load the files and assign to class
file = r'C:\Users\Dustin\Desktop\Masters Program\Fall Semester\aaaaStatistical Learning and Data mining\Homework and Readings\Homework\HW2\CALIBRI.csv'
data = get_data(file)
cl2 = data.copy()

file = r'C:\Users\Dustin\Desktop\Masters Program\Fall Semester\aaaaStatistical Learning and Data mining\Homework and Readings\Homework\HW2\COURIER.csv'
data = get_data(file)
cl1 = data.copy()

file = r'C:\Users\Dustin\Desktop\Masters Program\Fall Semester\aaaaStatistical Learning and Data mining\Homework and Readings\Homework\HW2\TIMES.csv'
data = get_data(file)
cl3 = data.copy()

del data, file

#Print the sizes
n_cl1 = cl1.shape; print( n_cl1)
n_cl2 = cl2.shape; print( n_cl2)
n_cl3 = cl3.shape; print( n_cl3)
v = 400

#combine the three classes for full set of Data
data=pd.concat([cl1,cl2,cl3],axis=0)
data.index = range(len(data))
n_data = data.shape; print(n_data)
if n_data[0] == (n_cl1[0]+n_cl2[0]+n_cl3[0]): #check
    print('\nLooks good')

t1=time.time()
"""
Part 0
"""
## mean and standard deviation
m=data[data.columns[3:]].mean() #Mean
sd=data[data.columns[3:]].std() #Standard Deviation

## Plots
#Scatter Plot
plt.figure(figsize=fsz)
plt.scatter(m, sd, alpha=0.5)
plt.title('Mean vs Standard Deviation')
plt.ylabel('Standard Deviation')
plt.xlabel('Mean')

#Histogram
#mean
plt.figure(figsize=fsz)
n, bins, patches = plt.hist(x=m, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Mean')
plt.ylabel('Frequency')
plt.title('Mean of Features')
maxfreq = n.max()
#standard deviation
plt.figure(figsize=fsz)
n, bins, patches = plt.hist(x=sd, bins='auto', color='green', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Standard Deviation')
plt.ylabel('Frequency')
plt.title('Standard Deviation of Features')
maxfreq = n.max()

## Centralize and stadardize the matrix
data_s = (data[data.columns[3:]]-m)/sd #Centralizing and Standardizing the Data
#print(data_s.mean(), data_s.std())   #Check the values , m=0 sd=1
""" just for visualization of centralized data
#plot of centralized and standardized data
fig1, ax1 = plt.subplots(figsize=fsz)
ax1.set_title('Samples of Shaped Data')
ax1.boxplot([data_s[data_s.columns[3]],data_s[data_s.columns[4]],data_s[data_s.columns[5]]], meanline = True, showmeans=True, labels = ['r0c0','r0c1','r0c2'])
"""
t2=time.time()
"""
Part 1
"""
## correlation matrix
corr_m = data_s.corr()
eigs = np.linalg.eig(corr_m)

## eigen values
eig_value = eigs[0]
eig_vecto = eigs[1] 
print(sum(eig_vecto[:,0]**2))
print('\n Sum of eig: ', sum(eig_value))

## Rj
ratio = np.cumsum(eig_value)/sum(eig_value)
# Where does Rj>.90
tratio_ind = np.where(ratio>.90) #seperates the values
r_ind=min(tratio_ind[0]) #gets the index
r_val = ratio[r_ind] #gets the value
del tratio_ind

## Plots
#plot eig_values
fig2, ax2 = plt.subplots(figsize=fsz)
ax2.set_title('Eigenvalues')
ax2.set_ylabel('Eignevalues')
ax2.scatter(range(len(eig_value)), eig_value, alpha=0.5)
#Plot Rj
fig3, ax3 = plt.subplots(figsize=fsz)
ax3.set_title('Ratio Trend')
ax3.set_ylabel('Ratio')
ax3.scatter(range(len(ratio)), ratio, c='b', alpha=.2)
ax3.scatter(r_ind, r_val, c='yellow', alpha=1)
ax3.plot(range(len(ratio)), [.90]*400, 'r--', alpha=.2)
ax3.text(20,r_val,'90% Line', horizontalalignment = 'right', verticalalignment = 'bottom')
ax3.text(r_ind,.1,'Eigenvalue 77 > 90%', verticalalignment = 'bottom')

## Plot seperation between top 3 Eigenvalues
#The normalized (unit “length”) eigenvectors, such that the column 
#v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
projected=np.dot(data_s,eig_vecto) #Eigenvalues are already transposed
projected = pd.concat([data[data.columns[0:3]],pd.DataFrame(projected)],axis=1)
data_sp = projected[projected.columns[0:6]]
#print(projected[projected.columns[3]].mean())
#Seperating indices for each font
cal=list(data_sp[data_sp['font'] == 'CALIBRI'].index) 
cou=list(data_sp[data_sp['font'] == 'COURIER'].index) 
tnr=list(data_sp[data_sp['font'] == 'TIMES'].index)
print((len(cal)+len(cou)+len(tnr)), len(data_sp)) #check 
#2Dplot 
plt.figure(figsize=fsz)
plt.title('Plot Principal Components 1 and 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.scatter(data_sp.iloc[cal,3], data_sp.iloc[cal,4], c='r' , alpha = .2, s=10)
plt.scatter(data_sp.iloc[cou,3], data_sp.iloc[cou,4], c='b', alpha = .2, s=10 )
plt.scatter(data_sp.iloc[tnr,3], data_sp.iloc[tnr,4], c='g', alpha = .2, s=10 )
plt.legend(['CALIBRI','COURIER' ,'TIMES'])
#plot three dimensional
fig=plt.figure(figsize=fsz)
ax4 = fig.add_subplot(111, projection='3d')
#ax.view_init(135, 135)
ax4.set_title('Plot Principal Components 1, 2, and 3')
ax4.set_xlabel('Principal Component 1')
ax4.set_ylabel('Principal Component 2')
ax4.set_zlabel('Principal Component 3')
ax4.scatter(data_sp.iloc[cal,3], data_sp.iloc[cal,4], data_sp.iloc[cal,5], c='r' , alpha = .2)
ax4.scatter(data_sp.iloc[cou,3], data_sp.iloc[cou,4], data_sp.iloc[cou,5], c='b', alpha = .2)
ax4.scatter(data_sp.iloc[tnr,3], data_sp.iloc[tnr,4], data_sp.iloc[tnr,5], c='g', alpha = .2)
ax4.legend(['CALIBRI','COURIER' ,'TIMES'])
#plot calibri and courier
fig=plt.figure(figsize=fsz)
ax5 = fig.add_subplot(111, projection='3d')
ax5.view_init(20, 45)
ax5.set_title('Plot Principal Components 1, 2, and 3')
ax5.set_xlabel('Principal Component 1')
ax5.set_ylabel('Principal Component 2')
ax5.set_zlabel('Principal Component 3')
ax5.scatter(data_sp.iloc[cal,3], data_sp.iloc[cal,4], data_sp.iloc[cal,5], c='r' , alpha = .2)
ax5.scatter(data_sp.iloc[cou,3], data_sp.iloc[cou,4], data_sp.iloc[cou,5], c='b', alpha = .2)
ax5.legend(['CALIBRI','COURIER'])
#plot calibri and times
fig=plt.figure(figsize=fsz)
ax6 = fig.add_subplot(111, projection='3d')
ax6.view_init(70, 200)
ax6.set_title('Plot Principal Components 1, 2, and 3')
ax6.set_xlabel('Principal Component 1')
ax6.set_ylabel('Principal Component 2')
ax6.set_zlabel('Principal Component 3')
ax6.scatter(data_sp.iloc[cal,3], data_sp.iloc[cal,4], data_sp.iloc[cal,5], c='r' , alpha = .2)
ax6.scatter(data_sp.iloc[tnr,3], data_sp.iloc[tnr,4], data_sp.iloc[tnr,5], c='g', alpha = .2)
ax6.legend(['CALIBRI','TIMES'])
#print(ax6.azim)
#plot courier and times
fig=plt.figure(figsize=fsz)
ax7 = fig.add_subplot(111, projection='3d')
ax7.view_init(20, 120)
ax7.set_title('Plot Principal Components 1, 2, and 3')
ax7.set_xlabel('Principal Component 1')
ax7.set_ylabel('Principal Component 2')
ax7.set_zlabel('Principal Component 3')
ax7.scatter(data_sp.iloc[cou,3], data_sp.iloc[cou,4], data_sp.iloc[cou,5], c='b', alpha = .2)
ax7.scatter(data_sp.iloc[tnr,3], data_sp.iloc[tnr,4], data_sp.iloc[tnr,5], c='g', alpha = .2)
ax7.legend(['COURIER' ,'TIMES'])
#print(ax7.azim)
t3=time.time()
"""
Part 2
"""
#Proportion out the section for test and train
tt_data = pd.concat([data[data.columns[0]],data_s],axis=1)
c1 = shuffle(tt_data[tt_data['font'] == 'COURIER']) #shuffles the first cluster
c1_pivot = int(round(n_cl1[0]*.8,0)) #round the pivot point
c1_train = c1[0:c1_pivot] #get train for classifiction
c1_test = c1[c1_pivot:] #get test for classification
c2 = shuffle(tt_data[tt_data['font'] == 'CALIBRI']) #shuffles the second cluster
c2_pivot = int(round(n_cl2[0]*.8,0)) #round the pivot point
c2_train = c2[0:c2_pivot] #get train for classifiction
c2_test = c2[c2_pivot:] #get test for classification
c3 = shuffle(tt_data[tt_data['font'] == 'TIMES']) #shuffles the thirds cluster
c3_pivot = int(round(n_cl3[0]*.8,0)) #round the pivot point
c3_train = c3[0:c3_pivot] #get train for classifiction
c3_test = c3[c3_pivot:] #get test for classification

#Get your X,Y Train and X,y Test
train = pd.concat([c1_train,c2_train,c3_train],axis=0) #all of train Data
train.index = range(len(train))
test = pd.concat([c1_test,c2_test,c3_test],axis=0) #all of test Data
test.index = range(len(test))

nbs=k=[3] #neighbors

X_train, X_test = train[train.columns[1:]] , test[test.columns[1:]]
y_train, y_test = train[train.columns[0]] , test[test.columns[0]]

def KNN(X_train,y_train,X_test, nbs):
    pred=[]
    for i in range(len(nbs)):
        neigh = KNC(n_neighbors=nbs[i]) #Built in function that chooses the best method
        neigh.fit(X_train, y_train)
        preda=list(neigh.predict(X_test))
        pred.append(preda) #code with all of the predictions
    
    #print(pred)
    return pred

pred=KNN(X_train, y_train, X_test, nbs) #call the funciton

def percentage(pred, cal, cou, tnr, y_test):
    cal=len(y_test[y_test == 'CALIBRI'])
    cou=len(y_test[y_test == 'COURIER']) 
    tnr=len(y_test[y_test == 'TIMES'])
    idx = y_test.index 
    total_per = []
    cal_per = []
    cou_per = []
    tnr_per = []
    for j in range(len(pred)):
        correct = 0
        wrong = 0
        cal_wrong=0
        cou_wrong = 0
        tnr_wrong = 0
        for i in range(len(pred[j])):
            if (pred[j][i] == y_test[idx[i]]) == True:
                correct += 1
            else:
                wrong += 1
                if pred[j][i] == 'CALIBRI':
                    cal_wrong += 1
                    continue
                if pred[j][i] == 'COURIER':
                    cou_wrong += 1
                    continue
                if pred[j][i] == 'TIMES':
                    tnr_wrong += 1
                    continue
        total_per.append(correct/len(pred[j]))
        cal_per.append((cal-cal_wrong)/cal)
        cou_per.append((cou-cou_wrong)/cou)
        tnr_per.append((tnr-tnr_wrong)/tnr)
    return total_per, cal_per, cou_per, tnr_per

total_per, cal_per, cou_per, tnr_per = percentage(pred,cal,cou,tnr, y_test)
print(cal_per, cou_per, tnr_per)
print(total_per)

#Plot Accuracy
plt.figure(figsize=fsz)
plt.scatter(nbs,cal_per,c='b', alpha = .2)
plt.plot(nbs,cal_per,'b', alpha = .2)
plt.scatter(nbs,cou_per,c='r', alpha = .2)
plt.plot(nbs,cou_per,'r', alpha = .2)
plt.scatter(nbs,tnr_per,c='g', alpha = .2)
plt.plot(nbs,tnr_per,'g', alpha = .2)
plt.plot(nbs,total_per,'black', alpha = 1)
plt.scatter(nbs,total_per,c='black', alpha = 1)
plt.xlabel('k - Values')
plt.ylabel('% Correct')
plt.title('Percent Correct for each K Value')
plt.legend(['CALIBRI','COURIER' ,'TIMES'])

#Confusion Matrix
conf_m = confusion_matrix(y_test, pred[3])
class_report = classification_report(y_test, pred[0])

t4=time.time()
time = [(t1-t0)/60,(t2-t1)/60,(t3-t2)/60,(t4-t3)/60]
total_time = (t4-t0)/60
print('Total Time: ',total_time)
print('Times: ', time)


