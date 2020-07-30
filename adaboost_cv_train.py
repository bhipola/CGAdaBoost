# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:01:14 2019

@author: belen
"""



from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
from sklearn import tree
from sklearn.tree import plot_tree
import pydotplus
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO

# =============================================================================

    
def aciertos(y_est,y) :
    i=0
    ind=[]
    while i<len(y):
        if y_est[i]==y[i]: ind.append(1)
        else: ind.append(0)
        i=i+1
    return(ind)
    
def aciertos_suma(y_est,y) :
    i=0
    ind=[]
    while i<len(y):
        if y_est[i]==y[i]: ind.append(1)
        else: ind.append(0)
        i=i+1
    return(np.sum(ind))
    
def tasa_error(y_est,y) : #def como num errores/ total de obs
    i=0
    ind=[]
    while i<len(y):
        if y_est[i]==y[i]: ind.append(0)
        else: ind.append(1)
        i=i+1
    return(np.sum(ind)/len(y_est))
    

def fallos(y_est,y) :
    i=0
    ind=[]
    while i<len(y):
        if y_est[i]!=y[i]: ind.append(1)
        else: ind.append(0)
        i=i+1
    return(ind)
    
def coincidencias(l1,l2):
    i=0
    N00=0
    N10=0
    N01=0
    N11=0
    while i<len(l1):
        if l1[i]==l2[i]==1: N11=N11+1
        elif l1[i]==l2[i]==0: N00=N00+1
        elif (l1[i]==0 and l2[i]==1): N01=N01+1
        elif (l1[i]==1 and l2[i]==0): N10=N10+1
        i=i+1
    if N00+N11+N10+N01!=len(l1): print('Error: la longitud de las listas y de los aciertos y fallos no coindicen. Comprobar que la lista solo esté formada por unos y ceros.')
    return N11,N00,N10,N01


def DFM(c1,c2,y):
    aciertos_1=aciertos(c1,y)
    aciertos_2=aciertos(c2,y)
    N11, N00, N10, N01= coincidencias(aciertos_1, aciertos_2)
    DFM=(N00/(N11+N00+N10+N01))
    f=1/(DFM+0.01)
    return(f)

def corrcoef(c1,c2,y):
    aciertos_1=aciertos(c1,y)
    aciertos_2=aciertos(c2,y)
    N11, N00, N10, N01= coincidencias(aciertos_1, aciertos_2)
    CC=((N11*N00-N10*N01)/(np.sqrt((N11+N10)*(N01+N00)*(N11+N01)*(N00+N10))))
    f=1/(CC+0.01+1)
    return(f)

def Q(c1,c2,y):
    aciertos_1=aciertos(c1,y)
    aciertos_2=aciertos(c2,y)
    N11, N00, N10, N01= coincidencias(aciertos_1, aciertos_2)
    Q=(N11*N00-N01*N10)/(N11*N00+N01*N10)
    f=1/(Q+0.01+1)
    return(f)
    

#c_filas es el output estimado por cada clasificador
def contribution_value_DFM(y_val,c_filas_val):#esto es a partor del clasificador num2
    cont=[]
    i=0
    while i<len(c_filas_val):
       j=0
       f_i=np.array([])
       while j<len(c_filas_val):
           if i==j: j=j+1
           else: 
               f_i=np.append(f_i,[DFM(c_filas_val[i],c_filas_val[j],y_val)])
               j=j+1
       cont.append(0.5*np.sum(f_i))
       i=i+1
    return(cont)

#c_filas es el output estimado por cada clasificador
def contribution_value_CC(y_val,c_filas_val):#esto es a partor del clasificador num2
    cont=[]
    i=0
    while i<len(c_filas_val):
       j=0
       f_i=np.array([])
       while j<len(c_filas_val):
           if i==j: j=j+1
           else: 
               f_i=np.append(f_i,[corrcoef(c_filas_val[i],c_filas_val[j],y_val)])
               j=j+1
       cont.append(0.5*np.sum(f_i))
       i=i+1
    return(cont)

#c_filas es el output estimado por cada clasificador
def contribution_value_Q(y_val,c_filas_val):#esto es a partor del clasificador num2
    cont=[]
    i=0
    while i<len(c_filas_val):
       j=0
       f_i=np.array([])
       while j<len(c_filas_val):
           if i==j: j=j+1
           else: 
               f_i=np.append(f_i,[Q(c_filas_val[i],c_filas_val[j],y_val)])
               j=j+1
       cont.append(0.5*np.sum(f_i))
       i=i+1
    return(cont)
    
    
def decision_tree_threshold(X_train, y_train,weight):
    clf = tree.DecisionTreeClassifier(max_depth=1)
    clf=clf.fit(X_train.reshape(-1,1), y_train, sample_weight=weight)
    threshold = clf.tree_.threshold
   # tree.plot_tree(clf) 
    return(threshold[0])

def decision_tree_pred(X_train, y_train, X_test,weight):
    clf = tree.DecisionTreeClassifier(max_depth=1)
    clf=clf.fit(X_train.reshape(-1,1), y_train, sample_weight=weight)
    pred=clf.predict(X_test.reshape(-1,1))
#    print([sum(pred==1),sum(pred==-1)])
    return (pred)


def weight(W,t,contribution_final,y_train,X_train):
    aciertos_t=aciertos(decision_tree_pred(X_train[:,t-1], y_train, X_train[:,t-1],W[t-1]),y_train)
    i=0
    W_anterior=W[t-1]
    weights_new=[]
    if len(contribution_final)==1:
       while i<len(X_train):
           weights_new.append(W_anterior[i]*(1/(contribution_final[0]))**(aciertos_t[i]))
           i=i+1
    else:
        while i<len(X_train):
            weights_new.append(W_anterior[i]*(1/max(contribution_final))**(aciertos_t[i]))
            i=i+1
    W=weights_new/np.sum(weights_new)
    return(np.array([W]))
    
def scouting(X_train, y_train, weight, orden):
    W_e=np.array([]) 
    nan_values=np.where([np.isnan(orden)])[1]
    if sum(np.isnan(weight)>0):
        j=nan_values[0]
        W_e_c=np.nan
    else:
        if len(nan_values)>1:
            for i in (nan_values):#para toda variable que quede
                    W_e=np.concatenate((W_e,np.array([np.sum(weight*fallos(decision_tree_pred(X_train[:,i],y_train,X_train[:,i],weight),y_train))])),axis=0)
            b = np.where(np.isclose(W_e, min(W_e)))
            j=nan_values[b]
            if len(j)>1:
                j=min(j)
            W_e_c=min(W_e)
        elif len(nan_values)==1:
            j=nan_values[0]
            W_e_c=np.sum(weight*fallos(decision_tree_pred(X_train[:,j],y_train,X_train[:,j],weight),y_train))
    return(j, W_e_c)

def scouting_cv(X_train, y_train, weight, orden):
    c_filas=np.array([np.zeros(len(X_train))]) 
    nan_values=np.where([np.isnan(orden)])[1]
    if len(nan_values)>1:
        for i in (nan_values):#para toda variable que quede
            c_filas=np.concatenate((c_filas,np.array([decision_tree_pred(X_train[:,i],y_train,X_train[:,i],weight)])), axis=0)
        cv=contribution_value(y_train, c_filas[1:len(X_train[0,:])+1])
        b=np.where(np.isclose(cv, max(cv)))
        i=nan_values[b]
        cv= max(cv)
        if len(i)>1:
            i=min(i)
    elif len(nan_values)==1:
        i=nan_values[0]
        cv=np.nan
        i=i
    return(int(i), cv)
    
# =============================================================================

def CGAdaboost_scouting(X_train, y_train, X_test, y_test):
    W=np.array([[]])
    cut_off=np.array([[]])
    contribution=[]
    predicted_val=np.array([[]]) 
    predicted_train=np.array([[]]) 
    orden=np.zeros(len(X_train[0,:]))
    orden.fill(np.nan)
    orden_final=[]
# =============================================================================
    W=np.concatenate((W, np.array([((1/(len(X_train)))*np.ones(len(X_train)))])), axis=1)
    k, W_e_c= scouting(X_train, y_train, W[0], orden)
    orden[k]=k
    orden_final.append(int(k))
    cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,k], y_train, W[0])]])),axis=1)
    predict_train_1=decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[0])
    contr_1=(1-tasa_error(predict_train_1,y_train))/(tasa_error(predict_train_1,y_train) +0.0000001)
    contribution.append([contr_1]) 
    W=np.concatenate((W,weight(W,1,[contr_1],y_train, X_train)), axis=0) 
    predict_val_1=np.array([decision_tree_pred(X_train[:,k], y_train, X_test[:,k],W[0])])
    predicted_val=np.concatenate((predicted_val, predict_val_1), axis=1)
    predict_train_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[0])])
    predicted_train=np.concatenate((predicted_train, predict_train_t), axis=1)
# =============================================================================
    t=1
    while t<len(X_train[0,:]):
        k, W_e_c= scouting(X_train, y_train, W[t], orden)
        orden[k]=k
        orden_final.append(int(k))
        predict_val_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_test[:,k],W[t])])
        predicted_val=np.concatenate((predicted_val, predict_val_t), axis=0)
        predict_train_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[t])])
        predicted_train=np.concatenate((predicted_train, predict_train_t), axis=0)
        contr_t=contribution_value(y_train,predicted_train)
        contribution.append(contr_t)
        W=np.concatenate((W,weight(W,t,contr_t,y_train, X_train)), axis=0)
        cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,k], y_train, W[t])]])),axis=1)
        t=t+1
    contribution_final=contribution[len(X_train[0,:])-1]/np.sum(contribution[len(X_train[0,:])-1])
    i=0
    predict_val_final=np.array([])
    while i<len(X_test):
        if np.sum((predicted_val[:,i]*contribution_final))<0:
            predict_val_final=np.concatenate((predict_val_final, np.array([-1])),axis=0)
        elif np.sum((predicted_val[:,i]*contribution_final))>0:
            predict_val_final=np.concatenate((predict_val_final, np.array([1])),axis=0)
        i=i+1
    j=0
    predict_train_final=np.array([])
    while j<len(X_train):
        if np.sum((predicted_train[:,j]*contribution_final))<0:
            predict_train_final=np.concatenate((predict_train_final, np.array([-1])),axis=0)
        elif np.sum((predicted_train[:,j]*contribution_final))>0:
            predict_train_final=np.concatenate((predict_train_final, np.array([1])),axis=0)
        j=j+1
    return orden_final,W, contribution, contribution_final, cut_off,predicted_train, predicted_val,predict_val_final,predict_train_final

# =============================================================================


def CGAdaboost_scouting_cv(X_train, y_train, X_test, y_test):
    W=np.array([[]])
    cut_off=np.array([[]])
    contribution=[]
    predicted_val=np.array([[]]) 
    predicted_train=np.array([[]]) 
    orden=np.zeros(len(X_train[0,:]))
    orden.fill(np.nan)
    orden_final=[]
# =============================================================================
    W=np.concatenate((W, np.array([((1/(len(X_train)))*np.ones(len(X_train)))])), axis=1)
    k, W_e_c= scouting_cv(X_train, y_train, W[0], orden)
    orden[k]=k
    orden_final.append(k)
    cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,k], y_train, W[0])]])),axis=1)
    predict_train_1=decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[0])
    contr_1=(1-tasa_error(predict_train_1,y_train))/(tasa_error(predict_train_1,y_train) +0.0000001)
    contribution.append([contr_1]) 
    W=np.concatenate((W,weight(W,1,[contr_1],y_train, X_train)), axis=0) 
    predict_val_1=np.array([decision_tree_pred(X_train[:,k], y_train, X_test[:,k],W[0])])
    predicted_val=np.concatenate((predicted_val, predict_val_1), axis=1)
    predict_train_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[0])])
    predicted_train=np.concatenate((predicted_train, predict_train_t), axis=1)
# =============================================================================
    t=1
    while t<len(X_train[0,:]):
        k, W_e_c= scouting_cv(X_train, y_train, W[t], orden)
        orden[k]=k
        orden_final.append(k)
        predict_val_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_test[:,k],W[t])])
        predicted_val=np.concatenate((predicted_val, predict_val_t), axis=0)
        predict_train_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[t])])
        predicted_train=np.concatenate((predicted_train, predict_train_t), axis=0)
        contr_t=contribution_value(y_train,predicted_train)
        contribution.append(contr_t)
        W=np.concatenate((W,weight(W,t,contr_t,y_train, X_train)), axis=0)
        cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,k], y_train, W[t])]])),axis=1)
        t=t+1
    contribution_final=contribution[len(X_train[0,:])-1]/np.sum(contribution[len(X_train[0,:])-1])
    i=0
    predict_val_final=np.array([])
    while i<len(X_test):
        if np.sum((predicted_val[:,i]*contribution_final))<0:
            predict_val_final=np.concatenate((predict_val_final, np.array([-1])),axis=0)
        elif np.sum((predicted_val[:,i]*contribution_final))>0:
            predict_val_final=np.concatenate((predict_val_final, np.array([1])),axis=0)
        i=i+1
    j=0
    predict_train_final=np.array([])
    while j<len(X_train):
        if np.sum((predicted_train[:,j]*contribution_final))<0:
            predict_train_final=np.concatenate((predict_train_final, np.array([-1])),axis=0)
        elif np.sum((predicted_train[:,j]*contribution_final))>0:
            predict_train_final=np.concatenate((predict_train_final, np.array([1])),axis=0)
        j=j+1
    return orden_final, W, contribution, contribution_final, cut_off,predicted_train, predicted_val,predict_val_final,predict_train_final


# =============================================================================
def CGAdaboost_sin_scouting(X_train, y_train, X_test, y_test):
    W=np.array([[]])
    cut_off=np.array([[]])
    contribution=[]
    predicted_val=np.array([[]]) 
    predicted_train=np.array([[]]) 
# =============================================================================
    W=np.concatenate((W, np.array([((1/(len(X_train)))*np.ones(len(X_train)))])), axis=1)
    cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,0], y_train, W[0])]])),axis=1)
    predict_train_1=decision_tree_pred(X_train[:,0], y_train, X_train[:,0],W[0])
    contr_1=(1-tasa_error(predict_train_1,y_train))/(tasa_error(predict_train_1,y_train) +0.0000001)
    contribution.append([contr_1])#NOTA: ojo aquí porque el contribution value se define en función del error en el TRAINING
     #SET mientras que cuando n>1 se define como medida de diversidad en el VALIDATION SET.
     #Las definiciones no están alineadas  
    W=np.concatenate((W,weight(W,1,[contr_1],y_train, X_train)), axis=0) 
    predict_val_1=np.array([decision_tree_pred(X_train[:,0], y_train, X_test[:,0],W[0])])
    predicted_val=np.concatenate((predicted_val, predict_val_1), axis=1)
    predict_train_t=np.array([decision_tree_pred(X_train[:,0], y_train, X_train[:,0],W[0])])
    predicted_train=np.concatenate((predicted_train, predict_train_t), axis=1)
# =============================================================================
    t=1
    while t<len(X_train[0,:]):
        predict_val_t=np.array([decision_tree_pred(X_train[:,t], y_train, X_test[:,t],W[t])])
        predicted_val=np.concatenate((predicted_val, predict_val_t), axis=0)
        predict_train_t=np.array([decision_tree_pred(X_train[:,t], y_train, X_train[:,t],W[t])])
        predicted_train=np.concatenate((predicted_train, predict_train_t), axis=0)
        contr_t=contribution_value(y_train,predicted_train)
        contribution.append(contr_t)
        W=np.concatenate((W,weight(W,t,contr_t,y_train, X_train)), axis=0)
        cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,t], y_train, W[t])]])),axis=1)
        t=t+1
    contribution_final=contribution[len(X_train[0,:])-1]/np.sum(contribution[len(X_train[0,:])-1])
    i=0
    predict_val_final=np.array([])
    while i<len(X_test):
        if np.sum((predicted_val[:,i]*contribution_final))<0:
            predict_val_final=np.concatenate((predict_val_final, np.array([-1])),axis=0)
        elif np.sum((predicted_val[:,i]*contribution_final))>0:
            predict_val_final=np.concatenate((predict_val_final, np.array([1])),axis=0)
        i=i+1
    j=0
    predict_train_final=np.array([])
    while j<len(X_train):
        if np.sum((predicted_train[:,j]*contribution_final))<0:
            predict_train_final=np.concatenate((predict_train_final, np.array([-1])),axis=0)
        elif np.sum((predicted_train[:,j]*contribution_final))>0:
            predict_train_final=np.concatenate((predict_train_final, np.array([1])),axis=0)
        j=j+1
    return W, contribution, contribution_final, cut_off,predicted_train, predicted_val,predict_val_final,predict_train_final

# =============================================================================
    
def weight_update(W_t, W_e_k, k,X_train, y_train):
    predict_train_t=decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W_t)
    W_nuevo=np.array([])
    e_t=(W_e_k/(np.sum(W_t)))+0.000001
    for obs in np.arange(0,len(X_train)):
        if predict_train_t[obs]!=y_train[obs]:#fallo
            W_nuevo_t=np.array([W_t[obs]*np.sqrt((1-e_t)/e_t)])
            W_nuevo=np.concatenate((W_nuevo, W_nuevo_t))
        elif predict_train_t[obs]==y_train[obs]:#acierto
            W_nuevo_t=np.array([W_t[obs]*np.sqrt((e_t)/(1-e_t))])
            W_nuevo=np.concatenate((W_nuevo, W_nuevo_t))
    return(W_nuevo/np.sum(W_nuevo))
    
    
def Adaboost(X_train, y_train, X_test, y_test):
    W=np.array([[]])
    cut_off=np.array([[]])
    contribution=[]
    orden=np.zeros(len(X_train[0,:]))
    orden_final=[]
    orden.fill(np.nan)
    predicted_val=np.array([[]])    
    predicted_train=np.array([[]]) 
    W=np.concatenate((W, np.array([((1/(len(X_train)))*np.ones(len(X_train)))])), axis=1)
    k, W_e_c= scouting(X_train, y_train,  W[0], orden)
    orden[k]=k
    orden_final.append(int(k))
    e_t=(W_e_c/(np.sum(W[0])))+0.000001
    #caso e_t=0
    contribution.append(0.5*np.log((1-e_t)/(e_t)))
    cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,k], y_train, W[0])]])),axis=1)
    predict_train_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[0])])
    predict_val_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_test[:,k],W[0])])
    predicted_val=np.concatenate((predicted_val, predict_val_t), axis=1)
    predicted_train=np.concatenate((predicted_train, predict_train_t), axis=1)
    W=np.concatenate((W, np.array([weight_update(W[0],W_e_c,k,X_train, y_train)])), axis=0)
    # =============================================================================
    t=1
    while t<len(X_train[0,:]):
        k, W_e_c= scouting(X_train, y_train,  W[t], orden)
        orden[k]=k
        orden_final.append(int(k))
        e_t=(W_e_c/np.sum(W[t]))+0.000001
        contribution.append(0.5*np.log((1-e_t)/e_t))
        cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,k], y_train, W[t])]])),axis=1)
        predict_train_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[t])])
        predicted_train=np.concatenate((predicted_train, predict_train_t), axis=0)
        predict_val_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_test[:,k],W[t])])
        predicted_val=np.concatenate((predicted_val, predict_val_t), axis=0)
        W=np.concatenate((W, np.array([weight_update(W[t],W_e_c,k,X_train, y_train)])), axis=0)
        t=t+1
    i=0
    predict_val_final=np.array([])
    while i<len(X_test):
        if np.sum((predicted_val[:,i]*contribution))<0:
            predict_val_final=np.concatenate((predict_val_final, np.array([-1])),axis=0)
        elif np.sum((predicted_val[:,i]*contribution))>0:
            predict_val_final=np.concatenate((predict_val_final, np.array([1])),axis=0)
    #        print(np.sum((predicted_val[:,i]*contribution)))
        i=i+1
    j=0
    predict_train_final=np.array([])
    while j<len(X_train):
        if np.sum((predicted_train[:,j]*contribution))<0:
            predict_train_final=np.concatenate((predict_train_final, np.array([-1])),axis=0)
        elif np.sum((predicted_train[:,j]*contribution))>0:
            predict_train_final=np.concatenate((predict_train_final, np.array([1])),axis=0)
        j=j+1
    return orden_final, contribution, predict_train_final,predict_val_final, predicted_val,predicted_train

##################################################################################

def CGAdaboost_scouting_val(X_train, y_train, X_test, y_test):
    W=np.array([[]])
    cut_off=np.array([[]])
    contribution=[]
    predicted_val=np.array([[]]) 
    predicted_train=np.array([[]]) 
    orden=np.zeros(len(X_train[0,:]))
    orden.fill(np.nan)
# =============================================================================
    W=np.concatenate((W, np.array([((1/(len(X_train)))*np.ones(len(X_train)))])), axis=1)
    k, W_e_c= scouting(X_train, y_train, W[0], orden)
    orden[k]=k
    cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,k], y_train, W[0])]])),axis=1)
    predict_train_1=decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[0])
    contr_1=(1-tasa_error(predict_train_1,y_train))/(tasa_error(predict_train_1,y_train) +0.0000001)
    contribution.append([contr_1])#NOTA: ojo aquí porque el contribution value se define en función del error en el TRAINING
     #SET mientras que cuando n>1 se define como medida de diversidad en el VALIDATION SET.
     #Las definiciones no están alineadas  
    W=np.concatenate((W,weight(W,1,[contr_1],y_train, X_train)), axis=0) 
    predict_val_1=np.array([decision_tree_pred(X_train[:,k], y_train, X_test[:,k],W[0])])
    predicted_val=np.concatenate((predicted_val, predict_val_1), axis=1)
    predict_train_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[0])])
    predicted_train=np.concatenate((predicted_train, predict_train_t), axis=1)
# =============================================================================
    t=1
    while t<len(X_train[0,:]):
        k, W_e_c= scouting(X_train, y_train, W[t], orden)
        orden[k]=k
        predict_val_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_test[:,k],W[t])])
        predicted_val=np.concatenate((predicted_val, predict_val_t), axis=0)
        predict_train_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[t])])
        predicted_train=np.concatenate((predicted_train, predict_train_t), axis=0)
        contr_t=contribution_value(y_test,predicted_val)
        contribution.append(contr_t)
        W=np.concatenate((W,weight(W,t,contr_t,y_train, X_train)), axis=0)
        cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,k], y_train, W[t])]])),axis=1)
        t=t+1
    contribution_final=contribution[len(X_train[0,:])-1]/np.sum(contribution[len(X_train[0,:])-1])
    i=0
    predict_val_final=np.array([])
    while i<len(X_test):
        if np.sum((predicted_val[:,i]*contribution_final))<0:
            predict_val_final=np.concatenate((predict_val_final, np.array([-1])),axis=0)
        elif np.sum((predicted_val[:,i]*contribution_final))>0:
            predict_val_final=np.concatenate((predict_val_final, np.array([1])),axis=0)
#        print(np.sum((predicted_val[:,i]*contribution)))
        i=i+1
    j=0
    predict_train_final=np.array([])
    while j<len(X_train):
        if np.sum((predicted_train[:,j]*contribution_final))<0:
            predict_train_final=np.concatenate((predict_train_final, np.array([-1])),axis=0)
        elif np.sum((predicted_train[:,j]*contribution_final))>0:
            predict_train_final=np.concatenate((predict_train_final, np.array([1])),axis=0)
        j=j+1
    return W, contribution, contribution_final, cut_off, predicted_val,predict_val_final,predict_train_final


# =============================================================================
    

def CGAdaboost_scouting_cv_val(X_train, y_train, X_test, y_test):
    W=np.array([[]])
    cut_off=np.array([[]])
    contribution=[]
    predicted_val=np.array([[]]) 
    predicted_train=np.array([[]]) 
    orden=np.zeros(len(X_train[0,:]))
    orden.fill(np.nan)
    orden_final=[]
# =============================================================================
    W=np.concatenate((W, np.array([((1/(len(X_train)))*np.ones(len(X_train)))])), axis=1)
    k, W_e_c= scouting_cv(X_train, y_train, W[0], orden)
    orden[k]=k
    orden_final.append(k)
    cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,k], y_train, W[0])]])),axis=1)
    predict_train_1=decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[0])
    contr_1=(1-tasa_error(predict_train_1,y_train))/(tasa_error(predict_train_1,y_train) +0.0000001)
    contribution.append([contr_1])#NOTA: ojo aquí porque el contribution value se define en función del error en el TRAINING
     #SET mientras que cuando n>1 se define como medida de diversidad en el VALIDATION SET.
     #Las definiciones no están alineadas  
    W=np.concatenate((W,weight(W,1,[contr_1],y_train, X_train)), axis=0) 
    predict_val_1=np.array([decision_tree_pred(X_train[:,k], y_train, X_test[:,k],W[0])])
    predicted_val=np.concatenate((predicted_val, predict_val_1), axis=1)
    predict_train_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[0])])
    predicted_train=np.concatenate((predicted_train, predict_train_t), axis=1)
# =============================================================================
    t=1
    while t<len(X_train[0,:]):
        k, W_e_c= scouting_cv(X_train, y_train, W[t], orden)
        orden[k]=k
        orden_final.append(k)
        predict_val_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_test[:,k],W[t])])
        predicted_val=np.concatenate((predicted_val, predict_val_t), axis=0)
        predict_train_t=np.array([decision_tree_pred(X_train[:,k], y_train, X_train[:,k],W[t])])
        predicted_train=np.concatenate((predicted_train, predict_train_t), axis=0)
        contr_t=contribution_value(y_test,predicted_val)
        contribution.append(contr_t)
        W=np.concatenate((W,weight(W,t,contr_t,y_train, X_train)), axis=0)
        cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,k], y_train, W[t])]])),axis=1)
        t=t+1
    contribution_final=contribution[len(X_train[0,:])-1]/np.sum(contribution[len(X_train[0,:])-1])
    i=0
    predict_val_final=np.array([])
    while i<len(X_test):
        if np.sum((predicted_val[:,i]*contribution_final))<0:
            predict_val_final=np.concatenate((predict_val_final, np.array([-1])),axis=0)
        elif np.sum((predicted_val[:,i]*contribution_final))>0:
            predict_val_final=np.concatenate((predict_val_final, np.array([1])),axis=0)
        i=i+1
    j=0
    predict_train_final=np.array([])
    while j<len(X_train):
        if np.sum((predicted_train[:,j]*contribution_final))<0:
            predict_train_final=np.concatenate((predict_train_final, np.array([-1])),axis=0)
        elif np.sum((predicted_train[:,j]*contribution_final))>0:
            predict_train_final=np.concatenate((predict_train_final, np.array([1])),axis=0)
        j=j+1
    return orden_final,W, contribution, contribution_final, cut_off, predicted_val,predict_val_final,predict_train_final

def CGAdaboost_sin_scouting_val(X_train, y_train, X_test, y_test):
    W=np.array([[]])
    cut_off=np.array([[]])
    contribution=[]
    predicted_val=np.array([[]]) 
    predicted_train=np.array([[]]) 
# =============================================================================
    W=np.concatenate((W, np.array([((1/(len(X_train)))*np.ones(len(X_train)))])), axis=1)
    cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,0], y_train, W[0])]])),axis=1)
    predict_val_1=decision_tree_pred(X_train[:,0], y_train, X_test[:,0],W[0])
    contr_1=(1-tasa_error(predict_val_1,y_test))/(tasa_error(predict_val_1,y_test) +0.0000001)
    contribution.append([contr_1])#NOTA: ojo aquí porque el contribution value se define en función del error en el TRAINING
     #SET mientras que cuando n>1 se define como medida de diversidad en el VALIDATION SET.
     #Las definiciones no están alineadas  
    W=np.concatenate((W,weight(W,1,[contr_1],y_train, X_train)), axis=0) 
    predict_val_1=np.array([decision_tree_pred(X_train[:,0], y_train, X_test[:,0],W[0])])
    predicted_val=np.concatenate((predicted_val, predict_val_1), axis=1)
    predict_train_t=np.array([decision_tree_pred(X_train[:,0], y_train, X_train[:,0],W[0])])
    predicted_train=np.concatenate((predicted_train, predict_train_t), axis=1)
# =============================================================================
    t=1
    while t<len(X_train[0,:]):
        predict_val_t=np.array([decision_tree_pred(X_train[:,t], y_train, X_test[:,t],W[t])])
        predicted_val=np.concatenate((predicted_val, predict_val_t), axis=0)
        predict_train_t=np.array([decision_tree_pred(X_train[:,t], y_train, X_train[:,t],W[t])])
        predicted_train=np.concatenate((predicted_train, predict_train_t), axis=0)
        contr_t=contribution_value(y_test,predicted_val)
        contribution.append(contr_t)
        W=np.concatenate((W,weight(W,t,contr_t,y_train, X_train)), axis=0)
        cut_off=np.concatenate((cut_off, np.array([[decision_tree_threshold(X_train[:,t], y_train, W[t])]])),axis=1)
        t=t+1
    contribution_final=contribution[len(X_train[0,:])-1]/np.sum(contribution[len(X_train[0,:])-1])
    i=0
    predict_val_final=np.array([])
    while i<len(X_test):
        if np.sum((predicted_val[:,i]*contribution_final))<0:
            predict_val_final=np.concatenate((predict_val_final, np.array([-1])),axis=0)
        elif np.sum((predicted_val[:,i]*contribution_final))>0:
            predict_val_final=np.concatenate((predict_val_final, np.array([1])),axis=0)
#        print(np.sum((predicted_val[:,i]*contribution)))
        i=i+1
    j=0
    predict_train_final=np.array([])
    while j<len(X_train):
        if np.sum((predicted_train[:,j]*contribution_final))<0:
            predict_train_final=np.concatenate((predict_train_final, np.array([-1])),axis=0)
        elif np.sum((predicted_train[:,j]*contribution_final))>0:
            predict_train_final=np.concatenate((predict_train_final, np.array([1])),axis=0)
        j=j+1
    return W, contribution, contribution_final, cut_off, predicted_val,predict_val_final,predict_train_final

# =============================================================================
    
def evolucion(X_test, y_test, predicted_val, contribution):
    j=0
    pred_evol=np.array([[]])
    while j<len(X_test[0,:]):
        i=0
        predict_val_pasos=np.array([])
        while i<len(X_test):
            if np.sum((predicted_val[:,i][0:j+1]*contribution[j]))<0:
                predict_val_pasos=np.concatenate((predict_val_pasos, np.array([-1])),axis=0)
            elif np.sum((predicted_val[:,i][0:j+1]*contribution[j]))>=0:
                predict_val_pasos=np.concatenate((predict_val_pasos, np.array([1])),axis=0)
            i=i+1
        aciertos_pasos=[aciertos_suma(predict_val_pasos, y_test)/len(y_test)]
        pred_evol=np.concatenate((pred_evol,np.array([aciertos_pasos])), axis=1)
        j=j+1
    return(pred_evol)

#-----------------contribution value-------------------------------------------
contribution_value=contribution_value_DFM
#------------------------------------------------------------------------------
######contribution train#######
#scouting normal 
orden_final1,W1, contribution1, contribution_final1, cut_off1, predicted_train1, predicted_val1,predict_val_final1,predict_train_final1=CGAdaboost_scouting(X_train, y_train, X_test, y_test)
accuracy_sc_val=aciertos_suma(predict_val_final1, y_test)/len(y_test)
print("scouting normal val",accuracy_sc_val)
accuracy_sc_train=aciertos_suma(predict_train_final1, y_train)/len(y_train)
print("scouting normal train",accuracy_sc_train)

#scouting cv
orden_final_cv, W_cv, contribution_cv, contribution_final_cv, cut_off_cv,predicted_train_cv, predicted_val_cv,predict_val_final_cv,predict_train_final_cv=CGAdaboost_scouting_cv(X_train, y_train, X_test, y_test)
accuracy_val_sc_cv=aciertos_suma(predict_val_final_cv, y_test)/len(y_test)
print("scouting cv val",accuracy_val_sc_cv)
accuracy_train_sc_cv=aciertos_suma(predict_train_final_cv, y_train)/len(y_train)
print("scouting cv train",accuracy_train_sc_cv)

#sin scouting
W2, contribution2, contribution_final2, cut_off2, predicted_train2, predicted_val2,predict_val_final2,predict_train_final2=CGAdaboost_sin_scouting(X_train, y_train, X_test, y_test)
accuracy_val_sin=aciertos_suma(predict_val_final2, y_test)/len(y_test)
print("sin scouting val",accuracy_val_sin)
accuracy_train_sin=aciertos_suma(predict_train_final2, y_train)/len(y_train)
print("sin scouting train",accuracy_train_sin)

#adaboost original
orden_final3, contribution3, predict_train_final3,predict_val_final3, predicted_val3,predicted_train3=Adaboost(X_train, y_train, X_test, y_test)   
accuracy_val=aciertos_suma(predict_val_final3, y_test)/len(y_test)
print("adaboost orig val", accuracy_val)
accuracy_train=aciertos_suma(predict_train_final3, y_train)/len(y_train)
print("adaboost original train",accuracy_train)

#accuracy_val_tree=[]
#for var in np.arange(0,len(X_train[0,:])):   
#    accuracy_val_tree.append(aciertos_suma(predicted_val3[var], y_test)/len(y_test))
#max(accuracy_val_tree)

#-----------------contribution value-------------------------------------------
contribution_value=contribution_value_DFM
#------------------------------------------------------------------------------
#######contribution val########
print("------contribution value calculado con DVAL-------")
#scouting normal 
W_sc, contribution_sc, contribution_final_sc, cut_off_sc, predicted_val_sc,predict_val_final_sc,predict_train_final_sc=CGAdaboost_scouting_val(X_train, y_train, X_test, y_test)
accuracy_sc_val_val=aciertos_suma(predict_val_final_sc, y_test)/len(y_test)
print("scouting normal val",accuracy_sc_val_val)
accuracy_sc_train_val=aciertos_suma(predict_train_final_sc, y_train)/len(y_train)
print("scouting normal train",accuracy_sc_train_val)

#scouting cv
W_cv_val, contribution_cv_val, contribution_final_cv_val, cut_off_cv_val, predicted_train_cv_val,predicted_val_cv_val,predict_val_final_cv_val,predict_train_final_cv_val=CGAdaboost_scouting_cv_val(X_train, y_train, X_test, y_test)
accuracy_sc_cv_val_val=aciertos_suma(predict_val_final_cv_val, y_test)/len(y_test)
print("scouting cv val",accuracy_sc_cv_val_val)
accuracy_sc_cv_train_val=aciertos_suma(predict_train_final_cv_val, y_train)/len(y_train)
print("scouting cv train",accuracy_sc_cv_train_val)

#sin scouting
W_cg, contribution2_cg, contribution_final_cg, cut_off_cg, predicted_val_cg,predict_val_final_cg,predict_train_final_cg =CGAdaboost_sin_scouting_val(X_train, y_train, X_test, y_test)
accuracy_val_sin_val=aciertos_suma(predict_val_final_cg, y_test)/len(y_test)
print("sin scouting val",accuracy_val_sin_val)
accuracy_train_sin_val=aciertos_suma(predict_train_final_cg, y_train)/len(y_train)
print("sin scouting train",accuracy_train_sin_val)

#-----------------contribution value-------------------------------------------
contribution_value=contribution_value_CC
#------------------------------------------------------------------------------
print('CC')
print("CV TRAIN")
#sin scouting CV TRAIN 
W2_CC, contribution2_CC, contribution_final2_CC, cut_off2_CC, predicted_train2_CC, predicted_val2_CC,predict_val_final2_CC,predict_train_final2_CC=CGAdaboost_sin_scouting(X_train, y_train, X_test, y_test)
accuracy_val_sin_cc_train=aciertos_suma(predict_val_final2_CC, y_test)/len(y_test)
print("sin scouting val",accuracy_val_sin_cc_train)
accuracy_train_sin_cc_train=aciertos_suma(predict_train_final2_CC, y_train)/len(y_train)
print("sin scouting train",accuracy_train_sin_cc_train)

print("CV  VAL")
#sin scouting CV VALID
W_cg, contribution2_cg, contribution_final_cg, cut_off_cg, predicted_val_cg,predict_val_final_cg,predict_train_final_cg =CGAdaboost_sin_scouting_val(X_train, y_train, X_test, y_test)
accuracy_val_sin_cc_val=aciertos_suma(predict_val_final_cg, y_test)/len(y_test)
print("sin scouting val",accuracy_val_sin_cc_val)
accuracy_train_sin_cc_val=aciertos_suma(predict_train_final_cg, y_train)/len(y_train)
print("sin scouting train",accuracy_train_sin_cc_val)


#-----------------contribution value-------------------------------------------
contribution_value=contribution_value_Q
#-------------------------------------------------------------------------------
print('Q')
print("CV TRAIN")
#sin scouting CV TRAIN 
W2_Q, contribution2_Q, contribution_final2_Q, cut_off2_Q, predicted_train2_Q, predicted_val2_Q,predict_val_final2_Q,predict_train_final2_Q=CGAdaboost_sin_scouting(X_train, y_train, X_test, y_test)
accuracy_val_sin_q_val_train=aciertos_suma(predict_val_final2_Q, y_test)/len(y_test)
print("sin scouting val",accuracy_val_sin_q_val_train)
accuracy_train_sin_q_train=aciertos_suma(predict_train_final2_Q, y_train)/len(y_train)
print("sin scouting train",accuracy_train_sin_q_train)

print("CV  VAL")
#sin scouting CV VALID
W_cg, contribution2_cg, contribution_final_cg, cut_off_cg, predicted_val_cg,predict_val_final_cg,predict_train_final_cg =CGAdaboost_sin_scouting_val(X_train, y_train, X_test, y_test)
accuracy_val_sin_q_val=aciertos_suma(predict_val_final_cg, y_test)/len(y_test)
print("sin scouting val",accuracy_val_sin_q_val)
accuracy_train_sin_q_val=aciertos_suma(predict_train_final_cg, y_train)/len(y_train)
print("sin scouting train",accuracy_train_sin_q_val)



val=[accuracy_val_sin_cc_train,accuracy_val_sin, accuracy_val_sin_q_val_train, accuracy_sc_val, accuracy_val_sc_cv]
train=[accuracy_train_sin_cc_train,accuracy_train_sin,accuracy_val_sin_q_train,accuracy_sc_train,accuracy_train_sc_cv]

df_results= pd.DataFrame(np.array([val, train]))

df_results.to_csv(r'C:/Users/belen/OneDrive/Escritorio/TFG_MATES/res_1.csv')

df_results.to_excel(r'C:/Users/belen/OneDrive/Escritorio/TFG_MATES/res_1.xlsx')

####       GRÁFICOS EVOLUCIÓN- TODO JUNTO    #####
plt.plot(np.arange(len(X_train[0,:])),evolucion(X_train, y_train, predicted_train1,contribution1)[0], c='y')
patch_0 = mpatches.Patch(color='yellow', label='scouting')
plt.plot(np.arange(len(X_train[0,:])),evolucion(X_train, y_train, predicted_train_cv,contribution_cv)[0],c='b')
patch_1 = mpatches.Patch(color='blue', label='scouting cv')
plt.plot(np.arange(len(X_train[0,:])),evolucion(X_train, y_train, predicted_train2,contribution2)[0], c='g')
patch_2 = mpatches.Patch(color='green', label='sin scouting')
plt.plot(np.arange(len(X_train[0,:])),evolucion(X_train, y_train, predicted_train3,contribution3)[0], c='m')
patch_3 = mpatches.Patch(color='magenta', label='Adaboost orig')
plt.title("Evolución de la tasa de aciertos según la iteración")
plt.legend(handles=[patch_0,patch_1, patch_2, patch_3])
plt.xlabel("iteración")
plt.ylabel("tasa acierto")
