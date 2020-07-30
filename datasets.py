# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:42:19 2019

@author: belen
"""


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
from sklearn import tree


# =============================================================================
def cambio_array(panda):
    #cambio de formato de pandas a np.array   
    X_obs=np.array([np.zeros(panda.shape[1])])
    i=0
    while i<panda.shape[0]:
        X_train_all_vars=np.array([[]])
        j=0
        while j<panda.shape[1]:
            X_train_all_vars=np.concatenate((X_train_all_vars, np.array([[panda.iloc[i,j]]])),axis=1)
            j=j+1
        X_obs=np.concatenate((X_obs,X_train_all_vars),axis=0)
        i=i+1
    return(np.delete(X_obs, 0,0))

# =============================================================================    

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================    

#
#p0=predict_feature(X_train[:,0], y_train, X_test[:,0], W0)
#print(sum(p0==1))
#print(sum(p0==-1))
#best0=posibles_splits_feature(X_train[:,0],y_train, W0)
#print(best0[0][1])
#print(aciertos_suma(p0,y_test)/len(y_test))

# -*- coding: utf-8 -*-

# =============================================================================
# =============================================================================
#HABERMAN
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/haberman.data')
df['T']=np.where(df['T']==1, -1, 1) #-1: the patient survived 5 years or longer, 1: the patient died within 5 year


target=df['T']
data=df.drop('T', axis=1)
#no hay missings
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda)
X_test=cambio_array(X_test_panda)
y_train=np.array([y_train_panda])[0]
y_test=np.array([y_test_panda])[0]
# =============================================================================
#BREAST-DIAGNOSTIC, T: (2 for benign, 4 for malignant)
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/breast_cancer_wisconsin.data')
df['target']=np.where(df['T']==2, -1, 1) #-1: beingn, 1: malignant
df=df.drop('T', axis=1)
df=df.drop('id', axis=1)

#MISSINGS

colnames=list(df.columns)
for nombres in colnames:
    df = df[df[nombres] != "?"]

#df=df[df.Clump_Thickness!='?'][df.Uniformity_Cell_Size!='?'][df.Uniformity_Cell_Shape!='?'][df.Marginal_Adhesion!='?'][df.Single_Epithelial_Cell_Size!='?'][df.Bare_Nuclei!='?'][df.Bland_Chromatin!='?'][df.Normal_Nucleoli!='?'][df.Mitoses!='?']

target=df['target']
data=df.drop('target', axis=1)

X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)
# =============================================================================
#HEPATITIS
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/hepatitis.data')
df['T']=np.where(df['T']==1, -1, 1) #-1: die, 1: live

#MISSINGS
df=df[df.STEROID!='?'][df.FATIGUE!='?'][df.MALAISE!='?'][df.ANOREXIA!='?'][df.LIVER_BIG!='?'][df.SPLEEN_PALPABLE!='?'][df.LIVER_FIRM!='?'][df.SPIDERS!='?'][df.ASCITES!='?'][df.VARICES!='?'][df.BILIRUBIN!='?'][df.ALK_PHOSPHATE!='?'][df.SGOT!='?'][df.ALBUMIN!='?'][df.PROTIME!='?']
target=df['T']
data=df.drop('T', axis=1)


X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)

# =============================================================================
#ionosphere
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/ionosphere.data')
df['target']=np.where(df['35']=='g', -1, 1) #-1: good, 1: bad year
df=df.drop('35', axis=1)
target=df['target']
data=df.drop('target', axis=1)
#no hay missings
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda)
X_test=cambio_array(X_test_panda)
y_train=np.array([y_train_panda])[0]
y_test=np.array([y_test_panda])[0]

# =============================================================================
#SPAM

df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/spambase.data') 
#-1: non-spam, 1: spam
df['target']=np.where(df['58']==0, -1, 1)
target=df['target']
df=df.drop('target', axis=1)
data=df.drop('58', axis=1)
#no hay missings
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda)
X_test=cambio_array(X_test_panda)
y_train=np.array([y_train_panda])[0]
y_test=np.array([y_test_panda])[0]

# =============================================================================
# =============================================================================
#SPECT
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/SPECT.test') 
df['T']=np.where(df['T']==0, -1, 1)

target=df['T']
data=df.drop('T', axis=1)
#no hay missings
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda)
X_test=cambio_array(X_test_panda)
y_train=np.array([y_train_panda])[0]
y_test=np.array([y_test_panda])[0]
# =============================================================================
#SPECTF
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/SPECTF.train') 
df['T']=np.where(df['T']==0, -1, 1)

target=df['T']
data=df.drop('T', axis=1)
#no hay missings
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda)
X_test=cambio_array(X_test_panda)
y_train=np.array([y_train_panda])[0]
y_test=np.array([y_test_panda])[0]
# =============================================================================
#tictactoe
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/tic-tac-toe.data') 
df['T']=np.where(df['T']=="positive", -1, 1) #-1: win for x// 1: lose for x
df['1']=np.where(df['1']=="x", 1, 2)
df['2']=np.where(df['2']=="x", 1, 2)#categorizo de forma numérica
df['3']=np.where(df['3']=="x", 1, 2)
df['4']=np.where(df['4']=="x", 1, 2)
df['5']=np.where(df['5']=="x", 1, 2)
df['6']=np.where(df['6']=="x", 1, 2)
df['7']=np.where(df['7']=="x", 1, 2)
df['8']=np.where(df['8']=="x", 1, 2)
df['9']=np.where(df['9']=="x", 1, 2)
target=df['T']
data=df.drop('T', axis=1)
#no hay missings
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda)
X_test=cambio_array(X_test_panda)
y_train=np.array([y_train_panda])[0]
y_test=np.array([y_test_panda])[0]
# =============================================================================
#VOTING

df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/house-votes-84.data')
df['T']=np.where(df['T']=="republican", -1, 1) #-1: republian, 1: democrat

#MISSINGS
df=df[df['1']!='?'][df['2']!='?'][df['3']!='?'][df['4']!='5'][df['6']!='?'][df['7']!='?'][df['8']!='?'][df['9']!='?'][df['10']!='?'][df['11']!='?'][df['12']!='?'][df['13']!='?'][df['14']!='?'][df['15']!='?'][df['16']!='?']
# cat numérica1)
df['1']=np.where(df['1']=="y", 0, 1)
df['2']=np.where(df['2']=="y", 0, 1)
df['3']=np.where(df['3']=="y", 0, 1)
df['4']=np.where(df['4']=="y", 0, 1)
df['5']=np.where(df['5']=="y", 0, 1)
df['6']=np.where(df['6']=="y", 0, 1)
df['7']=np.where(df['7']=="y", 0, 1)
df['8']=np.where(df['8']=="y", 0, 1)
df['9']=np.where(df['9']=="y", 0, 1)
df['10']=np.where(df['10']=="y", 0, 1)
df['11']=np.where(df['11']=="y", 0, 1)
df['12']=np.where(df['12']=="y", 0, 1)
df['13']=np.where(df['13']=="y", 0, 1)
df['14']=np.where(df['14']=="y", 0, 1)
df['15']=np.where(df['15']=="y", 0, 1)
df['16']=np.where(df['16']=="y", 0, 1)
target=df['T']
data=df.drop('T', axis=1)

X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)

# =============================================================================
# ringnorm
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/ringnormcsv.csv', sep=';')
df['T']=np.where(df['T']==0, -1, 1)

target=df['T']
data=df.drop('T', axis=1)
#no hay missings
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda)
X_test=cambio_array(X_test_panda)
y_train=np.array([y_train_panda])[0]
y_test=np.array([y_test_panda])[0]
# =============================================================================
# twonorm
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/twonormcsv.csv', sep=';')
df['T']=np.where(df['T']==0, -1, 1)

target=df['T']
data=df.drop('T', axis=1)
#no hay missings
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda)
X_test=cambio_array(X_test_panda)
y_train=np.array([y_train_panda])[0]
y_test=np.array([y_test_panda])[0]
# =============================================================================
#titanic
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/titaniccsv.csv', sep=';')

df['T']=np.where(df['T']==0, -1, 1) #-1: no sobrevive; 1  sobrevive

target=df['T']
data=df.drop('T', axis=1)
#no hay missings
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)

X_train=cambio_array(X_train_panda)
X_test=cambio_array(X_test_panda)
y_train=np.array([y_train_panda])[0]
y_test=np.array([y_test_panda])[0]
# =============================================================================
# =============================================================================
#
#df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/otros_datasets/audit_risk.csv')
#df['Risk']=np.where(df['Risk']==0, -1, 1) #-1: , 1: fraudulent?
#
#target=df['Risk']
#data=df.drop('Risk', axis=1)
#
#X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
#data, target, stratify=target, random_state=42)
#
#X_train=cambio_array(X_train_panda).astype(float)
#X_test=cambio_array(X_test_panda).astype(float)
#y_train=np.array([y_train_panda])[0].astype(float)
#y_test=np.array([y_test_panda])[0].astype(float)

# =============================================================================
#cesarea
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/otros_datasets/caesarian.arff')
df['T']=np.where(df['T']==0, -1, 1) #-1: no hay cesarea, 1: si hay
target=df['T']
data=df.drop('T', axis=1)
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)
X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)

# =============================================================================
#cancer cervical
df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/otros_datasets/risk_factors_cervical_cancer.csv')
df=df[df['Age']!='?'][df['Number of sexual partners']!='?'][df['First sexual intercourse']!='?'][df['Num of pregnancies']!='?'][df['Smokes']!='?'][df['Smokes (years)']!='?'][df['Smokes (packs/year)']!='?'][df['Hormonal Contraceptives']!='?'][df['Hormonal Contraceptives (years)']!='?'][df['IUD']!='?'][df['IUD (years)']!='?'][df['STDs']!='?'][df['STDs (number)']!='?'][df['STDs:condylomatosis']!='?'][df['STDs:cervical condylomatosis']!='?'][df['STDs:vaginal condylomatosis']!='?'][df['STDs:vulvo-perineal condylomatosis']!='?'][df['STDs:syphilis']!='?'][df['STDs:pelvic inflammatory disease']!='?'][df['STDs:genital herpes']!='?'][df['STDs:molluscum contagiosum']!='?'][df['STDs:AIDS']!='?'][df['STDs:HIV']!='?'][df['STDs:Hepatitis B']!='?'][df['STDs:HPV']!='?'][df['STDs: Number of diagnosis']!='?'][df['STDs: Time since first diagnosis']!='?'][df['STDs: Time since last diagnosis']!='?'][df['Dx:Cancer']!='?'][df['Dx:CIN']!='?'][df['Dx:HPV']!='?'][df['Dx']!='?'][df['Hinselmann']!='?'][df['Schiller']!='?'][df['Citology']!='?'][df['Biopsy']!='?']

df['Biopsy']=np.where(df['Biopsy']==0, -1, 1) 
target=df['Biopsy']
data=df.drop('Biopsy', axis=1)
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)
X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)

# =============================================================================
##defaults
df = pd.read_excel('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/otros_datasets/default of credit card clients.xls')

df['Y']=np.where(df['Y']==0, -1, 1) 
target=df['Y']
data=df.drop('Y', axis=1)
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)
X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)
# =============================================================================
##coche


df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/otros_datasets/aps_failure_training_set.csv')

df['Y']=np.where(df['class']=="neg", -1, 1) 
df=df.drop('class', axis=1)

colnames=list(df.columns)
for nombres in colnames:
    df = df[df[nombres] != "na"]

target=df['Y']
data=df.drop('Y', axis=1)
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)
X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)

# =============================================================================

df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/otros_datasets/dota2Train.csv')

target=df['T']
data=df.drop('T', axis=1)
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)
X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)
# =============================================================================

df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/otros_datasets/EEG Eye State.arff')


df[' T']=np.where(df[' T']==0, -1, 1)

target=df[' T']
data=df.drop(' T', axis=1)
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)
X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)


# =============================================================================

df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/otros_datasets/magic04.data')

df['T']=np.where(df['T']=='g', -1, 1)

target=df['T']
data=df.drop('T', axis=1)
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)
X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)

# =============================================================================

df = pd.read_csv('C:/Users/belen/OneDrive/Escritorio/TFG_MATES/otros_datasets/phishing.arff')


target=df['T']
data=df.drop('T', axis=1)
X_train_panda, X_test_panda, y_train_panda, y_test_panda = train_test_split(
data, target, stratify=target, random_state=42)
X_train=cambio_array(X_train_panda).astype(float)
X_test=cambio_array(X_test_panda).astype(float)
y_train=np.array([y_train_panda])[0].astype(float)
y_test=np.array([y_test_panda])[0].astype(float)


# =============================================================================
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)
print("Keys of cancer_dataset: \n{}".format(cancer.keys()))
