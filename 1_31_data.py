from glob import glob
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#정의된함수
def opfn(x,y):
    return glob("C:/daejeon_data/daejeon_data/"+str(x)+"*/*"+str(y)+"kw*/*/4*/*.csv")

def nofn(x,y):
    return glob("C:/daejeon_data/daejeon_data/"+str(x)+"*/*"+str(y)+"kw*/*/5*/*.csv")

def train_current_data_r(x,y):
    a=opfn(x,y)
    colnames=['csv_r','csv_s','csv_t']
    b=[]
    for i in range(0,len(a)):
        data=pd.read_csv(a[i],names=colnames)
        CSV=data.loc[range(15,len(data)),'csv_r']
        CSV=CSV.values.tolist()
        CSV=list(map(float,CSV))
        for k in range(0,len(CSV)-59):
            b.append(CSV[k:60+k])
    c=nofn(x,y)
    for i in range(0,len(c)):
        data=pd.read_csv(c[i],names=colnames)
        CSV=data.loc[range(15,len(data)),'csv_r']
        CSV=CSV.values.tolist()
        CSV=list(map(float,CSV))
        for k in range(0,len(CSV)-59):
            b.append(CSV[k:60+k])
    b=np.reshape(b,(-1,60))
    return b

def test_current_data(x,y):
    a=opfn(x,y)
    colnames=['csv_r','csv_s','csv_t']
    b=[]
    for i in range(0,len(a)):
        data=pd.read_csv(a[i],names=colnames)
        CSV=data.loc[range(15,len(data)),:]
        CSV=CSV.values.tolist()
        CSV=sum(CSV,[])
        CSV=list(map(float,CSV))
        for k in range(0,int(len(CSV)/3)-59):
            b.append(CSV[3*k:180+3*k])
    c=nofn(x,y)
    for i in range(0,len(c)):
        data=pd.read_csv(c[i],names=colnames)
        CSV=data.loc[range(15,len(data)),:]
        CSV=CSV.values.tolist()
        CSV=sum(CSV,[])
        CSV=list(map(float,CSV))
        for k in range(0,int(len(CSV)/3)-59):
            b.append(CSV[3*k:180+3*k])
    b=np.reshape(b,(len(c)*5941*2,180))
    print(len(CSV))
    return b

def test_kimm_data(x,y):
    a=opfn(x,y)
    b=[]
    colnames=[0,1]
    for i in range(0,len(a)):
        data=pd.read_csv(a[i],names=colnames)
        CSV=data.loc[range(4,len(data)),0]
        CSV=CSV.values.tolist()
        CSV=list(map(float,CSV))
        for k in range(0,len(CSV)-59):
            b.append(CSV[k:60+k])
    c=nofn(x,y)
    for i in range(0,len(c)):
        data=pd.read_csv(c[i],names=colnames)
        CSV=data.loc[range(4,len(data)),0]
        CSV=CSV.values.tolist()
        CSV=list(map(float,CSV))
        for k in range(0,len(CSV)-59):
            b.append(CSV[k:60+k])
    b=np.reshape(b,(-1,60))
    return b

def test_vibration_data(x,y):
    a=opfn(x,y)
    b=[]
    colnames=[0,1]
    for i in range(0,len(a)):
        data=pd.read_csv(a[i],names=colnames)
        CSV=data.loc[range(4,len(data)),0]
        CSV=CSV.values.tolist()
        CSV=list(map(float,CSV))
        for k in range(0,len(CSV)-59):
            b.append(CSV[k:60+k])
    c=nofn(x,y)
    for i in range(0,len(c)):
        data=pd.read_csv(c[i],names=colnames)
        CSV=data.loc[range(4,len(data)),0]
        CSV=CSV.values.tolist()
        CSV=list(map(float,CSV))
        for k in range(0,len(CSV)-59):
            b.append(CSV[k:60+k])
    b=np.reshape(b,(-1,60))
    return b

def test_current_lable(x):
    if x==2.2:
        L = ['normal' for i in range(178230)]+['bearing' for i in range(35646)]+['belt' for i in range(71292)] + ['misalignment' for i in range(35646)]+['unbalance' for i in range(35646)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==3.7:
        L = ['normal' for i in range(5941*24)]+['bearing' for i in range(5941*6)] + ['misalignment' for i in range(5941*12)]+['unbalance' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==5.5:
        L = ['normal' for i in range(5941*24)]+['bearing' for i in range(5941*6)]+['belt' for i in range(5941*6)] + ['misalignment' for i in range(5941*6)]+['unbalance' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==7.5:
        L = ['normal' for i in range(5941*18)]+['bearing' for i in range(5941*6)]+['belt' for i in range(5941*6)] + ['misalignment' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==11:
        L = ['normal' for i in range(5941*24)]+['bearing' for i in range(5941*6)]+['belt' for i in range(5941*6)] + ['misalignment' for i in range(5941*6)]+['unbalance' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==15:
        L = ['normal' for i in range(5941*18)]+['bearing' for i in range(5941*6)]+['belt' for i in range(5941*6)] +['unbalance' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==18.5:
        L = ['normal' for i in range(5941*12)]+['bearing' for i in range(5941*6)]+['belt' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==22:
        L = ['normal' for i in range(5941*18)]+['belt' for i in range(5941*6)] + ['misalignment' for i in range(5941*6)]+['unbalance' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==30:
        L = ['normal' for i in range(5941*6*1)]+ ['misalignment' for i in range(5941*6*1)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==37:
        L = ['normal' for i in range(5941*6*1)]+ ['misalignment' for i in range(5941*6*1)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==55:
        L = ['normal' for i in range(5941*6*2)]+['belt' for i in range(5941*6*1)] + ['unbalance' for i in range(5941*6*1)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L#이때 x는 소비전력기준

def test_kimm_lable(x):
    if x==2.2:
        L = ['normal' for i in range(11941*6*5)]+['bearing' for i in range(11941*6)]+['belt' for i in range(11941*6*2)] + ['misalignment' for i in range(11941*6)]+['unbalance' for i in range(11941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==3.7:
        L = ['normal' for i in range(11941*18)]+['bearing' for i in range(11941*6)]+['misalignment' for i in range(11941*6)] +['unbalance' for i in range(11941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==3.75:
        L = ['normal' for i in range(11941*6)]+['misalignment' for i in range(11941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)    
        return L
    elif x==5.5:
        L = ['normal' for i in range(11941*24)]+['bearing' for i in range(11941*6)]+['belt' for i in range(11941*6)] + ['misalignment' for i in range(11941*6)]+['unbalance' for i in range(11941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==7.5:
        L = ['normal' for i in range(11941*18)]+['bearing' for i in range(11941*6)]+['belt' for i in range(11941*6)] + ['misalignment' for i in range(11941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==11:
        L = ['normal' for i in range(11941*24)]+['bearing' for i in range(11941*6)]+['belt' for i in range(11941*6)] + ['misalignment' for i in range(11941*6)]+['unbalance' for i in range(11941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==15:
        L = ['normal' for i in range(11941*18)]+['bearing' for i in range(11941*6)]+['belt' for i in range(11941*6)] +['unbalance' for i in range(11941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==18.5:
        L = ['normal' for i in range(11941*12)]+['bearing' for i in range(11941*6)]+['belt' for i in range(11941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==22:
        L = ['normal' for i in range(11941*18)]+['belt' for i in range(11941*6)] + ['misalignment' for i in range(11941*6)]+['unbalance' for i in range(11941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==30:
        L = ['normal' for i in range(11941*6*1)]+ ['misalignment' for i in range(11941*6*1)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==37:
        L = ['normal' for i in range(11941*6*1)]+ ['misalignment' for i in range(11941*6*1)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==55:
        L = ['normal' for i in range(11941*6*2)]+['belt' for i in range(11941*6*1)] + ['unbalance' for i in range(11941*6*1)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L#이때 x는 소비전력기준

def test_vibration_lable(x):
    if x==2.2:
        L = ['normal' for i in range(5941*6*5)]+['bearing' for i in range(5941*6*1)]+['belt' for i in range(5941*6*2)] + ['misalignment' for i in range(5941*6*1)]+['unbalance' for i in range(5941*6*1)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==3.7:
        L = ['normal' for i in range(5941*6*2)]+['bearing' for i in range(5941*6*1)] + ['misalignment' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==3.75:
        L = ['normal' for i in range(5941*6)] + ['misalignment' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==5.5:
        L = ['normal' for i in range(5941*6*4)]+['bearing' for i in range(5941*6)]+['belt' for i in range(2941*6)] + ['misalignment' for i in range(2941*6)]+['unbalance' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==7.5:
        L = ['normal' for i in range(5941*6*4)]+['bearing' for i in range(5941*6)]+['belt' for i in range(5941*6)] + ['misalignment' for i in range(5941*6)]+['unbalance' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==11:
        L = ['normal' for i in range(5941*24)]+['bearing' for i in range(5941*6)]+['belt' for i in range(5941*6)] + ['misalignment' for i in range(5941*6)]+['unbalance' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==15:
        L = ['normal' for i in range(5941*18)]+['bearing' for i in range(5941*6)]+['belt' for i in range(5941*6)] +['unbalance' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==18.5:
        L = ['normal' for i in range(5941*12)]+['bearing' for i in range(5941*6)]+['belt' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==22:
        L = ['normal' for i in range(5941*18)]+['belt' for i in range(5941*6)] + ['misalignment' for i in range(5941*6)]+['unbalance' for i in range(5941*6)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==30:
        L = ['normal' for i in range(5941*6*1)]+ ['misalignment' for i in range(5941*6*1)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==37:
        L = ['normal' for i in range(5941*6*1)]+ ['misalignment' for i in range(5941*6*1)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L
    elif x==55:
        L = ['normal' for i in range(5941*6*2)]+['belt' for i in range(5941*6*1)] + ['unbalance' for i in range(5941*6*1)]
        L_arr = np.array(L)
        L = L_arr.reshape(len(L_arr),1)
        return L#이때 x는 소비전력기준

#submission불러오기
sub=pd.read_csv('C:\\dataSet_Submission_Re\\Submission_Commit.csv')
for i in range(0,len(sub)):
    A=sub.Category[i]
    B=sub.Motor[i]
    if B==int(B):
        B=int(B)
    else:
        B

    print(i)

    if A=='Kimm':
        I=str(i+1).zfill(3)
        a=glob("C:\dataSet_Submission_Re\*"+I+".csv")
        data=pd.read_csv(a[0])
        CSV=data.loc[range(0,len(data)),'value']
        CSV=CSV.values.tolist()
        testdata=list(map(float,CSV))
        testdata=np.reshape(testdata,(1,-1))
        knn_kimm=KNeighborsClassifier(n_neighbors=1)
        knn_kimm.fit(test_kimm_data(A,B), test_kimm_lable(B))
        prediction=knn_kimm.predict(testdata)
        sub.Label[i]=prediction[0]


    elif A=='Vibration':
        I=str(i+1).zfill(3)
        a=glob("C:\dataSet_Submission_Re\*"+I+".csv")
        data=pd.read_csv(a[0])
        CSV=data.loc[range(0,len(data)),'value']
        CSV=CSV.values.tolist()
        testdata=list(map(float,CSV))
        testdata=np.reshape(testdata,(1,-1))

        knn_vib=KNeighborsClassifier(n_neighbors=1)
        knn_vib.fit(test_vibration_data(A,B), test_vibration_lable(B))
        prediction=knn_vib.predict(testdata)
        sub.Label[i]=prediction[0]


    elif A=='Current':
        I=str(i+1).zfill(3)
        a=glob("C:\dataSet_Submission_Re\*"+I+".csv")
        if I == str(142):
            data=pd.read_csv(a[0])
            CSV=data.loc[range(0,len(data)),'value']
            CSV=CSV.values.tolist()
            testdata=list(map(float,CSV))
            testdata=np.reshape(testdata,(1,-1))

            knn_current=KNeighborsClassifier(n_neighbors=1)
            knn_current.fit(train_current_data_r(A,B), test_current_lable(B))
            prediction=knn_current.predict(testdata)
            sub.Label[i]=prediction[0]

        else:
            colnames =[0,1,2,3]
            data=pd.read_csv(a[0],names=colnames)
            CSV=data.loc[range(1,len(data)),range(1,4)]
            CSV=CSV.values.tolist()
            CSV=sum(CSV,[])
            testdata=list(map(float,CSV))
            testdata=np.reshape(testdata,(1,-1))

            knn_current=KNeighborsClassifier(n_neighbors=1)
            knn_current.fit(test_current_data(A,B), test_current_lable(B))
            prediction=knn_current.predict(testdata)
            sub.Label[i]=prediction[0]
    print(sub)

sub.to_csv("tested_submission.csv",sep=",",index=False)
test_end=pd.read_csv("tested_submission.csv")
print(test_end)