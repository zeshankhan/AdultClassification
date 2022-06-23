from dataset import get_dataset
import pandas as pd
import numpy as np
from sklearn import preprocessing

def normalized(df,col):
    df[col]=(df[col]-min(df[col]))/(max(df[col])-min(df[col]))
    return df

def onehot(df,col):
    oneh=np.zeros((len(df[col]),len(set(df[col]))),np.uint8)
    for i in range(oneh.shape[0]):
        oneh[i,df[col][i]]=1
    #print(onehot)
    #print(df.shape)
    oneh1=pd.DataFrame(oneh)
    oneh1.columns=[col+str(i) for i in range(oneh1.shape[1])]
    df=df.join(oneh1)
    df.drop(col, axis=1, inplace=True,errors='ignore')
    #print("done")
    return df

def onehot_enc(df1,df2,col):
    colms=max(len(set(df1[col])),len(set(df2[col])))
    
    onehot=np.zeros((len(df1[col]),colms),np.uint8)
    for i in range(onehot.shape[0]):
        onehot[i,df1[col][i]]=1
    oneh1=pd.DataFrame(onehot)
    oneh1.columns=[col+str(i) for i in range(oneh1.shape[1])]
    df1=df1.join(oneh1)
    
    onehot=np.zeros((len(df2[col]),colms),np.uint8)
    for i in range(onehot.shape[0]):
        onehot[i,df1[col][i]]=1
    oneh1=pd.DataFrame(onehot)
    oneh1.columns=[col+str(i) for i in range(oneh1.shape[1])]
    df2=df2.join(oneh1)
    
    df1.drop(col, axis=1, inplace=True,errors='ignore')
    df2.drop(col, axis=1, inplace=True,errors='ignore')
    #print("done")
    return df1,df2

def pre_process():
    training,testing=get_dataset(loc="../input/adultData.xlsx")
    training.drop('education', axis=1, inplace=True,errors='ignore')
    testing.drop('education', axis=1, inplace=True,errors='ignore')
    cols=training.columns
    #print(cols)
    
    [ x if x%2 else x*100 for x in range(1, 10) ]
    
    training['sex']=[1 if sex==' Male' else 0 for sex in training['sex']]
    training['income']=[1 if income==' >50K' else 0 for income in training['income']]
    
    testing['sex']=[1 if sex==' Male' else 0 for sex in testing['sex']]
    testing['income']=[1 if income==' >50K' else 0 for income in testing['income']]
    
    cl=[1,9,10,11]
    for i in cl:
        #print(i,cols[i])
        training=normalized(training,cols[i])
        testing=normalized(testing,cols[i])    
    cl=[2,3,4,5,6,7,12]
    for i in cl:
        #print(i,cols[i])
        state_gov = preprocessing.LabelEncoder()
        state_gov=state_gov.fit(training[cols[i]].tolist()+testing[cols[i]].tolist())
        training[cols[i]]=state_gov.transform(training[cols[i]])
        testing[cols[i]]=state_gov.transform(testing[cols[i]])
        training,testing=onehot_enc(training,testing,cols[i])
    return training, testing
