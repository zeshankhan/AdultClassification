from preprocess import pre_process
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

def train_models():
    training,testing=pre_process()
    training.astype(np.float64)
    testing.astype(np.float64)
    train_y=training['income']
    train_X=training.drop('income', axis=1, inplace=False,errors='ignore')
    test_y=testing['income']
    test_X=testing.drop('income', axis=1, inplace=False,errors='ignore')
    
    
    vhard = VotingClassifier(estimators=[('dt', DecisionTreeClassifier()), ('scv', LinearDiscriminantAnalysis()), ('gnb', GaussianNB())], voting='hard')
    vsoft = VotingClassifier(estimators=[('dt', DecisionTreeClassifier()), ('scv', LinearDiscriminantAnalysis()), ('gnb', GaussianNB())], voting='soft')
    
    models=[("KNN",KNeighborsClassifier()),("LR",LogisticRegression()),("RF",RandomForestClassifier()),("LDA",LinearDiscriminantAnalysis()),("GNB",GaussianNB()),("DT",DecisionTreeClassifier()),("Hard_Majority_Voting",vhard),("Soft_Majority_Voting",vsoft),("SVC",SVC())]
    
    for name,model in models:
        model.fit(train_X,train_y)
        filename = '../models/finalized_model_'+name+'.sav'
        pickle.dump(model, open(filename, 'wb'))
        #model1 = pickle.load(open(filename, 'rb'))
        #pred_y = model1.predict(test_X)
        #print(f1_score(test_y,pred_y),accuracy_score(test_y,pred_y),confusion_matrix(test_y,pred_y))
        print(name,model)
    return "Model Trained and saved in Models Directory"
