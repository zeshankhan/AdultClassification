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

_,testing=pre_process()
testing.astype(np.float64)

test_y=testing['income']
test_X=testing.drop('income', axis=1, inplace=False,errors='ignore')

vhard = VotingClassifier(estimators=[('dt', DecisionTreeClassifier()), ('scv', LinearDiscriminantAnalysis()), ('gnb', GaussianNB())], voting='hard')

filename = '../models/finalized_model_Hard_Majority_Voting.sav'
model1 = pickle.load(open(filename, 'rb'))
pred_y = model1.predict(test_X)
print(f1_score(test_y,pred_y),accuracy_score(test_y,pred_y),confusion_matrix(test_y,pred_y))