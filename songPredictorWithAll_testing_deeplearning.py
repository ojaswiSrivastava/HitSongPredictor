import pandas as pd, matplotlib.pyplot as plt
import seaborn as sns, numpy as np

#Import csv File
df = pd.read_csv("/home/ojaswi/Documents/projectreport/MillionSongsDataset.csv")

#Exploratory Data Analysis
df.shape
df.columns
df.count()
df.head(5)

#Vizualizations
plt.scatter(df["artist_familiarity"],df["song_hotttnesss"], s=1)
plt.xlabel('song_hotttnesss')
plt.ylabel('artist_familiarity')
plt.title('Artist Familiarity vs. Song Hotness')

plt.hist(df["duration"], normed=False, bins=30)
plt.ylabel('no. of songs')
plt.xlabel('duration')

plt.hist(df["tempo"], normed=False, bins=30)
plt.ylabel('no. of songs')
plt.xlabel('tempo')
plt.savefig("/home/ojaswi/Documents/Main Project/tempo.png")

#Preparing Data Set
new_df= df[['artist_hotttnesss','duration', 'end_of_fade_in', 'key', 'key_confidence', 'loudness',
       'mode', 'mode_confidence','start_of_fade_out', 'tempo','time_signature',
       'time_signature_confidence','year']]
new_df.count()

#Vizualizing Correlation Matrix
corr = new_df.corr()

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.title('Correlation matrix Heatmap')


new_df= df[['artist_hotttnesss','duration', 'end_of_fade_in', 'key', 'key_confidence', 'loudness',
       'mode', 'mode_confidence','start_of_fade_out', 'tempo','time_signature',
       'time_signature_confidence','year']]
new_df.count()
#new_df = new_df.fillna(new_df.mean())
#pd.isnull(new_df).sum() > 0
#new_df = new_df.dropna()
#print(new_df.info()) 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


import timeit

#Splitting Data into Training and Testing Dataset
xTrain, xTest, yTrain, yTest = train_test_split(new_df, df["bbhot"], test_size = 0.3)


LR_model = LogisticRegression()
KNN_model = KNeighborsClassifier(n_neighbors=5)
DT_model = DecisionTreeClassifier()
SVC_model = SVC()
RFC_model = RandomForestClassifier()


start = timeit.default_timer()

LR_model.fit(xTrain,yTrain)
KNN_model.fit(xTrain,yTrain)
DT_model.fit(xTrain,yTrain)
SVC_model.fit(xTrain,yTrain)
RFC_model.fit(xTrain,yTrain)


#Prediction
LR_prediction = LR_model.predict(xTest)
KNN_prediction = KNN_model.predict(xTest)
DT_prediction = DT_model.predict(xTest)
SVC_prediction = SVC_model.predict(xTest)
RFC_prediction = RFC_model.predict(xTest)

stop = timeit.default_timer()

print('Time: ', stop - start)  

#Testing Different Models
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


accuracy_score(yTest,LR_prediction)
accuracy_score(yTest,KNN_prediction)
accuracy_score(yTest,DT_prediction)
accuracy_score(yTest,SVC_prediction)
accuracy_score(yTest,RFC_prediction)


roc_auc_score(yTest,LR_prediction)
roc_auc_score(yTest,KNN_prediction)
roc_auc_score(yTest,DT_prediction)
roc_auc_score(yTest,SVC_prediction)
roc_auc_score(yTest,RFC_prediction)

#precision_score(yTest,LR_prediction)
#precision_score(yTest,KNN_prediction)
#precision_score(yTest,DT_prediction)
#precision_score(yTest,SVC_prediction)
#precision_score(yTest,RFC_prediction)
#
#recall_score(yTest,LR_prediction)
#recall_score(yTest,KNN_prediction)
#recall_score(yTest,DT_prediction)
#recall_score(yTest,SVC_prediction)
#recall_score(yTest,RFC_prediction)



#Validating Models
from sklearn.model_selection import cross_val_score

cross_val_score(LR_model, xTrain,yTrain,cv=10)
cross_val_score(KNN_model, xTrain,yTrain,cv=10)
cross_val_score(DT_model, xTrain,yTrain,cv=10)
cross_val_score(SVC_model, xTrain,yTrain,cv=10)
cross_val_score(RFC_model, xTrain,yTrain,cv=10)

#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV


DT_param = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

#Hyperparameter Tuning LogisticRegression model
LR_parameters = {"C":[0.0001,0.001,0.01,0.1,0.5,1,5,10,100]}
cv_random = GridSearchCV(LR_model, LR_parameters)
cv_random.fit(xTrain,yTrain)
cv_random.best_params_
cv_random.best_score_

#Hyperparameter Tuning DecisionTreeClassifier model
DT_param = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
cv_random = GridSearchCV(DT_model, DT_param)
cv_random.fit(xTrain,yTrain)
cv_random.best_params_
cv_random.best_score_


#Printing Input data 
with pd.option_context( 'display.max_columns', None):  # more options can be specified also
    print(xTest)
    
#Printing Predicted Data
DT_prediction[:, None]

#Articial Neural Network
from keras.models import Sequential
from keras.layers import Dense

#Transforming and Scaling the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.transform(xTest)

#Set up the sequential ANN model
model = Sequential()

#Input Layer
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu', input_dim=13))

#Hidden Layer
model.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))

#Output Layer
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy' ])

ann_start_time = timeit.default_timer()

model.fit(xTrain, yTrain, batch_size = 10, epochs=100)
ANN_prediction = model.predict(xTest)

ann_stop_time = timeit.default_timer()

print('Time: ', ann_stop_time - ann_start_time)  

ANN_prediction = np.where(ANN_prediction > 0.5, 1, 0)

accuracy_score(yTest,ANN_prediction)
roc_auc_score(yTest,ANN_prediction)

#for number in ANN_prediction:
#    print(number)





