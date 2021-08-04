# Some information about Heart_disease Data
url = 'https://archive.ics.uci.edu/ml/datasets/heart+disease'

# Importing the Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Importing the data .

dataset = pd.read_csv('data/heart-disease.csv')
dataset.head()


# Test missing values of Data .
dataset.isna().sum()


# Visualise the Data to select important Features
sns.set_theme = "ticks"
sns.pairplot(dataset)

# Visualise The Features and Its Effect in target

for i in range(len(dataset.columns)-1):
    fig,ax = plt.subplots()
    print(f'visualize {dataset.columns[i]} with target\n')
    ax = sns.barplot(data = dataset , x = dataset['target'] , y = dataset.columns[i])
    plt.show()

# Splitting the Data to Features and Target
X = dataset.drop(['age' , 'trestbps' , 'chol' , 'fbs' , 'slope' ,
                  'target'] ,axis = 1)
y = dataset.iloc[:,-1]

# scalling
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)


# Splitting the Data to Training and Test
X_train , X_test , y_train , y_test = train_test_split( X ,
                                                        y ,
                                                        test_size = 0.2 , 
                                                        random_state = 0 )

# Classification : 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {'Logestic_Regression' : LogisticRegression() ,
          'KNN' : KNeighborsClassifier() ,
          'Random_Forest_Classifier' : RandomForestClassifier() ,
          'SVC' : SVC() ,
          'Decision_Tree' : DecisionTreeClassifier()
          }

def fit_and_score(models , X_train , X_test , y_train , y_test) :
    model_scores = {}
    model_confusion = {}
    for name , model in models.items() :
        # fitting the data :
        model.fit(X_train , y_train)
        model_scores[name] = model.score(X_test , y_test)
        y_predict = model.predict(X_test)
        model_confusion[name] = confusion_matrix(y_test , y_predict)
    return model_scores , model_confusion

fit_and_score(models = models ,
              X_train = X_train,X_test = X_test,
              y_train = y_train,y_test = y_test )
