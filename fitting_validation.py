##### Fitting and Validation of models #####

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

1. Fitting (example)

rf = RandomForestClassifier(random_state = 3)
rf.fit(x, y)
y_pred = rf.predict(x_train)

2. Cross-val score

print(cross_val_score(rf, train, y, scoring = 'accuracy', cv = 5).mean())

3. Searching optimal hyperparameters

from sklearn.model_selection import GridSearchCV

pipeline_rf = Pipeline([ 
                     ('sca', RobustScaler()),
                     ('rf', RandomForestClassifier(random_state = 3))
                     
])

parameters = {}
parameters['rf__min_samples_split'] = [2,3,4,5] 
parameters['rf__max_depth'] = [4, 6, 8, None] 
parameters['rf__n_estimators'] = [10, 25, 50, 100] 

CV = GridSearchCV(pipeline_rf, parameters, scoring = 'accuracy', n_jobs= 4, cv = 5)
CV.fit(train, y)   

print('Best score and parameter combination = ')

print(CV.best_score_)
print(CV.best_params_)


4. Generating a K-fold (useful for customized cross-validation)

from sklearn.model_selection import KFold
kf = KFold(n_splits = 5) # 5-fold cross-validation
# provides train/test indices
# can also use StratifiedKFold for categorical data
# to have equal distribution of classes in each fold

for train_index, test_index in kf.split(X):
    #train_index, test_index are indexes of 
    #training and testing subsets of X
    
5. Randomly re-shuffling the data for cross-validation 

data = some_dataframe
n = len(data)
data['num'] = [np.random.rand() for i in range(n)]
data.sort_values(by = ['num'], inplace = True)

6. One-class SVM for novelty detection

# unsupervised learning algorithm - uses data  
# to plot a SVM boundary, beyond 
# which we only have outliers
# useful for working with very imbalanced datasets

from sklearn.svm import OneClassSVM

oneclass_svm.fit(Xt);
classif = oneclass_svm.predict(Xt);
novelties = [True if el == -1 else False for el in classif]

7. Isolation Forest for anomaly detection

# points are given anomaly score, based on how many
# partitions are needed to separate the point 
# e.g. few partitions mean the point is probably 
# an outlier

from sklearn.ensemble import IsolationForest

# training the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)

# predictions
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
