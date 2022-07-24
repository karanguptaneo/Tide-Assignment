# train.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
# loading and creating the Target Variable
data = pd.read_csv("data_interview_test.csv" , sep = ":")
data['Target'] = 0 # filling all the Value with Zero
matched_bool_array = data['matched_transaction_id'] == data['feature_transaction_id']
data.loc[ matched_bool_array , 'Target' ] = 1 # assigning 1 value to all the rows which matched

# train cv , test split
X_train, X_test, y_train, y_test = train_test_split( data.drop( ['Target' ,'receipt_id' , 'company_id' , 'matched_transaction_id' , 'feature_transaction_id' ] , axis = 1 ) , data['Target'], stratify = data['Target'], test_size=0.20 , random_state = 5 )
feature_names = X_train.columns
X_train, X_cv, y_train, y_cv = train_test_split( X_train , y_train, stratify = y_train, test_size=0.15 )

# over sampling the data
sm = SMOTE(random_state = 0)
X_train, y_train = sm.fit_resample(X_train, y_train.ravel())

# forming a pipeline for transforming the data followed by training the model

pipeline = Pipeline([('scaler', StandardScaler()), ('SGD',SGDClassifier(loss='log_loss',random_state = 2,alpha = 10**-4) ) ])
pipeline.fit(X_train.values , y_train)

with open('log_regress.sav', 'wb') as model:
    pickle.dump(pipeline , model)
    
with open('log_regress.sav' , 'rb') as model:
    pipeline = pickle.load(model)

#pipeline.score(X_test.values , y_test)
