import ujson
from sklearn import base, pipeline
import pandas as pd
import dill
from sklearn.externals import joblib

class Estimator(base.BaseEstimator, base.RegressorMixin):
    def __init__(self):
        """
        File containing the traning data, in json format
        """
        self.df = pd.DataFrame()

    def fit(self, training):
        # fit the model ...
        
        json_file = file(training, 'r')
        lines = json_file.readlines()

        json_dicts = [] # Will be a list of json dicts
        for line in lines:
            json_dicts.append(ujson.loads(line))
        
        self.df = self.df.from_dict(json_dicts)
        self.city_data = self.df.groupby('city')
        
        return self

    def predict(self, X):
        return self.city_data.stars.mean()[X]
        
model = Estimator()
model.fit('yelp_train_academic_dataset_business.json')
joblib.dump(model, 'city_model.pkl') 
dill.dump(model, open("city_model.dill", 'w'))

clf1 = joblib.load('city_model.pkl') 
with open("./city_model.dill") as city_model_file:
    model = dill.load(city_model_file)

# X_train = range(5)
# y_train = range(5)
# 
# X_test = range(6,11)
# y_test = range(6,11)
# estimator = Estimator(X_train)  # initialize
# estimator.fit(X_train, y_train)  # fit data
# y_pred = estimator.predict(X_test)  # predict answer
# estimator.score(X_test, y_test)  # evaluate performance
# 
# transformer = Transformer(X_train)  # initialize
# X_trans_train = transformer.fit_transform(X_train)  # fit / transform data
# estimator.fit(X_trans_train, y_train)  # fit new model on training data
# X_trans_test = transformer.transform(X_test)  # transform test data
# estimator.score(X_trans_test, y_test)  # fit new model

