###############################################################################
###Author: Wu, Ziwei
###Description: This program performs multiple machine learning algorithms
### and fine-tuning algoritms to predict the housing price
###############################################################################
import pandas as pd
import numpy as np 
import os 
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

#load the dataset 
print("\nLoading the datasets, getting ready for machine learning...")
train = pd.read_csv("datasets/train_processed.csv")
test = pd.read_csv("datasets/test_processed.csv")
y_train = np.genfromtxt("datasets/y_train.csv")
test_ID = np.genfromtxt("datasets/test_ID.csv")

print(y_train)
print(test_ID)
############################################################################### 
### Cross Validation Implementation
###############################################################################

#function calculates the root squared logarithmic error
def rmsle_cross_validation(model, n_folds = 10, training = train, testing = test):
    """ 
    the function calculates the cross validation score with k folds with
    shuttled dataset each iteration 

    Args:
    arg1: a machine learning model 
    arg2: number of folds, default is 10
    arg3: training data, default is train 
    arg4: testing data, default is test

    return: 
    The mean squared error, the lower score is better
    """ 
    k_fold_shuttle = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring = "neg_mean_squared_error", cv = k_fold_shuttle)) 
    return rmse

###########################################################################
### Base machine learning models with parameters tuning
###########################################################################

#Lasso 
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 1))


#Kernal Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


#Elastic Net Regression(This model takes the pros of both lasso and KRR)
ENet = make_pipeline(RobustScaler(),ElasticNet(alpha = 0.0005, l1_ratio = .9, random_state = 3))


#Gradient Boost Regressor(lad loss function is highly robust)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='lad', random_state =5)


#XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                              nthread=-1)


#LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


###########################################################################
### Base models performances 
###########################################################################
score = rmsle_cross_validation(lasso)
print("\nLasso score: {:5f}\n".format(score.mean()))

score = rmsle_cross_validation(KRR)
print("\nKernal Ridge Regression score: {:5f}\n".format(score.mean()))

score = rmsle_cross_validation(ENet)
print("\nElastic Net Regression Score: {:5f}\n".format(score.mean()))

score = rmsle_cross_validation(model_xgb)
print("\nXGBoost score: {:5f}\n".format(score.mean()))

score = rmsle_cross_validation(model_lgb)
print("\nLightGBM : {:5f}\n".format(score.mean()))


###########################################################################
### Build an average models of all above models 
###########################################################################
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 

base_models = (ENet, KRR, lasso)
average_models = AveragingModels (models = base_models)
score = rmsle_cross_validation(average_models)
print("\nAverage base models score: {:.5f}\n".format(score.mean()))

###########################################################################
### Build an stacked average regressor 
###########################################################################
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=10):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

base_models = (ENet, KRR, lasso)
stacked_averaged_models = StackingAveragedModels(base_models,
                                                 meta_model = lasso)

score = rmsle_cross_validation(stacked_averaged_models)
print("\nStacking Averaged regressor score: {:.5f}\n".format(score.mean()))

###########################################################################
### Emsembling stack regressor with XGBoost, LightGBM and GBoost
###########################################################################
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))


GBoost.fit(train, y_train)
GBoost_train_pred = GBoost.predict(train)
GBoost_pred = np.expm1(GBoost.predict(test))
print(rmsle(y_train, GBoost_train_pred))


model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))


print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 + GBoost_train_pred*0.10+
               xgb_train_pred*0.10 + lgb_train_pred*0.10 ))

ensemble = stacked_pred*0.70 + xgb_pred*0.10 + lgb_pred*0.10 + GBoost_pred*0.10

###########################################################################
### Submission 
###########################################################################
test_ID = test_ID.astype(np.int) 
submission = pd.DataFrame()
submission["Id"] = test_ID
submission["SalePrice"] = ensemble
path = 'submission/submission.csv'
submission.to_csv(path,index=False)
print("Submission file is written to ", path, ", Good job!")
