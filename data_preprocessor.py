################################################################################
### Author: Wu, Ziwei
### Description: This program cleans up the data and perform feature engineering 
### in order to get the data for machine learning 
################################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import os
from scipy import stats
from scipy.stats import norm, skew 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import FeatureUnion

#Let user know that script is starting
print("Data preprocessing script starting...")

#load the datasets
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

train_ID = train['Id']
test_ID = test['Id']
test_ID = test_ID.values
np.savetxt("datasets/test_ID.csv", test_ID)

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True) 

outliers = train[(train['GrLivArea']>4000) & (train['SalePrice']<30000)].index
train = train.drop(outliers)

##############################################################################
### Adjust the y variable SalePrice to make it more gaussian 
### (Machine learning algorithm works well with gaussian distribution)
##############################################################################

(mu, sigma) = norm.fit(train['SalePrice'])
train["SalePrice"] = np.log1p(train["SalePrice"])

#############################################################################
### Combine the train and test data before processing 
#############################################################################
y_train = train.SalePrice.values
ntrain = train.shape[0]
ntest = test.shape[0]

#function that combine train and test data
def combine_train_test(train, test):
    print("Combining the training and testing datasets...")
    all_data = pd.concat((train, test)).reset_index(drop=True)
    print("Train dataset and test dataset has been combined, the size(rows, columns) of new data is: ", all_data.shape)
    print()
    return all_data 

all_data = combine_train_test(train, test)
all_data.drop(['SalePrice'], axis=1, inplace=True)

###########################################################################
###clean the missing data
###########################################################################
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
# check for remaining missing values 
def find_missing_data(dataset):
    print("Finding missing values in the dateset...")
    if dataset.isnull().values.any() == False:
        print("There is no missing values in the dataset")
        print()
    else:    
        total = dataset.isnull().sum().sort_values(ascending=False)
        percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print(missing_data)
        print()

find_missing_data(all_data)

###########################################################################
###                 Feature Engineering
###########################################################################

###########################################################################
###  Convert data types of certain features 
###########################################################################
#MsubClass is building class, convert it to str type
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

#Changing OverallCond and OverallQual into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['OverallQual'] = all_data['OverallQual'].astype(str)

###########################################################################
### Label Encoding categorical datas implies a ordinal relationship
###########################################################################
from sklearn.preprocessing import LabelEncoder
labels = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'OverallQual')

# Function takes a list of labels, and use LabelEncoder to encode all values 
# of each labels(features)
def label_encoder(labels, dataset):
    """
    Encode all values of each categorical label to numerical values 
    
    args: 
    Argument 1: a list of labels in a dataset
    Argument 2: a panda dataframe

    return: 
    A panda dataframe with labels encoded

    """
    print("Performing label encoding... ")
    count = 0
    for item in labels:
        label_encoder = LabelEncoder() 
        label_encoder.fit(list(dataset[item].values)) 
        dataset[item] = label_encoder.transform(list(dataset[item].values))
        print(item, " has been encoded to numerical values.")
        count += 1 
    print("Total ", count, " labels has been encoded")
    print('Rows and Columns of Dataset: {}'.format(dataset.shape))
    print()
    return dataset

all_data = label_encoder(labels, all_data)

###########################################################################
### Create new features by combining existing features
###########################################################################
#create Total square foot feature by combines the area of basement, 1st and 2nd floors
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

###########################################################################
### Find and fix very skewed numerical features 
###########################################################################

# Check the skewed of all numerical features
def skewed_transformer(dataset):
    """
    Find the numerical features 
    that are skewed and make them 
    more Guassian by Box Cox transformation

    Args: 
    a pandas dataframe
    
    Return: 
    a pandas dataframe with more Gaussian distributed numerical features
    
    """
    print("Perform skewed transformation using Box Cox...")
    numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index
    skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]
    print(skewness)
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        dataset[feat] = boxcox1p(dataset[feat], lam)
        print(feat, "being transformed, its skewness is now", dataset[feat].skew())
    print("The current dimension of dataset is: ", dataset.shape)
    print()
    
    return dataset

#Find skewed numerical feature and transform using Box Cox transformation
all_data = skewed_transformer(all_data)

###########################################################################
### Getting dummy categorical features
###########################################################################
def get_dummies(dataset):
    """
    Get dummies categorical featuers for the dataset 
    """
    print("Dataset will be transformed by get_dummies...")
    dataset = pd.get_dummies(dataset)
    print("Obtain dummies categorical features completed.")
    print("The new dimension of dataset is ", dataset.shape)
    print()

    return dataset

all_data = get_dummies(all_data)

###########################################################################
### Seperate the dataset back to training and testing sets and outputs 
### them to two files called train_processed.csv and test_processed.csv
###########################################################################
train_processed = all_data[:ntrain]
test_processed = all_data[ntrain:]

def write_to_csv(dataset, path_name):
    """
    Write the dataset to a csv file  
    
    Args:
    arg1: a panda data frame
    arg2: the path that file is written to
    """
    dataset.to_csv(path_name)
    print("Dataset is written to csv format as (", path_name, ")")

#convert y_train to a dateframe before saving it to csv
write_to_csv(train_processed, "datasets/train_processed.csv")
write_to_csv(test_processed, "datasets/test_processed.csv")

np.savetxt("datasets/y_train.csv", y_train)

#Let user know that the data preprocessing is completed
print("Data Preprocessing is completed successful, datasets are ready for machine learning:)")
