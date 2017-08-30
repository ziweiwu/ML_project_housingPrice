################################################################################
###Author:Wu, Ziwei
###Description: This program explores the dataset by looking at the attributes
###and save the figures in images folder
################################################################################
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#save the plot as png images in images folder
def save_graph(graph_name):
    path_root = "images/"
    path_name = path_root + graph_name +".png"
    plt.savefig(path_name, dpi=300)
    print("The figure saved as: ", path_name)
    plt.close()

#find the numbers and percentage of missing data in a pandaframe
def find_missing_data(dataset):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

#loads csv data into a panda dataframe
housing = pd.read_csv("datasets/train.csv")

#plot an histogram of all attributes and save the figure
housing.hist(bins=50, figsize = (30,25))
save_graph("attributes_hist")

#plot an distribution of sale price and save the figure
sns.distplot(housing["SalePrice"])
save_graph("SalePrice_dist")
