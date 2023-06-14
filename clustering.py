"""
Run clustering algorithm (n=3 according to R package nbclust) using MCI neuropsychological scores
2023
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
duration = 5 seconds
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


# reproducibility
np.random.seed(0)

# get the participant scores
df_scores = pd.read_excel('_createdDataframes/df_scores.xlsx', index_col=0) # 1032 subjects

# rename lmci and emci as MCI
df_scores['Group'][df_scores['Group'] == 'LMCI'] = 'MCI'
df_scores['Group'][df_scores['Group'] == 'EMCI'] = 'MCI'
assert len(np.unique(df_scores.Group)) == 3

# order for category (visuospatial, memory, language, executive)
col_for_clustering = ['ADAS Q3 constr. praxis', 'ADAS Q6 ideat. praxis',  # visuospatial
        'ADAS Q8 word recognition', 'Logical memory scale II','RAVLT short delay', 'RAVLT long delay', 'RAVLT recognition', 'RAVLT learning', 'RAVLT forgetting',# memory
        'VFT animals', 'Boston naming test', 'ADAS Q5 naming', # language
        'TMT score A', 'TMT score B-A'# excecutive
        ]


#########################################################
# # Compare performance of ADNI 1 and ADNI 2 participants #
# #########################################################
# for col in df_scores.columns:
#        print(' '* 10)
#        print('*'*10)
#        print(col)
#        print(df_scores[col][df_scores.Session == 'ADNI1'][df_scores.Group == 'MCI'].describe())
#        print(df_scores[col][df_scores.Session == 'ADNI2'][df_scores.Group == 'MCI'].describe())
       

# # check for outliers
# potential_outliers = []
# std_pop = df_scores[col_for_clustering].std()
# mean_pop = df_scores[col_for_clustering].mean()
# high_thr = mean_pop + 4*std_pop
# low_thr = mean_pop - 4*std_pop
# for ind in df_scores.index:
#     for col in col_for_clustering:
#         if df_scores[col].loc[ind] < low_thr[col]:
#             print("Outlier low : ", col, ind)
#             potential_outliers.append(ind)
#         elif df_scores[col].loc[ind] > high_thr[col]:
#             print("Outlier high : ", col, ind)
#             potential_outliers.append(ind)

# potential_outliers = list(np.unique(potential_outliers))
# df_scores[col_for_clustering].loc[potential_outliers]

# for ind in potential_outliers:
#     for col in col_for_clustering:
#         print(int(mean_pop[col]), ' : ', df_scores[col].loc[ind])#, col, ind)
# # no weird oultiers values



################################################
# rescale so that low score == low performance #
################################################

df_scores['ADAS Q3 constr. praxis'] = df_scores['ADAS Q3 constr. praxis'] * -1
df_scores['ADAS Q5 naming'] = df_scores['ADAS Q5 naming'] * -1
df_scores['ADAS Q6 ideat. praxis'] = df_scores['ADAS Q6 ideat. praxis'] * -1
df_scores['ADAS Q8 word recognition'] = df_scores['ADAS Q8 word recognition'] * -1
df_scores['ADAS Q10 compr. spoken lang.'] = df_scores['ADAS Q10 compr. spoken lang.'] * -1
df_scores['ADAS Q11 word finding difficuty'] = df_scores['ADAS Q11 word finding difficuty'] * -1
df_scores['RAVLT forgetting'] = df_scores['RAVLT forgetting'] * -1
df_scores['RAVLT learning'] = df_scores['RAVLT learning'] * -1
df_scores['TMT score A'] = df_scores['TMT score A'] * -1
df_scores['TMT score B-A'] = df_scores['TMT score B-A'] * -1

df_scores_reversed = df_scores[col_for_clustering].copy()
df_scores_reversed['Group'] = df_scores['Group']

# chech that CN mean scores is higher than MCI mean scores
for col in df_scores_reversed.columns[:-1]:
    if not df_scores_reversed[col][df_scores_reversed['Group'] == 'CN'].mean() > df_scores_reversed[col][df_scores_reversed['Group'] == 'MCI'].mean():
        print(col)


# standardize
data = StandardScaler().fit_transform(df_scores_reversed[df_scores_reversed.columns[:-1]].values)
df_scores_standardized = pd.DataFrame(data=data, columns=df_scores_reversed.columns[:-1], index=df_scores_reversed.index)
# add diagnosis + other information:
df_scores_standardized = df_scores_standardized.join(df_scores[['smwc1_path', 'smwc2_path', 'smwc3_path', 'Group', 'Sex', 'Age',
       'Session', 'ID', 'TurnedAD', 'APOE4', 'GDS_tot', 'FAQ_tot', 'ANART']],  how='inner')


####################
# Get MCI patients #
####################

df_mci = df_scores_standardized[df_scores_standardized['Group'] == 'MCI'] # 640 MCI patients




########## R code to apply the package NbClust (to get optimal number of cluster) #######
"""
R code:
# instal package (Cran package: Munster)
install.packages("NbClust") 
install.packages("readxl")

require("NbClust")
library(readxl)

data <- read_excel("//calebasse/lefortb211/Bureau/work/Unicaen/ADNI/Adni1_Adni2/df_MCI_cluster.xlsx")
data = subset(data, select = -c(RID, APOE4, cluster, Sex, Age, Session, ID, TurnedAD, smwc1_path, smwc2_path, smwc3_path, Group, GDS_tot, FAQ_tot, ANART))
set.seed(42)
data <- scale(data)
NbClust(data, min.nc = 2, max.nc = 8, method="kmeans")

my.data <- read.table(pipe("pbpaste"), sep = "\t", header=TRUE)
set.seed(42)
# check the data
head(my.data)
# standardize the data
my.data <- scale(my.data)
# check again
head(my.data)
# Apply the nbclust package
NbClust(my.data, min.nc = 2, max.nc = 5, method="kmeans")
""" 

'''
library(readxl)
data <- read_excel("//calebasse/lefortb211/Bureau/Neuropsychological/to_cluster.xlsx")
data = subset(data, select = -c(RID))
set.seed(42)
# data <- my_data[ , c("CCI13", "CCI14", "CCI15", "CCI16", "CCI17", "CCI18", "CCI19","CCI20")] 
data <- scale(data)
NbClust(data, min.nc = 2, max.nc = 5, method="kmeans")

******************************************************************* 
* Among all indices:                                                
* 7 proposed 2 as the best number of clusters 
* 10 proposed 3 as the best number of clusters 
* 2 proposed 4 as the best number of clusters 
* 1 proposed 5 as the best number of clusters 
* 1 proposed 7 as the best number of clusters 
* 2 proposed 8 as the best number of clusters 

                   ***** Conclusion *****                            
 
* According to the majority rule, the best number of clusters is  3 
 
 
******************************************************************* 

'''


################################################################
## Run clustering analysis with nb_clust == 3 on MCI patients ##
################################################################

data = df_mci[col_for_clustering].values # take off non tests column

model = AgglomerativeClustering(n_clusters=3) #linkage=ward
model = model.fit(data)
cluster_labels = model.fit_predict(data)

df_mci['cluster'] = cluster_labels

# reformat cluster name from 0, 1, 2 to low/middle/high
# get scores per cluster
df_0 = df_mci[col_for_clustering][df_mci['cluster'] == 0]
df_1 = df_mci[col_for_clustering][df_mci['cluster'] == 1]
df_2 = df_mci[col_for_clustering][df_mci['cluster'] == 2]

# check if 0 is low (if not, find the good one)
cluster_df = [df_0, df_1, df_2]
mean_cls = np.array([df_0.mean(axis=0).mean(), df_1.mean(axis=0).mean(), df_2.mean(axis=0).mean()])
index_min = np.argmin(mean_cls)
df_mci['cluster'][df_mci['cluster'] == index_min] = 'Low'
index_max = np.argmax(mean_cls)
df_mci['cluster'][df_mci['cluster'] == index_max] = 'High'
index_middle = min(set([0, 1, 2]) - set([index_min, index_max])) # min to extract from set
df_mci['cluster'][df_mci['cluster'] == index_middle] = 'Middle'


# save 
df_scores_standardized['cluster'] = df_scores_standardized['Group'].values
df_scores_standardized['cluster'].loc[list(df_mci['cluster'].index)] = df_mci['cluster']

df_scores_standardized.to_excel('_createdDataframes/df_scores_std.xlsx')


# check significativity (circular but Leslie asked for it)
from scipy import stats
for ind, test in enumerate(col_for_clustering):
    df_ = df_scores_standardized[[test, 'cluster']]
    low = df_[test][df_['cluster'] == 'Low'].values
    middle = df_[test][df_['cluster'] == 'Middle'].values
    high = df_[test][df_['cluster'] == 'High'].values
    cn = df_[test][df_['cluster'] == 'CN'].values
    tStat, pValue0 = stats.ttest_ind(low, middle, equal_var = False)
    tStat, pValue1 = stats.ttest_ind(low, high, equal_var = False)
    tStat, pValue2 = stats.ttest_ind(middle, high, equal_var = False)
    tStat, pValue3 = stats.ttest_ind(middle, cn, equal_var = False)
    tStat, pValue4 = stats.ttest_ind(high, cn, equal_var = False)
    if pValue0<0.05:
        print('')
        print('low vs middle significatif => ', test)
        print('')
    if pValue1<0.05:
        print('')
        print('low vs high significatif => ', test)
        print('')
    if pValue2<0.05:
        print('')
        print('middle vs high significatif => ', test)
        print('')
    if pValue3<0.05:
        print('')
        print('middle vs cn significatif => ', test)
        print('')
    if pValue4<0.05:
        print('')
        print('high vs cn significatif => ', test)
        print('')






from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix

from scipy.cluster import hierarchy
import matplotlib

hierarchy.set_link_color_palette(['dimgray',
 'gainsboro',
 'darkgrey'])
 # 'orange',
 # 'orange',
 # 'orange',
 # 'orange',
 # 'orange',
 # 'orange',
 # 'orange'])

data = df_mci[col_for_clustering].values # take off non tests column
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)#linkage=ward
model = model.fit(data)
# plot the top three levels of the dendrogram

linkage_matrix = plot_dendrogram(model, truncate_mode="level", 
    p=50, color_threshold=35, above_threshold_color="black")
plt.axis('off')
plt.savefig('_figures/dendogram.png')
plt.savefig('_figures/figure1b.png')
plt.show()