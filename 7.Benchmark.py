
##############################################################
## I: Probing complex relationships among the grey matter rois  ###
##############################################################

import numpy as np
from scipy.stats import scoreatpercentile
import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt
import matplotlib.patches as mpatches

#  sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import  LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


np.random.seed(0)

def benchmark(X, y):
    estimators_classif = [
    # linear estimators
    ['SVM L2', {'C': np.logspace(-5, +5, 11)}, LinearSVC(penalty='l2', dual=False, class_weight='balanced')],
    ['LogReg L2', {'C': np.logspace(-5, +5, 11)}, LogisticRegression(penalty='l2', class_weight='balanced')],
    ['Ridge L2', {'alpha': np.logspace(-5, +5, 11)}, RidgeClassifier(solver='lsqr', class_weight='balanced')],

    # non-linear estimators
    ['kNN', {'n_neighbors': np.arange(1, 26)}, KNeighborsClassifier()],
    ['RandForest', {'max_depth': [3, 5, 10, None], 'n_estimators': [50, 100, 200]}, RandomForestClassifier(class_weight='balanced')],
    ['AdaBoost', {'n_estimators': [50, 100, 150, 200]}, AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1,class_weight='balanced'))]
    ]
    res_cont = {}
    qc = {}
    for est_name, est_grid, est in estimators_classif:
        print("Running ", est_name)
        folder = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        folder.get_n_splits(X, y)
        clf = GridSearchCV(est, est_grid)
        train_accs = []
        test_accs = []
        for train_inds, test_inds in folder.split(X, y):
            clf.fit(X[train_inds], y[train_inds])
            y_pred = clf.score(X[train_inds], y[train_inds])
            train_accs.append(y_pred)
            
            y_pred = clf.score(X[test_inds], y[test_inds])
            test_accs.append(y_pred)

        res_cont[est_name] = {}
        res_cont[est_name]['train'] = train_accs
        res_cont[est_name]['test'] = test_accs
         
        qc[est_name] = np.mean(test_accs)
        # print(qc)
        # print(est_name, est_grid, est, np.mean(test_accs))
        
    res_array = [[k, res_cont[k]['train'], res_cont[k]['test']] 
        for k in res_cont.keys()]
    df = pd.DataFrame(res_array, columns=['estimator', 'train', 'test'])
    print(qc) 

    for estimator in df.estimator:
        assert np.mean(list(df['test'][df['estimator'] == estimator])) == qc[estimator]
    return df, res_array


def plot_violin(df, n, title):
    colortest='orangered'
    colortrain='wheat'
    plt.figure(figsize=(20, 10), dpi=80)

    ax = sns.violinplot(x=None, y=None, data=df.test, color=colortest, inner=None)
    ax = sns.violinplot(x=None, y=None, data=df.train, color=colortrain, inner=None)
    for violin, alpha in zip(ax.collections[::], [1]*n + [0.4]*n):
        violin.set_alpha(alpha)
    # ax = sns.swarmplot(x=None, y=None, data=df.test, color='white', edgecolor="gray")

    plt.axvline(x=2.5, ymin= 0, ymax = 1, linewidth=2, color='k', linestyle = '--')
    plt.axhline(y=0.5, xmin=0, xmax = n, linewidth=1, color='red', linestyle = 'dotted')
    plt.xlabel('Pattern-learning algorithm', fontsize=23)
    plt.ylabel('Prediction accuracy (+/- variance)', fontsize=23)
    # We change the fontsize of minor ticks label 
    plt.tick_params(axis='both', labelsize=20)

    # ax = sns.violinplot(x=None, y=None, data=df, palette="muted")
    # plt.plot(range(n), res_array[:, 1], label='train', color='black')
    # plt.plot(range(n), res_array[:, 2], label='testg', color='blue')
    plt.xticks(range(n), df.estimator.values)
    test_s = mpatches.Patch(color=colortest, label='Test set')
    train_s = mpatches.Patch(color=colortrain, label='Train set')
    plt.legend(handles=[train_s, test_s], loc='lower left', prop={'size':20})
    plt.ylim(0.3, 1)
    plt.tight_layout()
    plt.savefig('_figures/final_figs/{}.png'.format(title), DPI=400, facecolor='white')
    plt.close('all')




#################################
## Benchmark with gray matter ###
#################################


# load required dataframe
# grey matter quantity per roi
df = pd.read_pickle('_pickles/gm_cleaned_atlas_harvard_oxford')
# group label
cluster = pd.read_excel('_createdDataframe/df_scores_std.xlsx', index_col=0)['cluster']
df['cluster'] = cluster

front_lob_cn = df['Left Frontal Pole'][df["cluster"] == 'CN'].mean()
front_lob_low = df['Left Frontal Pole'][df["cluster"] == 'Low'].mean()
front_lob_middle = df['Left Frontal Pole'][df["cluster"] == 'Middle'].mean()
front_lob_high = df['Left Frontal Pole'][df["cluster"] == 'High'].mean()

assert front_lob_cn > front_lob_high > front_lob_middle > front_lob_low

###############
## mixed MCI ##
###############

print("Running IRM with mixed MCI")

df_ = df[(df.cluster == 'CN')|(df.cluster == 'Low')]
N_low = df_.cluster[df_.cluster=='Low'].__len__()
N_cn = df_.cluster[df_.cluster=='CN'].__len__()
df_.cluster[df_.cluster=='Low'] = 0
df_.cluster[df_.cluster=='CN'] = 1
assert df_.cluster[df_.cluster==0].__len__() == N_low
assert df_.cluster[df_.cluster==1].__len__() == N_cn

X = df_[df_.columns[:-1]].values
X_colnames = df_.columns[:-1]
y = df_[df_.columns[-1]].astype('int64').values  # diagnosis

df_, res_array = benchmark(X, y)
plot_violin(df_, len(res_array), "benchmark_IRM_mixed")
# quality check
all_in = {}
all_in['IRMmixed'] = df_

###############
## amnestic MCI ##
###############

print("#######")
print("Running IRM with amnestic MCI")
print("#######")

df_ = df[(df.cluster == 'CN')|(df.cluster == 'Middle')]
N_middle = df_.cluster[df_.cluster=='Middle'].__len__()
N_cn = df_.cluster[df_.cluster=='CN'].__len__()
df_.cluster[df_.cluster=='Middle'] = 0
df_.cluster[df_.cluster=='CN'] = 1
assert df_.cluster[df_.cluster==0].__len__() == N_middle
assert df_.cluster[df_.cluster==1].__len__() == N_cn

X = df_[df_.columns[:-1]].values
X_colnames = df_.columns[:-1]
y = df_[df_.columns[-1]].astype('int64').values  # diagnosis

df_, res_array = benchmark(X, y)
plot_violin(df_, len(res_array), "benchmark_IRM_amnestic")
# quality check
all_in['IRMamnestic'] = df_

###############
## normal MCI ##
###############

print("#######")
print("Running IRM with FP MCI")
print("#######")

df_ = df[(df.cluster == 'CN')|(df.cluster == 'High')]
N_high = df_.cluster[df_.cluster=='High'].__len__()
N_cn = df_.cluster[df_.cluster=='CN'].__len__()
df_.cluster[df_.cluster=='High'] = 0
df_.cluster[df_.cluster=='CN'] = 1
assert df_.cluster[df_.cluster==0].__len__() == N_high
assert df_.cluster[df_.cluster==1].__len__() == N_cn

X = df_[df_.columns[:-1]].values
X_colnames = df_.columns[:-1]
y = df_[df_.columns[-1]].astype('int64').values  # diagnosis

df_, res_array = benchmark(X, y)
plot_violin(df_, len(res_array), "benchmark_IRM_falsePositiveMCI")
# quality check
all_in['IRMfp'] = df_



####################################
## Benchmark with CSF biomarkers ###
####################################

np.random.seed(0)

df = pd.read_pickle('_pickles/csf_level_std')

Tau_cn = df['TAU'][df["cluster"] == 'CN'].mean()
Tau_low = df['TAU'][df["cluster"] == 'Low'].mean()
Tau_middle = df['TAU'][df["cluster"] == 'Middle'].mean()
Tau_high = df['TAU'][df["cluster"] == 'High'].mean()

assert Tau_cn < Tau_high < Tau_middle < Tau_low


###############
## mixed MCI ##
###############

print("#######")
print("Running CSF with mixed MCI")
print("#######")

df_ = df[(df.cluster == 'CN')|(df.cluster == 'Low')]
N_low = df_.cluster[df_.cluster=='Low'].__len__()
N_cn = df_.cluster[df_.cluster=='CN'].__len__()
df_.cluster[df_.cluster=='Low'] = 0
df_.cluster[df_.cluster=='CN'] = 1
assert df_.cluster[df_.cluster==0].__len__() == N_low
assert df_.cluster[df_.cluster==1].__len__() == N_cn

X = df_[df_.columns[:-1]].values
X_colnames = df_.columns[:-1]
y = df_[df_.columns[-1]].astype('int64').values  # diagnosis

df_, res_array = benchmark(X, y)
plot_violin(df_, len(res_array), "benchmark_CSF_mixed")
# quality check
all_in['CSFmixed'] = df_

###############
## amnestic MCI ##
###############

print("#######")
print("Running CSF with amnestic MCI")
print("#######")

df_ = df[(df.cluster == 'CN')|(df.cluster == 'Middle')]
N_middle = df_.cluster[df_.cluster=='Middle'].__len__()
N_cn = df_.cluster[df_.cluster=='CN'].__len__()
df_.cluster[df_.cluster=='Middle'] = 0
df_.cluster[df_.cluster=='CN'] = 1
assert df_.cluster[df_.cluster==0].__len__() == N_middle
assert df_.cluster[df_.cluster==1].__len__() == N_cn

X = df_[df_.columns[:-1]].values
X_colnames = df_.columns[:-1]
y = df_[df_.columns[-1]].astype('int64').values  # diagnosis

df_, res_array = benchmark(X, y)
plot_violin(df_, len(res_array), "benchmark_CSF_amnestic")
# quality check
all_in['CSFamnestic'] = df_

###############
## normal MCI ##
###############

print("#######")
print("Running CSF with FP MCI")
print("#######")

df_ = df[(df.cluster == 'CN')|(df.cluster == 'High')]
N_high = df_.cluster[df_.cluster=='High'].__len__()
N_cn = df_.cluster[df_.cluster=='CN'].__len__()
df_.cluster[df_.cluster=='High'] = 0
df_.cluster[df_.cluster=='CN'] = 1
assert df_.cluster[df_.cluster==0].__len__() == N_high
assert df_.cluster[df_.cluster==1].__len__() == N_cn

X = df_[df_.columns[:-1]].values
X_colnames = df_.columns[:-1]
y = df_[df_.columns[-1]].astype('int64').values  # diagnosis

df_, res_array = benchmark(X, y)
plot_violin(df_, len(res_array), "benchmark_CSF_falsePositiveMCI")
# quality check
all_in['CSFfp'] = df_


# check results
for k in all_in.keys():
    print(k)
    print(all_in[k]['test'].values)
    for e in all_in[k]['estimator']:
        print(np.mean(list(all_in[k][all_in[k]['estimator'] == e]['test'])))
