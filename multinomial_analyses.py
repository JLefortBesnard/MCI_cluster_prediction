"""
Run multi-class regression analysis for MCI subgroups
2023
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
duration = 30min
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd
from scipy import stats #v1.5.2
from nilearn import datasets as ds
import nibabel as nib
from nilearn.image import resample_img
from nilearn.input_data import NiftiLabelsMasker
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
from nilearn import plotting

# required paths
nii_path = "/home/jlefortb/ADNI_project/_niftiFiles/low_mean.nii" # file picked randomly but good dimension

def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
	''' Plotting function for the x axis labels to be centered
	with the plot ticks
	Parameters
	----------
	See stackoverflow 
	https://stackoverflow.com/questions/27349341/how-to-display-the-x-axis-labels-in-seaborn-data-visualisation-library-on-a-vert
	
	'''
	axes = []
	if which in ['x', 'both']:
		axes.append(ax.xaxis)
	elif which in ['y', 'both']:
		axes.append(ax.yaxis)
	for axis in axes:
		for t in axis.get_ticklabels():
			t.set_horizontalalignment(ha)
			t.set_rotation(rotation)
			t.set_rotation_mode(rotation_mode)

def accuracy_testing(permutation_accs, final_acc):
	# check if accuracy is significant
	permutation_accs = np.array(permutation_accs).reshape(-1)
	above = stats.scoreatpercentile(permutation_accs, 99)
	percentile_of_acc = stats.percentileofscore(permutation_accs, final_acc)
	if final_acc > above:
		return True, percentile_of_acc
	else:
		return False, percentile_of_acc

def coefficent_testing(labels, permutation_coefs, final_coefs):
	permutation_coefs = np.array(permutation_coefs).reshape(3, 1000*5, -1) # reshape permutation + kfolds
	df_significant = pd.DataFrame(index=['FP', 'Amnestic', 'Mixed'], columns=labels)
	for ind, MCI_subgroup_coefs in enumerate(permutation_coefs):
		current = ['FP', 'Amnestic', 'Mixed'][ind]
		print(current)
		# extract permutation coef per ROI for hypothesis testing
		coeff_per_roi = []
		for roi_nb in range(len(labels)):
			coeff_roi = []
			for coef_iter in MCI_subgroup_coefs:
				coeff_roi.append(coef_iter[roi_nb])
			coeff_per_roi.append(coeff_roi)
		coeff_per_roi = np.array(coeff_per_roi)
		assert coeff_per_roi.shape[0] == len(labels)
		assert coeff_per_roi.shape[1] == 1000*5 # 5CV * permutation number

		# extract 95 and 5% percentile and check if original weight outside these limits to check for significance
		pvals = []
		for n_roi in range(len(labels)):
			coeff_roi = coeff_per_roi[n_roi] 
			above = stats.scoreatpercentile(coeff_roi, 99)
			below = stats.scoreatpercentile(coeff_roi, 1)
			print(above, below)
			print("real value: ", final_coefs[final_coefs.columns[n_roi]].loc[current])
			if final_coefs[final_coefs.columns[n_roi]].loc[current] < below or final_coefs[final_coefs.columns[n_roi]].loc[current]> above:
				pvals.append(1)
				df_significant[labels[n_roi]].loc[current] = final_coefs[final_coefs.columns[n_roi]].loc[current]
			else:
				pvals.append(0)
				df_significant[labels[n_roi]].loc[current] = 0
		pvals = np.array(pvals)
		print('{} ROIs are significant at p<0.05 (corrected)'.format(np.sum(pvals > 0)))
	return df_significant


def launch_permutation(X, Y, algo, labels, n_permutations=100):
	perm_rs = np.random.RandomState(0)
	permutation_accs = []
	permutation_coefs = []
	n_permutations = n_permutations
	print("running {} permutations".format(n_permutations))
	for i_iter in range(n_permutations):
		if i_iter%25 == 0:
			print("Turn {} / {}".format(i_iter, n_permutations))
		Y_perm = perm_rs.permutation(Y)
		sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=i_iter)
		sss.get_n_splits(X, Y_perm)
		permutation_accs_cv = []
		permutation_coefs_cv = []
		for train_index, test_index in sss.split(X, Y_perm):
			# fit algorithm on training data
			algo.fit(X[train_index], Y_perm[train_index])
			# compute accuracies on testing data
			acc = algo.score(X[test_index], Y_perm[test_index])
			permutation_accs_cv.append(acc)
			permutation_coefs_cv.append(algo.coef_)
		permutation_accs.append(permutation_accs_cv)
		permutation_coefs.append(permutation_coefs_cv)
	return permutation_accs, permutation_coefs

# debugging
def class_proportions(y):
	prop1, prop2, prop3 = np.unique(y, return_counts=True)[1]*100/len(y) 
	print("Class 1 :", prop1, "%")
	print("Class 2 :", prop2, "%")
	print("Class 3 :", prop3, "%")


def runs(X, Y, algo, n_runs, labels):
	# keep information for the confusion matrix
	all_y_pred = []
	all_y_true = []
	all_accs = []
	# save accuracies and model coefficients
	accs = []
	coefs = []
	print("Running {} subsamples".format(n_runs))
	# run as many times as inputed to ensure stability
	for i_subsample in range(n_runs):
		if i_subsample%25 == 0:
			print("** Turn {} / {}".format(i_subsample, n_runs))
		folder = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=i_subsample)
		folder.get_n_splits(X, Y)
		# # debug
		# print("full sample proportion :")
		# class_proportions(y)
		# print("CV proportion :")
		sample_acc = []
		sample_coefs = []
		for train_inds, test_inds in folder.split(X, Y):
			acc, coef, y_pred, y_true = logRegMulti(algo, X, Y, train_inds, test_inds)
			# # debug
			# class_proportions(y[train_inds])
			# class_proportions(y[test_inds])
			# print("test_indexes: ", test_inds)
			all_y_pred.append(y_pred)
			all_y_true.append(y_true)
			sample_acc.append(acc)
			all_accs.append(acc)
			sample_coefs.append(coef)
		coefs.append(np.mean(sample_coefs, axis=0))
		accs.append(np.mean(sample_acc))
	all_y_pred = np.array(all_y_pred).reshape(-1)
	all_y_true = np.array(all_y_true).reshape(-1)
	final_coefs = np.mean(coefs, axis=0)
	final_acc = np.mean(accs)
	# store into dataframe:
	df_coefs = pd.DataFrame(columns=labels, index=['FP', 'Amnestic', 'Mixed'], data=final_coefs)
	return all_y_pred, all_y_true, df_coefs, final_acc, all_accs, sample_coefs


# fit and score multinomial log reg
def logRegMulti(algo, X, Y, train_inds, test_inds):
	# catch warnings
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		# fit algorithm on training data
		algo.fit(X[train_inds], Y[train_inds])
	# compute accuracies on testing data
	acc = algo.score(X[test_inds], Y[test_inds])
	y_pred = algo.predict(X[test_inds])
	# save accuracies
	# save coefficients
	# save prediction and real Y for confusion matrix
	return [acc, algo.coef_, y_pred, Y[test_inds]]


# load required dataframe
# grey matter quantity per roi
df = pd.read_pickle('_pickles/gm_cleaned_atlas_harvard_oxford_combat')
labels = df.columns.values
# group label
cluster = pd.read_excel('_createdDataframes/df_scores_std.xlsx', index_col=0)['cluster']
df['cluster'] = cluster
# get rid of smc subject
df = df[df['cluster'] != 'SMC']
df = df[df['cluster'] != 'CN']
X = df[df.columns[:-1]].values
y = df['cluster']

# categorize y
# y[y == 'CN'] = 0
y[y == 'Low'] = 2
y[y == 'Middle'] = 1
y[y == 'High'] = 0
y = y.values.astype('int')


#########################
# ONE VERSUS REST
#########################

algo = LogisticRegression(class_weight='balanced',
        solver="sag", max_iter=500, random_state=42, multi_class="ovr")
run = 1000
all_y_pred, all_y_true, df_coefs, final_acc, all_accs, sample_coefs = runs(X, y, algo, run, labels)
permutation_accs, permutation_coefs = launch_permutation(X, y, algo, labels, n_permutations=run)
testing_acc = accuracy_testing(permutation_accs, final_acc) # percentil: 98.34
df_significant = coefficent_testing(labels, permutation_coefs, df_coefs)
df_significant.to_excel("_createdDataframes/df_significant_ovr.xlsx")
# print the training scores
print("training score :{}, {}".format(final_acc , "OVR"))


df_coefs.to_excel("_createdDataframes/df_coefs_ovr.xlsx")

results_dic = {"all_y_pred":all_y_pred, "all_y_true":all_y_true, "final_acc":final_acc, "all_accs":all_accs, "sample_coefs":sample_coefs, "testing_acc":testing_acc}
np.save('_pickles/OVR_results_data.npy', results_dic) 
print(final_acc, np.std(all_accs), np.mean(all_accs))



## CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import itertools
# compute and plot the confusion matrix
f, ax = plt.subplots(figsize=(8, 8))
class_names = ['Cluster-derived normal MCI', 'Amnestic MCI', 'Mixed MCI']
# matrix
cm = confusion_matrix(all_y_true, all_y_pred)
cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = (cm_ * 100) # percentage
for indx, i in enumerate(cm):
	for indy, j in enumerate(i):
		j = round(j, 1)
		cm[indx, indy] = j
plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Reds)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, fontsize=20)
plt.yticks(tick_marks, class_names, fontsize=20)
rotateTickLabels(ax, -55, 'x')
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j]) + "%",
             horizontalalignment="center",
             color= "black", fontsize=20)
plt.xlabel('Predicted label', fontsize=25)
plt.ylabel('True label (ovr)', fontsize=25)
plt.tight_layout()
plt.savefig('_figures/confusion_matrix_ovr.png')
plt.show()




#########################
# ONE VERSUS REST CSF
#########################
# load required dataframe
# grey matter quantity per roi
df = pd.read_pickle('_pickles/csf_level_std_cleaned')
labels = df.columns[:-1].values

# get rid of smc subject
df = df[df['cluster'] != 'SMC']
df = df[df['cluster'] != 'CN']
X = df[df.columns[:-1]].values
y = df['cluster']

# categorize y
# y[y == 'CN'] = 0
y[y == 'Low'] = 2
y[y == 'Middle'] = 1
y[y == 'High'] = 0
y = y.values.astype('int')


algo = LogisticRegression(class_weight='balanced',
        solver="sag", max_iter=500, random_state=42, multi_class="ovr")
run = 1000
all_y_pred, all_y_true, df_coefs, final_acc, all_accs, sample_coefs = runs(X, y, algo, run, labels)
permutation_accs, permutation_coefs = launch_permutation(X, y, algo, labels, n_permutations=run)
testing_acc = accuracy_testing(permutation_accs, final_acc) # percentil: 98.34
df_significant = coefficent_testing(labels, permutation_coefs, df_coefs)
df_significant.to_excel("_createdDataframes/df_significant_ovr_csf.xlsx")
# print the training scores
print("training score :{}, {}".format(final_acc , "ovr"))
# final acc = 0.45886206896551723

df_coefs.to_excel("_createdDataframes/df_coefs_ovr.xlsx")

results_dic = {"all_y_pred":all_y_pred, "all_y_true":all_y_true, "final_acc":final_acc, "all_accs":all_accs, "sample_coefs":sample_coefs, "testing_acc":testing_acc}
np.save('_pickles/OVR_results_data.npy', results_dic) 
print(final_acc, np.std(all_accs), np.mean(all_accs))
print(testing_acc)

## CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import itertools
# compute and plot the confusion matrix
f, ax = plt.subplots(figsize=(8, 8))
class_names = ['Cluster-derived normal MCI', 'Amnestic MCI', 'Mixed MCI']
# matrix
cm = confusion_matrix(all_y_true, all_y_pred)
cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = (cm_ * 100) # percentage
for indx, i in enumerate(cm):
	for indy, j in enumerate(i):
		j = round(j, 1)
		cm[indx, indy] = j
plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Reds)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, fontsize=20)
plt.yticks(tick_marks, class_names, fontsize=20)
rotateTickLabels(ax, -55, 'x')
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j]) + "%",
             horizontalalignment="center",
             color= "black", fontsize=20)
plt.xlabel('Predicted label', fontsize=25)
plt.ylabel('True label (ovr)', fontsize=25)
plt.tight_layout()
plt.savefig('_figures/confusion_matrix_ovr_csf.png')
plt.show()

