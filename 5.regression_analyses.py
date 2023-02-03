# Compute regression analyses for each MCI subgroup versus controls using GM or CSF level


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import time
import itertools
from scipy import stats #v1.5.2
import nibabel as nib
from nilearn import datasets as ds
from nilearn.image import resample_img
from nilearn.input_data import NiftiLabelsMasker

np.random.seed(0)

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

#################################################
# compute logistic regression for each category #
# PREDICTION WITH GM #
#################################################


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

class prediction:
	def __init__(self, cluster0, cluster1, df_gm):
		self.label0 = cluster0
		self.label1 = cluster1
		self.df = df_gm.query('cluster == "{}" or cluster == "{}"'.format(cluster0, cluster1))
		self.N0 = len(self.df[self.df["cluster"] == self.label0])
		self.N1 = len(self.df[self.df["cluster"] == self.label1])
		print('')
		print('Running {} (n={}) vs {} (n={})'.format(self.label0, self.N0, self.label1, self.N1))
		print('')
		self.X = self.df[self.df.columns[:-1]].values
		self.Y = self.df[self.df.columns[-1]]
		# categorize
		self.Y[self.Y == self.label0] = 0
		self.Y[self.Y == self.label1] = 1
		self.Y = self.Y.values.astype('int')
		self.labels = self.df.columns[:-1].values
		self.title = "{}_vs_{}".format(self.label0, self.label1)

	def logReg(self, train_inds, test_inds):
		clf = LogisticRegression(penalty='l2', class_weight='balanced')
		# fit algorithm on training data
		clf.fit(self.X[train_inds], self.Y[train_inds])
		# compute accuracies on testing data
		acc = clf.score(self.X[test_inds], self.Y[test_inds])
		y_pred = clf.predict(self.X[test_inds])
		# save accuracies
		# save coefficients
		# save prediction and real Y for confusion matrix
		return [acc, clf.coef_[0, :], y_pred, self.Y[test_inds]]

	def runs(self, n_runs):
		# keep information for the confusion matrix
		self.all_y_pred = []
		self.all_y_true = []
		self.all_accs = []
		# save accuracies and model coefficients
		self.accs = []
		self.coefs = []
		print("Running {} subsamples".format(n_runs))
		# run as many times as inputed to ensure stability
		for i_subsample in range(n_runs):
			if i_subsample%25 == 0:
				print("Turn {} / {}".format(i_subsample, n_runs))
			folder = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=i_subsample)
			folder.get_n_splits(self.X, self.Y)
			sample_acc = []
			sample_coefs = []
			for train_inds, test_inds in folder.split(self.X, self.Y):
				acc, coef, y_pred, y_true = self.logReg(train_inds, test_inds)
				self.all_y_pred.append(y_pred)
				self.all_y_true.append(y_true)
				sample_acc.append(acc)
				self.all_accs.append(acc)
				sample_coefs.append(coef)
			self.coefs.append(np.mean(sample_coefs, axis=0))
			self.accs.append(np.mean(sample_acc))
		self.all_y_pred = np.array(self.all_y_pred).reshape(-1)
		self.all_y_true = np.array(self.all_y_true).reshape(-1)
		self.final_coefs = np.mean(self.coefs, axis=0)
		self.final_acc = np.mean(self.accs)

		# store into dataframe:
		self.df_coefs = pd.DataFrame(columns=['Label', 'coef'], data=np.array([self.labels, self.final_coefs]).T)
		
	def plot_confusion(self, csf=0):
		# compute and plot the confusion matrix
		f, ax = plt.subplots(figsize=(8, 8))
		class_names = [self.label0, self.label1]
		# matrix
		cm = confusion_matrix(self.all_y_true, self.all_y_pred)
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
		plt.ylabel('True label', fontsize=25)
		plt.tight_layout()
		if csf==1:
			plt.savefig('_figures/csf_confusion_matrix_{}.png'.format(self.title))
		else:
			plt.savefig('_figures/confusion_matrix_{}.png'.format(self.title))
		plt.close('all')

	def lauch_permutation(self, n_permutations):
		perm_rs = np.random.RandomState(0)
		self.permutation_accs = []
		self.permutation_coefs = []
		self.n_permutations = n_permutations
		print("running {} permutations".format(n_permutations))
		for i_iter in range(n_permutations):
			if i_iter%25 == 0:
				print("Turn {} / {}".format(i_iter, n_permutations))
			Y_perm = perm_rs.permutation(self.Y)
			sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=i_iter)
			sss.get_n_splits(self.X, Y_perm)
			permutation_accs_cv = []
			permutation_coefs_cv = []
			for train_index, test_index in sss.split(self.X, Y_perm):
				clf = LogisticRegression(penalty='l2', class_weight='balanced')
				# fit algorithm on training data
				clf.fit(self.X[train_index], Y_perm[train_index])
				# compute accuracies on testing data
				acc = clf.score(self.X[test_index], Y_perm[test_index])
				permutation_accs_cv.append(acc)
				permutation_coefs_cv.append(clf.coef_[0, :])
			self.permutation_accs.append(permutation_accs_cv)
			self.permutation_coefs.append(permutation_coefs_cv)
		self.permutation_accs = np.array(self.permutation_accs).reshape(-1)
		self.permutation_coefs = np.array(self.permutation_coefs).reshape(-1, len(self.labels))

	def coefficent_testing(self):
		# extract permutation coef per ROI for hypothesis testing
		self.coeff_per_roi = []
		for roi_nb in range(len(self.labels)):
			coeff_roi = []
			for coef_iter in self.permutation_coefs:
				coeff_roi.append(coef_iter[roi_nb])
			self.coeff_per_roi.append(coeff_roi)
		self.coeff_per_roi = np.array(self.coeff_per_roi)
		assert self.coeff_per_roi.shape[0] == len(self.labels)
		assert self.coeff_per_roi.shape[1] == self.n_permutations*5 # 5CV * permutation number

		# extract 95 and 5% percentile and check if original weight outside these limits to check for significance
		pvals = []
		self.df_significant = pd.DataFrame(index=self.labels, columns=['coef'])
		for n_roi in range(len(self.labels)):
			coeff_roi = self.coeff_per_roi[n_roi] 
			above = stats.scoreatpercentile(coeff_roi, 99)
			below = stats.scoreatpercentile(coeff_roi, 1)
			if self.final_coefs[n_roi] < below or self.final_coefs[n_roi] > above:
				pvals.append(1)
				self.df_significant.loc[self.labels[n_roi]] = self.final_coefs[n_roi]
			else:
				pvals.append(0)
				self.df_significant.loc[self.labels[n_roi]] = 0
		pvals = np.array(pvals)
		print('{} ROIs are significant at p<0.05 (corrected)'.format(np.sum(pvals > 0)))


	def accuracy_testing(self):
		# check if accuracy is significant
		above = stats.scoreatpercentile(self.permutation_accs, 99)
		percentile_of_acc = stats.percentileofscore(self.permutation_accs, self.final_acc)
		if self.final_acc > above:
			return True, percentile_of_acc
		else:
			return False, percentile_of_acc

	def print_results(self):
		print(("****"))
		print("RESULTS FROM ANALYSIS : ", self.title)
		print("Main accuracy : Mean = ", np.mean(self.accs), " Std = ", np.std(self.accs))
		print("Is accuracy significant => {}, percentile={} ".format(self.accuracy_testing()[0],self.accuracy_testing()[1]))
		print(("---"))
		print("Highest (absolute) weights : ")
		outlier_limit = np.mean(self.df_coefs['coef']**2) + np.std(self.df_coefs['coef']**2) # 1 std from the mean (squared) coefficient value
		print(self.df_coefs.loc[self.df_coefs['coef']**2 >= outlier_limit])
		print(("---"))
		print("Significant ROIs : ")
		print(self.df_significant[self.df_significant['coef'] != 0])

	def saving(self, smwc1_path, csf=0):
		if csf==1:
			# save as df
			self.df_significant.to_pickle('_pickles/csf_predictions/df_significant_csf_{}'.format(self.title))
			self.df_coefs.to_pickle('_pickles/csf_predictions/df_coefs_csf_{}'.format(self.title))
		else:
			# save as df
			self.df_significant.to_pickle('_pickles/gm_predictions/df_significant_roi_{}'.format(self.title))
			self.df_coefs.to_pickle('_pickles/gm_predictions/df_coefs_roi_{}'.format(self.title))
			# save as nifit
			atlas = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',  symmetric_split=True)
			nii = nib.load(smwc1_path)
			ratlas_nii = resample_img(
				atlas.maps, target_affine=nii.affine, interpolation='nearest')
			masker = NiftiLabelsMasker(labels_img=ratlas_nii, standardize=False, strategy='sum')
			cur_FS = masker.fit_transform(nii)
			# save significant weigths
			sign_coefs_nii = masker.inverse_transform(np.array(self.df_significant['coef'].values.astype('float64')).reshape(1, 96))
			sign_coefs_nii.to_filename("_niftiFiles/{}_significant.nii".format(self.title)) # transform as nii and save
			# save significant weigths
			coefs_nii = masker.inverse_transform(np.array(self.df_coefs['coef'].values.astype('float64')).reshape(1, 96))
			coefs_nii.to_filename("_niftiFiles/{}_coeffs.nii".format(self.title)) # transform as nii and save



def run_all_steps(Y0, Y1, df):
	analysis = prediction(Y0, Y1, df)
	print("   ***    ")
	print("STARTING ", analysis.title)
	print("   ***    ")
	time.sleep(2)
	analysis.runs(100)
	analysis.plot_confusion()
	analysis.lauch_permutation(100)
	analysis.coefficent_testing()
	analysis.print_results()
	smwc1_path = pd.read_excel('_createdDataframe/mri_paths.xlsx')['smwc1_path'].iloc[0]
	analysis.saving(smwc1_path)
	return analysis



###############################################################
# run prediction analysis per group with gm per roi as inputs #
###############################################################
np.random.seed(0)

# smc vs cn
analysis_1 = run_all_steps('SMC', 'CN', df)
# high vs cn
analysis_2 = run_all_steps('High', 'CN', df)
# middle vs cn
analysis_3 = run_all_steps('Middle', 'CN', df)
# Low vs cn
analysis_4 = run_all_steps('Low', 'CN', df)
# middle vs high
analysis_5 = run_all_steps('Middle', 'High', df)
# Low vs high
analysis_6 = run_all_steps('Low', 'High', df)
# Low vs middle
analysis_7 = run_all_steps('Low', 'Middle', df)

analyses = [analysis_1, analysis_2, analysis_3, analysis_4, analysis_5, analysis_6, analysis_7]
dict_visualisation = {}
for analysis in analyses:
	print(analysis.title)
	dict_visualisation[analysis.title] = analysis.all_accs
	print('{} (n={}) vs {} (n={})'.format(analysis.label0, analysis.N0, analysis.label1, analysis.N1))
	print("Main accuracy : Mean = ", np.mean(analysis.accs), " Std = ", np.std(analysis.accs))
	print("Is accuracy significant => {}, percentile={} ".format(analysis.accuracy_testing()[0],analysis.accuracy_testing()[1]))

df_visu = pd.DataFrame.from_dict(dict_visualisation)
df_visu.to_pickle("_pickles/df_visu_gm")


########################################################
# run prediction analysis per group with csf as inputs #
########################################################

np.random.seed(0)

df[['TAU', 'PTAU', 'ABETA']] = 0
df_csf1 = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/CSF.csv")
df_csf2 = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/csf_adni1_go_2.csv")
col_csf_to_keep = ['RID', 'ABETA', 'TAU', 'PTAU']
df_csf1 = df_csf1[col_csf_to_keep]
df_csf2 = df_csf2[col_csf_to_keep]
df_csf = pd.concat((df_csf1, df_csf2))
df_csf = df_csf.set_index("RID")

missing=[]
for index in df.index:
    success = 0
    try:
        df_csf[['TAU', 'PTAU', 'ABETA']].loc[index]
        success = 1
    except:
        print('no data for ', index)
        missing.append(index)
    if success == 1:
        if len(df_csf[['TAU']].loc[index]) > 1:
            df.loc[index, ['TAU', 'PTAU', 'ABETA']] = df_csf[['TAU', 'PTAU', 'ABETA']].loc[index].mean().values
        else:
            df.loc[index, ['TAU', 'PTAU', 'ABETA']] = df_csf[['TAU', 'PTAU', 'ABETA']].loc[index].values
print('missing {} participants'.format(len(missing)))
np.unique(df.loc[missing]['cluster'], return_counts=True)
# np.unique(df.loc[missing]['Session'], return_counts=True)

df_ = df[['TAU', 'PTAU', 'ABETA', 'cluster']].loc[(df[['TAU', 'PTAU', 'ABETA']]!=0).any(axis=1)]
df_ = df_.dropna()
df_ = df_[['TAU', 'PTAU', 'ABETA', 'cluster']] # 708
df_.to_pickle('_pickles/csf_level')
# standardize
data = StandardScaler().fit_transform(df_[df_.columns[:-1]].values)
df_[df_.columns[:-1]] = data
df_.to_pickle('_pickles/csf_level_std')

def run_all_steps(Y0, Y1, df):
	analysis = prediction(Y0, Y1, df)
	print("   ***    ")
	print("STARTING ", analysis.title)
	print("   ***    ")
	time.sleep(2)
	analysis.runs(100)
	analysis.plot_confusion(csf=1)
	analysis.lauch_permutation(100)
	analysis.coefficent_testing()
	analysis.print_results()
	smwc1_path = 'fake'
	analysis.saving(smwc1_path, csf=1)
	return analysis


analysis = prediction('SMC', 'CN', df_)
# smc vs cn
analysis_1 = run_all_steps('SMC', 'CN', df_)
# high vs cn
analysis_2 = run_all_steps('High', 'CN', df_)
# middle vs cn
analysis_3 = run_all_steps('Middle', 'CN', df_)
# Low vs cn
analysis_4 = run_all_steps('Low', 'CN', df_)
# middle vs high
analysis_5 = run_all_steps('Middle', 'High', df_)
# Low vs high
analysis_6 = run_all_steps('Low', 'High', df_)
# Low vs middle
analysis_7 = run_all_steps('Low', 'Middle', df_)


dict_visualisation = {}
analyses = [analysis_1, analysis_2, analysis_3, analysis_4, analysis_5, analysis_6, analysis_7]
for analysis in analyses:
	print(analysis.title)
	print('{} (n={}) vs {} (n={})'.format(analysis.label0, analysis.N0, analysis.label1, analysis.N1))
	print("Main accuracy : Mean = ", np.mean(analysis.accs), " Std = ", np.std(analysis.accs))
	print("Is accuracy significant => {}, percentile={} ".format(analysis.accuracy_testing()[0],analysis.accuracy_testing()[1]))
	print("Is accuracy significant => {}, percentile={} ".format(analysis.accuracy_testing()[0],analysis.accuracy_testing()[1]))
	print("Significant coefficients => {} ".format(analysis.df_significant))
	dict_visualisation[analysis.title] = analysis.all_accs
df_visu = pd.DataFrame.from_dict(dict_visualisation)
df_visu.to_pickle("_pickles/df_visu_csf")




#### VISUALISATION NIFTI FOR ARTICLE

# with standardized gm data

# save mean (standardized value) of brain fro each group
df = pd.read_pickle('_pickles/gm_cleaned_atlas_harvard_oxford')
# group label
cluster = pd.read_excel('_createdDataframe/df_scores_std.xlsx', index_col=0)['cluster']
df['cluster'] = cluster

low = df[df['cluster'] == 'Low'].mean().values.astype('float64').reshape(1, 96)+1
high = df[df['cluster'] == 'High'].mean().values.astype('float64').reshape(1, 96)+1
middle = df[df['cluster'] == 'Middle'].mean().values.astype('float64').reshape(1, 96)+1
CN = df[df['cluster'] == 'CN'].mean().values.astype('float64').reshape(1, 96)+1
atlas = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',  symmetric_split=True)
smwc1_path = pd.read_excel('_createdDataframe/mri_paths.xlsx')['smwc1_path'].iloc[0]
nii = nib.load(smwc1_path)
ratlas_nii = resample_img(
	atlas.maps, target_affine=nii.affine, interpolation='nearest')
masker = NiftiLabelsMasker(labels_img=ratlas_nii, standardize=False, strategy='sum')
cur_FS = masker.fit_transform(nii)
# save significant weigths
_nii = masker.inverse_transform(low)
_nii.to_filename("_niftiFiles/low_mean_std.nii") # transform as nii and save
_nii = masker.inverse_transform(middle)
_nii.to_filename("_niftiFiles/middle_mean_std.nii") # transform as nii and save
_nii = masker.inverse_transform(CN)
_nii.to_filename("_niftiFiles/CN_mean_std.nii") # transform as nii and save


# with standardized gm data
# save mean CN - mean group of brain for each MCI group
df = pd.read_pickle('_pickles/gm_cleaned_atlas_harvard_oxford')
# group label
cluster = pd.read_excel('_createdDataframe/df_scores_std.xlsx', index_col=0)['cluster']
df['cluster'] = cluster

CN = df[df['cluster'] == 'CN'].mean().values.astype('float64').reshape(1, 96)+1
low = df[df['cluster'] == 'Low'].mean().values.astype('float64').reshape(1, 96)+1
low = CN - low
high = df[df['cluster'] == 'High'].mean().values.astype('float64').reshape(1, 96)+1
high = CN - high
middle = df[df['cluster'] == 'Middle'].mean().values.astype('float64').reshape(1, 96)+1
middle = CN - middle
atlas = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',  symmetric_split=True)
smwc1_path = pd.read_excel('_createdDataframe/mri_paths.xlsx')['smwc1_path'].iloc[0]
nii = nib.load(smwc1_path)
ratlas_nii = resample_img(
	atlas.maps, target_affine=nii.affine, interpolation='nearest')
masker = NiftiLabelsMasker(labels_img=ratlas_nii, standardize=False, strategy='sum')
cur_FS = masker.fit_transform(nii)
# save significant weigths
_nii = masker.inverse_transform(low)
_nii.to_filename("_niftiFiles/low_minusCN_std.nii") # transform as nii and save
_nii = masker.inverse_transform(middle)
_nii.to_filename("_niftiFiles/middle_minusCN_std.nii") # transform as nii and save
_nii = masker.inverse_transform(high)
_nii.to_filename("_niftiFiles/high_minusCN_std.nii") # transform as nii and save


# with raw gm data
# save mean CN - mean group of brain for each MCI group
df = pd.read_pickle('_pickles/gm_atlas_harvard_oxford')
# group label
cluster = pd.read_excel('_createdDataframe/df_scores_std.xlsx', index_col=0)['cluster']
df['cluster'] = cluster
CN = df[df['cluster'] == 'CN'].mean().values.astype('float64').reshape(1, 96)
low = df[df['cluster'] == 'Low'].mean().values.astype('float64').reshape(1, 96)
low = CN - low
high = df[df['cluster'] == 'High'].mean().values.astype('float64').reshape(1, 96)
high = CN - high
middle = df[df['cluster'] == 'Middle'].mean().values.astype('float64').reshape(1, 96)
middle = CN - middle


atlas = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',  symmetric_split=True)
smwc1_path = pd.read_excel('_createdDataframe/mri_paths.xlsx')['smwc1_path'].iloc[0]
nii = nib.load(smwc1_path)
ratlas_nii = resample_img(
	atlas.maps, target_affine=nii.affine, interpolation='nearest')
masker = NiftiLabelsMasker(labels_img=ratlas_nii, standardize=False, strategy='sum')
cur_FS = masker.fit_transform(nii)
# save significant weigths
_nii = masker.inverse_transform(low)
_nii.to_filename("_niftiFiles/low_minusCN.nii") # transform as nii and save
_nii = masker.inverse_transform(middle)
_nii.to_filename("_niftiFiles/middle_minusCN.nii") # transform as nii and save
_nii = masker.inverse_transform(high)
_nii.to_filename("_niftiFiles/high_minusCN.nii") # transform as nii and save