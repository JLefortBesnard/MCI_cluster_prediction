
import pandas as pd
import numpy as np
# script to extract grey matter volume per ROI using Yeo atlas.
# standardize the data all together and remove variance from TIV and age.
######
# get rid, diagnosis group + cluster of participants
######

df_cluster = pd.read_excel('_createdDataframe/df_MCI_cluster.xlsx', index_col=0)
df = pd.read_excel('_createdDataframe/df_scores_std.xlsx', index_col=0)

# get scores per cluster
df_0 = df_cluster[df_cluster['cluster'] == 0]
df_1 = df_cluster[df_cluster['cluster'] == 1]
df_2 = df_cluster[df_cluster['cluster'] == 2]

cluster_df = [df_0, df_1, df_2]
mean_cls = np.array([df_0.mean(axis=0).mean(), df_1.mean(axis=0).mean(), df_2.mean(axis=0).mean()])
index_min = np.argmin(mean_cls)
df_cluster['cluster'][df_cluster['cluster'] == index_min] = 'Low'
index_max = np.argmax(mean_cls)
df_cluster['cluster'][df_cluster['cluster'] == index_max] = 'High'
index_middle = min(set([0, 1, 2]) - set([index_min, index_max])) # min to extract from set
df_cluster['cluster'][df_cluster['cluster'] == index_middle] = 'Middle'


df['Group'].loc[list(df_cluster['cluster'].index)] = df_cluster['cluster']

df=df[['Group', 'smwc1_path', 'smwc2_path', 'smwc3_path', 'Age']]

######
# get grey matter
######

df.to_pickle('_pickles/df_mri_per_group')

LowMCIfiles = df[df['Group'] == 'Low']
MiddleMCIfiles = df[df['Group'] == 'Middle']
HighMCIfiles = df[df['Group'] == 'High']
CNfiles = df[df['Group'] == 'CN']
SMCfiles = df[df['Group'] == 'SMC']



# from https://nilearn.github.io/dev/auto_examples/02_decoding/plot_oasis_vbm.html#inference-with-massively-univariate-model
from nilearn.mass_univariate import permuted_ols
from sklearn.feature_selection import VarianceThreshold
from nilearn.input_data import NiftiMasker
from nilearn.image import get_data
from nilearn.masking import apply_mask
from nilearn.masking import unmask
from nilearn.masking import compute_background_mask
from nilearn.plotting import plot_stat_map
from nilearn.plotting import plot_anat
np.random.seed(0)


class mass_univariate_inference:
	def __init__(self, X, Y):
		self.mask_img = 'C:\\Users\\lefortb211\\Downloads\\ADNI1_smwc_files\\smwc1\\smwc1ADNI_002_S_0729_MR_MP-RAGE__br_raw_20060718014510277_55_S16874_I19057.nii'
		self.X = X 
		self.Y = Y.astype('int')

	def mask_one_by_one(self):
		print('Compute mask one by one...')
		self.mask_img_masked = compute_background_mask(self.mask_img)
		print('Apply mask to {} participants...'.format(len(self.X)))
		print('       Estimated time for 10 scans = 20 seconds')
		self.gm_maps_masked = apply_mask(self.X[0], self.mask_img_masked)
		self.gm_maps_masked = self.gm_maps_masked.reshape(1, -1)
		for i in range(1, len(self.X)):
			print('Turn = {}/{}, shape = {}'.format(i, len(self.X), self.gm_maps_masked.shape))
			self.gm_maps_masked = np.append(self.gm_maps_masked, apply_mask(self.X[i], self.mask_img_masked).reshape(1, -1), axis=0)
			np.save('_niftiFiles/CN_low_masked.npy', self.gm_maps_masked)
		print("Success")
		print("   ")


	def threshold_variance(self, thresh):
		print('removing low variance...')
		self.variance_threshold = VarianceThreshold(threshold=thresh)
		gm_maps_thresholded = self.variance_threshold.fit_transform(self.gm_maps_masked)
		self.gm_maps_thresholded = gm_maps_thresholded
		self.gm_maps_masked = 0 # reset memory
		print("Success")
		print("Old shape: ", self.gm_maps_masked.shape)
		print("New shape: ", self.gm_maps_thresholded.shape)
		print("   ")


	def compute_test(self, confounds, saving_path):
		print('Statistical inference...')
		neg_log_pvals, t_scores_original_data, _ = permuted_ols(
			self.Y, self.gm_maps_thresholded,  
			confounding_vars=confounds,# + intercept as a covariate by default
			n_perm=2000,  # 1,000 in the interest of time; 10000 would be better
			verbose=1, # display progress bar
			n_jobs=1)  # can be changed to use more CPUs
		self.neg_log_pvals = neg_log_pvals
		self.t_scores_original_data = t_scores_original_data
		print('Get significant voxels...')
		self.signed_neg_log_pvals = neg_log_pvals * np.sign(t_scores_original_data)
		print('Unmasking...')
		signed_neg_log_pvals_unmasked = unmask(self.variance_threshold.inverse_transform(self.signed_neg_log_pvals), 
			self.mask_img_masked)
		print('Showing results...')
		threshold = -np.log10(0.1)  # 10% corrected
		n_detections = (get_data(signed_neg_log_pvals_unmasked) > threshold).sum()
		print('\n%d detections' % n_detections)
		print('saving...')
		signed_neg_log_pvals_unmasked.to_filename(saving_path) # save as nii
		self.signed_neg_log_pvals_unmasked = signed_neg_log_pvals_unmasked
		print("Success")
		print("   ")

######################################
# analysis with CN and LowMCI files: #
######################################

# check running time
import time
startTime = time.time()
both_files = pd.concat([LowMCIfiles, CNfiles])
# creating a dict file 
status = {'CN': 1,'Low': 0}
both_files['Group'] = [status[item] for item in both_files['Group']]
X = both_files['smwc1_path'].values
Y = both_files['Group'].values

confounds = both_files[['Age', 'TIV']].values.astype('float64')
CN_low = mass_univariate_inference(X, Y)
CN_low.mask_one_by_one() # need 13 Go de RAM ...
executionTime = (time.time() - startTime)
print('Execution time to extract gm, run GLM and save results in seconds: ' + str(executionTime))

file = np.load("_niftiFiles/CN_low_masked.npy")







stop



# checking
from nilearn import image
plot_anat(X[5])
plt.show()
CN_low.gm_maps_masked[5]
a = unmask(CN_low.gm_maps_masked, CN_low.mask_img_masked)
first_rsn = image.index_img(a, 5)
plot_anat(first_rsn)
plt.show()
assert nib.load(X[5]).get_data()[80][80][10] == first_rsn.get_data()[80][80][10]

gm_maps_thresholded = CN_low.threshold_variance(.01)
signed_neg_log_pvals_unmasked = CN_low.compute_test(confounds, "_niftiFiles/MassUnivariate_CN-Low.nii")


executionTime = (time.time() - startTime)
print('Execution time to extract gm, run GLM and save results in seconds: ' + str(executionTime))
# minutes


#########################################
# analysis with CN and MiddleMCI files: #
#########################################

# check running time
import time
startTime = time.time()
both_files = pd.concat([MiddleMCIfiles, CNfiles])
# creating a dict file 
status = {'CN': 1,'Middle': 0}
both_files['Group'] = [status[item] for item in both_files['Group']]
X = both_files['smwc1_path'].values
Y = both_files['Group'].values

confounds = both_files[['Age', 'TIV']].values.astype('float64')
signed_neg_log_pvals_unmasked = mass_univariate_inference(X, Y, confounds,"MassUnivariate_CN-Middle.nii")

executionTime = (time.time() - startTime)
print('Execution time to extract gm, run GLM and save results in seconds: ' + str(executionTime))
# minutes



#######################################
# analysis with CN and HighMCI files: #
#######################################

# check running time
import time
startTime = time.time()
both_files = pd.concat([HighMCIfiles, CNfiles])
# creating a dict file 
status = {'CN': 1,'High': 0}
both_files['Group'] = [status[item] for item in both_files['Group']]
X = both_files['smwc1_path'].values
Y = both_files['Group'].values

confounds = both_files[['Age', 'TIV']].values.astype('float64')
signed_neg_log_pvals_unmasked = mass_univariate_inference(X, Y, confounds,"MassUnivariate_CN-High.nii")

executionTime = (time.time() - startTime)
print('Execution time to extract gm, run GLM and save results in seconds: ' + str(executionTime))
# 4 minutes



#######################################
# analysis with HighMCI and MiddleMCI files: #
#######################################

# check running time
import time
startTime = time.time()
both_files = pd.concat([HighMCIfiles, MiddleMCIfiles])
# creating a dict file 
status = {'High': 1,'Middle': 0}
both_files['Group'] = [status[item] for item in both_files['Group']]
X = both_files['smwc1_path'].values
Y = both_files['Group'].values

confounds = both_files[['Age', 'TIV']].values.astype('float64')
signed_neg_log_pvals_unmasked = mass_univariate_inference(X, Y, confounds,"MassUnivariate_High-Middle.nii")

executionTime = (time.time() - startTime)
print('Execution time to extract gm, run GLM and save results in seconds: ' + str(executionTime))
#  minutes


#######################################
# analysis with HighMCI and LowMCI files: #
#######################################

# check running time
import time
startTime = time.time()
both_files = pd.concat([HighMCIfiles, LowMCIfiles])
# creating a dict file 
status = {'High': 1,'Low': 0}
both_files['Group'] = [status[item] for item in both_files['Group']]
X = both_files['smwc1_path'].values
Y = both_files['Group'].values

confounds = both_files[['Age', 'TIV']].values.astype('float64')
signed_neg_log_pvals_unmasked = mass_univariate_inference(X, Y, confounds,"MassUnivariate_High-Low.nii")

executionTime = (time.time() - startTime)
print('Execution time to extract gm, run GLM and save results in seconds: ' + str(executionTime))
# 4 minutes


#######################################
# analysis with MiddleMCI and LowMCI files: #
#######################################

# check running time
import time
startTime = time.time()
both_files = pd.concat([HighMCIfiles, MiddleMCIfiles])
# creating a dict file 
status = {'High': 1,'Middle': 0}
both_files['Group'] = [status[item] for item in both_files['Group']]
X = both_files['smwc1_path'].values
Y = both_files['Group'].values

confounds = both_files[['Age', 'TIV']].values.astype('float64')
signed_neg_log_pvals_unmasked = mass_univariate_inference(X, Y, confounds,"MassUnivariate_High-Middle.nii")

executionTime = (time.time() - startTime)
print('Execution time to extract gm, run GLM and save results in seconds: ' + str(executionTime))
# 4 minutes