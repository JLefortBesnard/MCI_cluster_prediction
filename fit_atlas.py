"""
Extract grey matter volume per ROI using Yeo atlas.
Then standardize + remove variance from TIV and age and MRI scanner magnet strength.
Save ouputs as pickles in _pickles/
2023
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
duration : 3 hours 16 minutes
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import psutil
# neuroimaging
import nibabel as nib
from nilearn import datasets as ds
from nilearn import plotting
from nilearn.image import resample_img
from nilearn.input_data import NiftiLabelsMasker
from nilearn.signal import clean
from neurocombat_sklearn import CombatModel


# reproducibility
np.random.seed(0)


# requited paths
df = pd.read_excel('_createdDataframes/df_scores.xlsx', index_col=0) # 1032 participants 
df['Sex'][df['Sex']=='M']=0
df['Sex'][df['Sex']=='F']=1
df['Session'][df['Session'] == "ADNI1"] = 1
df['Session'][df['Session'] == "ADNI2"] = 2


#######################################################
# Class for extracting grey matter per roi from atlas #
#######################################################
class get_cleanedGM_per_roi:
	def __init__(self, df):
		''' allow to extract gm per ROI and clean signal for confounds
		Parameters
		----------
		df : dataframe including a 'smwc1' (grey matter) column per subject (path to nifti cortex file) +
		a column per confound.

		'''
		self.df = df.copy() # dataframe with group, mri path, age and TIV
		self.length = len(self.df.index)

	def fit_atlas(self, strategy='sum', show=False):
		''' masking of the nifti of the participant cortex to extract ROI from atlas.

		Parameters
		----------
		strategy : define how the quantity of grey matter per voxel is summarized for each ROI
		 Must be one of: sum, mean, median, minimum, maximum, variance, standard_deviation
		 Default='sum'.

		show : True or False
		 plot the atlas onto the grey matter nifti file of the first subject to check the fit
		 Default=False


		Return
		----------
		df with grey matter quantity per ROI as self.df_gm

		'''
		 # get atlas and labels
		atlas = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',  symmetric_split=True)
		self.labels = atlas.labels[1:] # does not include the index 0 because it is the outside.
		print("Labels:", self.labels[:5])

		# first subject nifti load to get affine and check mask fit
		tmp_nii = nib.load(self.df['smwc1_path'].iloc[0])
		# show the mask fit onto a participant brain
		if show == True:
			plotting.plot_img(tmp_nii).add_overlay(atlas.maps, cmap=plotting.cm.black_blue)
			plt.savefig('_figures/atlas_fit.png')
			plt.show()
		# shared affine for mask and cortex scan
		ratlas_nii = resample_img(
			atlas.maps, target_affine=tmp_nii.affine, interpolation='nearest')
		# the grey matter volume will be stored in the df_gm dataframe
		self.df_gm = pd.DataFrame(index=self.df.index, columns=self.labels)
		# iterate through each subject
		for i_nii, nii_path in enumerate(self.df["smwc1_path"].values):
			print('')
			print('Doing ', i_nii, ' / ', self.df["smwc1_path"].__len__())
			print('Memory busy by dataframe = ', self.df_gm.memory_usage().sum()/8000000000, ' Go')
			print('The CPU usage is: ', psutil.cpu_percent(4))
			if psutil.cpu_percent(4) >= 90:
				print('Working memory saturated')
				stop
			# summarize grey matter quantity into each ROI
			nii = nib.load(nii_path)
			masker = NiftiLabelsMasker(labels_img=ratlas_nii, standardize=False, strategy=strategy)
			# extract each roi grey matter quantity as list (of length the number of ROIs)
			cur_FS = masker.fit_transform(nii)
			self.df_gm.loc[self.df.index[i_nii]] = cur_FS
			# check memory
			self.df_gm.to_pickle('_pickles/gm_atlas_harvard_oxford')
			# remove duplicates
			assert (self.df_gm.index.duplicated() == True).sum() == 0

	def compute_tiv(self):
		''' compute TIV and store it in the df

		Return
		----------
		self.df with 3 new columns (TIV_gm, TIV_wm, TIV_LCR, and TIV (total))

		'''
		# create new columns to store the tiv
		self.df['TIV'] = '*'
		self.df['TIV_gm'] = '*'
		self.df['TIV_wm'] = '*'
		self.df['TIV_lcr'] = '*'

		# store the voxel size for future checking
		dim_expected = nib.load(df['smwc1_path'].iloc[0]).header.values()[15]

		# iterate across subject to extract, compute and store the TIV
		for ind, row in enumerate(self.df.index):
			# load nifti images
			smwc1 = nib.load(self.df['smwc1_path'].loc[row])
			smwc2 = nib.load(self.df['smwc2_path'].loc[row])
			smwc3 = nib.load(self.df['smwc3_path'].loc[row])
			# check voxel dimension 
			assert smwc1.header.values()[15].sum() == dim_expected.sum()
			assert smwc2.header.values()[15].sum() == dim_expected.sum()
			assert smwc3.header.values()[15].sum() == dim_expected.sum()
			# compute TIV 
			tiv1 = smwc1.get_data().sum()/1000 # grey matter
			tiv2 = smwc2.get_data().sum()/1000 # white matter
			tiv3 = smwc3.get_data().sum()/1000 # LCR
			TIV = tiv1 + tiv2 + tiv3 # total
			assert TIV != 0, "Problem with participant {}".format(row) 
			assert tiv1 != 0, "Problem with participant {}".format(row)
			assert tiv2 != 0, "Problem with participant {}".format(row)
			assert tiv3 != 0, "Problem with participant {}".format(row)
			# online checking
			print("Sub={} ({}/{}), TIV={}, TIV_gm={}, TIV_wm={}, TIV_lcr={}".format(row, ind, self.length, int(TIV), int(tiv1), int(tiv2), int(tiv3)))
			# store TIV in df
			self.df['TIV_gm'].loc[row] = tiv1
			self.df['TIV_wm'].loc[row] = tiv2
			self.df['TIV_lcr'].loc[row] = tiv3
			self.df['TIV'].loc[row] = TIV
		assert '*' not in df.values, "A TIV value is missing"
		self.df.to_pickle('_pickles/tiv')
 

	def clean_signal(self, confounds, session):
		''' clean signal using nilearn.signal.clean function.
		Regressed out variance that could be explained by the factors confounds.
		Data are standardized

		Parameters
		----------
		labels : roi labels to be cleaned

		confounds : the variance from these element will be removed from the signal


		Return
		----------
		df with grey matter per ROI cleaned from variance explained by confound and standardized as self.df_gm_cleaned

		'''
		# extract confound
		confounds = self.df[confounds].values.astype('float64')
		# clean signal from confound explained variance
		FS_cleaned = clean(self.df_gm.values, confounds=confounds, detrend=False)

		# Apply combat for cleaning ADNI session differences
		model = CombatModel()
		FS_cleaned_combat = model.fit_transform(FS_cleaned, session.reshape(-1, 1))

		# store into a dataframe and save
		self.df_gm_cleaned = pd.DataFrame(columns=self.df_gm.columns, index=self.df_gm.index, data=FS_cleaned_combat)
		self.df_gm_cleaned.to_pickle('_pickles/gm_cleaned_atlas_harvard_oxford')





#######################################################
# extract gm per roi and clean signal for age and tiv #
#######################################################

# fit atlas
extract_gm_per_roi = get_cleanedGM_per_roi(df) # load class
extract_gm_per_roi.compute_tiv() # 15 minutes
extract_gm_per_roi.fit_atlas(show=True) # 3 hours
confounds = ['Age', 'TIV', 'Sex']
extract_gm_per_roi.clean_signal(confounds, df['Session'].values) # 30 seconds



######
# short way : avoid reextracting gray matter and TIV which was previously ran
######
data = np.load("/home/jlefortb/ADNI_project/_pickles/gm_atlas_harvard_oxford", allow_pickle=True)
tiv = np.load('/home/jlefortb/ADNI_project/_pickles/tiv', allow_pickle=True)['TIV'].values
confounds = df[['Age', 'Sex']]
confounds['Tiv'] = tiv
# clean signal from confound explained variance
FS_cleaned = clean(data.values, confounds=confounds.values, detrend=False)

# Apply combat for cleaning ADNI session differences
model = CombatModel()
FS_cleaned_combat = model.fit_transform(FS_cleaned, df['Session'].values.reshape(-1, 1))

# store into a dataframe and save
df_gm_cleaned = pd.DataFrame(columns=data.columns, index=data.index, data=FS_cleaned_combat)
df_gm_cleaned.to_pickle('_pickles/gm_cleaned_atlas_harvard_oxford_combat')