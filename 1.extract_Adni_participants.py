"""
Extract MRI scan path + age, group, sex, and RID per participants from the ADNI1 + remove duplicates
Save ouputs as dataframe in _createdDataframe//
2022
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io

duration : 10 seconds
"""

import pandas as pd
import numpy as np
import glob

np.random.seed(0)

print("CREATE DATAFRAME WITH MRI SCAN PATH FOR SUBJECT OF ADNI 1")


#######################################
# CREATE DATAFRAME WITH MRI SCAN PATH #
#######################################

# store file path with subject id for each smwc (1: grey matter, 2: white matter, 3: LCR)
smwc1_dic, smwc2_dic, smwc3_dic = {}, {}, {}
smwc_file_path = glob.glob('C:\\Users\\lefortb211\\Downloads\\ADNI1_smwc_files\\*\\*.nii')
for ind, path in enumerate(smwc_file_path):
	sub_id = path.split('\\')[-1][10:20]
	smwc_numb = path.split('\\')[-1][4:5]
	if smwc_numb == '1':
		smwc1_dic[sub_id] = path
	elif smwc_numb == '2':
		smwc2_dic[sub_id] = path
	elif smwc_numb == '3':
		smwc3_dic[sub_id] = path
	else:
		print('Problem with regular expression')
		stop
assert len(smwc1_dic) == len(smwc2_dic) == len(smwc3_dic)

# create a df with MRI matter type as column and participant ID as index
df = pd.DataFrame(columns=['smwc1_path', 'smwc2_path', 'smwc3_path'])
for key in smwc1_dic.keys():
	df.loc[key] = [smwc1_dic[key], smwc2_dic[key], smwc3_dic[key]] 
	
	# assert path is a nifti file
	assert df['smwc1_path'].loc[key][-3:] == 'nii', "Df wrongly filled for {}".format(path.split('\\')[-1])
	assert df['smwc2_path'].loc[key][-3:] == 'nii', "Df wrongly filled for {}".format(path.split('\\')[-1])
	assert df['smwc3_path'].loc[key][-3:] == 'nii', "Df wrongly filled for {}".format(path.split('\\')[-1])
	# assert smwc1, 2 and 3 are form the same participant
	assert df['smwc1_path'].loc[key][-20:] == df['smwc2_path'].loc[key][-20:] == df['smwc3_path'].loc[key][-20:], "Seems that the smwc1 and 2 are not from the same participant {}".format(path.split('\\')[-1])

############################################
# Add Age, Sex, and Group FOR EACH SUBJECT #
############################################

# extract CN participant RID
df_info1 = pd.read_csv('filesFromADNI/info1.csv')
df_info1 = df_info1.set_index('Subject').astype('str')
df_info1 = df_info1[['Group', 'Sex', 'Age']]
df_info2 = pd.read_csv('filesFromADNI/info2.csv')
df_info2 = df_info2.set_index('Subject').astype('str')
df_info2 = df_info2[['Group', 'Sex', 'Age']]

# merge df_info1
df_merged = df.join(df_info1, how='outer')
# drop subject with no mri scan
df_merged = df_merged.dropna(axis=0, how='any', subset=['smwc1_path','smwc2_path','smwc3_path'])
# remove duplicate, keep the last one
df_merged = df_merged.drop_duplicates(subset='smwc1_path', keep='last')

# find missing information from df_info2
for index_with_missing_info in df_merged[df_merged['Age'].isnull()].index:
	df_merged['Group'].loc[index_with_missing_info] = df_info2['Group'].loc[index_with_missing_info]
	df_merged['Age'].loc[index_with_missing_info] = df_info2['Age'].loc[index_with_missing_info]
	df_merged['Sex'].loc[index_with_missing_info] = df_info2['Sex'].loc[index_with_missing_info]

for col in df_merged.columns:
	nb_nan = df_merged[col].isnull().values.sum()
	print(nb_nan, " Nan in ", col)

df_merged['Session'] = "ADNI1"

# save df as adni 1 df
df_adni1 = df_merged



print("CREATE DATAFRAME WITH MRI SCAN PATH FOR SUBJECT OF ADNI 2")

#######################################
# CREATE DATAFRAME WITH MRI SCAN PATH #
#######################################

# extract all files from folders (each population)
files_eMCI = glob.glob('C:\\Users\\lefortb211\\Downloads\\ADNI_eMCI_smwc\\*')
files_lMCI = glob.glob('C:\\Users\\lefortb211\\Downloads\\ADNI_lMCI_smwc\\*')
files_CN = glob.glob('C:\\Users\\lefortb211\\Downloads\\ADNI_CN_smwc\\*')
files_SMC = glob.glob('C:\\Users\\lefortb211\\Downloads\\ADNI_SMC_smwc\\*')
smwc_files = files_eMCI + files_lMCI + files_CN + files_SMC

# store file path with subject id for each smwc (1: grey matter, 2: white matter, 3: LCR)
smwc1_dic, smwc2_dic, smwc3_dic = {}, {}, {}
for ind, path in enumerate(smwc_files):
    sub_id = path.split('\\')[-1][10:20]
    smwc_numb = path.split('\\')[-1][4:5]
    if smwc_numb == '1':
        smwc1_dic[sub_id] = path
    elif smwc_numb == '2':
        smwc2_dic[sub_id] = path
    elif smwc_numb == '3':
        smwc3_dic[sub_id] = path
    else:
        print('Problem with regular expression')
        stop
assert len(smwc1_dic) == len(smwc2_dic) == len(smwc3_dic)

# create a df with mri_scan path as column and participant ID as index
df = pd.DataFrame(columns=['smwc1_path', 'smwc2_path', 'smwc3_path'])
for key in smwc1_dic.keys():
    df.loc[key] = [smwc1_dic[key], smwc2_dic[key], smwc3_dic[key]] 
    
    # assert path is a nifti file
    assert df['smwc1_path'].loc[key][-3:] == 'nii', "Df wrongly filled for {}".format(path.split('\\')[-1])
    assert df['smwc2_path'].loc[key][-3:] == 'nii', "Df wrongly filled for {}".format(path.split('\\')[-1])
    assert df['smwc3_path'].loc[key][-3:] == 'nii', "Df wrongly filled for {}".format(path.split('\\')[-1])
    # assert smwc1, 2 and 3 are form the same participant
    assert df['smwc1_path'].loc[key][-20:] == df['smwc2_path'].loc[key][-20:] == df['smwc3_path'].loc[key][-20:], "Seems that the smwc1 and 2 are not from the same participant {}".format(path.split('\\')[-1])


############################################
# Add Age, Sex, and Group FOR EACH SUBJECT #
############################################

# add age, group and sex information
df_demog_CN = pd.read_csv("filesFromADNI/demographic_CN.csv")[['Subject', 'Group', 'Sex', 'Age']].set_index('Subject') # subject, group, sex, age
# 147 sub
df_demog_SMC = pd.read_csv("filesFromADNI/demographic_SMC.csv")[['Subject', 'Group', 'Sex', 'Age']].set_index('Subject') # subject, group, sex, age
# 76 sub
df_demog_eMCI = pd.read_csv("filesFromADNI/demographic_eMCI.csv")[['Subject', 'Group', 'Sex', 'Age']].set_index('Subject') # subject, group, sex, age
# 156 sub
df_demog_lMCI = pd.read_csv("filesFromADNI/demographic_lMCI.csv")[['Subject', 'Group', 'Sex', 'Age']].set_index('Subject') # subject, group, sex, age
# 141 sub

# concatenate all dataframe
df_demog = pd.concat([df_demog_CN, df_demog_SMC,df_demog_eMCI, df_demog_lMCI])
# merge with mri scan df
df_merged = df.join(df_demog, how='outer')
# drop subject with no mri scan
df_merged = df_merged.dropna(axis=0, how='any', subset=['smwc1_path','smwc2_path','smwc3_path'])
# remove duplicate, keep the last one
df_merged = df_merged.drop_duplicates(subset='smwc1_path', keep='last')

for col in df_merged.columns:
    nb_nan = df_merged[col].isnull().values.sum()
    print(nb_nan, " Nan in ", col)

df_merged['Session'] = "ADNI2"

# save df as adni 1 df
df_adni2 = df_merged

df = pd.concat([df_adni1, df_adni2], sort=False)
df.to_excel("_createdDataframe/mri_paths.xlsx")