"""
Extract scores from neuropsychological tests from each participant.
Take first non-nan score and remove duplicates
Save ouputs as dataframes in _createdDataframes/
2022
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
duration : 5 seconds
"""

import pandas as pd
import numpy as np
import time

# reproducibility
np.random.seed(0)

startTime = time.time()
# load adni participants with mri paths
df = pd.read_excel('_createdDataframe/mri_paths.xlsx', index_col=0) # 1103 participants 

# create a column with RID (4 last characters of ID)
df['RID'] = [ID[-4:] for ID in df.index]
# remove 0 that are before the rid, e.g., 0051 becomes 51
df['RID'] = df['RID'].str.lstrip("0")
# set ID as column
df['ID'] =  df.index
# set RID as index
df = df.set_index('RID')


################################
# extract neuropsy test scores #
################################

# *** load adas ***
print('*** loading & merging adas')
df_adas_adni2 = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/ADAS_ADNIGO23.csv")
df_adas_adni1 = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/ADAS_ADNI1.csv")
df_adas_all = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/ADASSCORES.csv")
df_adas = pd.concat([df_adas_adni1, df_adas_adni2, df_adas_all], sort=False)
# set RID as index
df_adas = df_adas.astype({'RID': str})
df_adas = df_adas.set_index('RID')

df_adas_Q11 = df_adas[["Q1SCORE", "Q2SCORE", "Q3SCORE", "Q4SCORE", "Q5SCORE",
                        "Q6SCORE", "Q7SCORE", "Q8SCORE", "Q9SCORE", "Q10SCORE",
                        "Q11SCORE"]]
df_adas_11 = df_adas[["Q1","Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8",
                        "Q9", "Q10","Q11"]]

df_adas_11.dropna(how='any', inplace=True)
df_adas_11 = df_adas_11[df_adas_11>=0]
df_adas_11.dropna(how='any', inplace=True)
# check for weird numbers
for i in np.unique(df_adas_11.values):
    assert 0 <= i <= 12

df_adas_Q11.dropna(how='any', inplace=True)
# check for weird numbers
for i in np.unique(df_adas_Q11.values):
    assert 0 <= i <= 12


df_adas_11 = df_adas_11.groupby('RID').first()
df_adas_Q11 = df_adas_Q11.groupby('RID').first()
df_adas_Q11.rename(columns = {'Q1SCORE':'Q1', 'Q2SCORE':'Q2',
                        'Q3SCORE':'Q3','Q4SCORE':'Q4',
                        'Q5SCORE':'Q5','Q6SCORE':'Q6',
                        'Q7SCORE':'Q7','Q8SCORE':'Q8',
                        'Q9SCORE':'Q9','Q10SCORE':'Q10',
                        'Q11SCORE':'Q11'}, inplace = True)
df_adas_ = pd.concat([df_adas_11, df_adas_Q11], sort=False)
df_adas_['ADAS11_total'] = df_adas_.sum(axis=1, skipna = True)
# keep the first score per rid
df_adas_11 = df_adas_.groupby('RID').first()
# assert no weird values
for i in np.unique(df_adas_11['ADAS11_total'].values):
    assert 0 <= i <= 12*11 # 12*11 == score max
# Merging with df including the mri paths
df = df.join(df_adas_11, how='inner')

# *** load neurobat file *** 
df_neurobat = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/NEUROBAT.csv")
# set RID as index
df_neurobat = df_neurobat.astype({'RID': str})
df_neurobat = df_neurobat.set_index('RID')

# *** load logical memory scale from neurobat file *** 
print('*** loading & merging logical memory scale')
df_neurobat.rename(columns = {'LDELTOTAL':'LogicalMemoryScaleII'}, inplace = True)
df_LogicalMemoryScaleII = df_neurobat['LogicalMemoryScaleII']
# drop nan
df_LogicalMemoryScaleII.dropna(inplace=True)
# keep the first score per rid
df_LogicalMemoryScaleII = df_LogicalMemoryScaleII.groupby('RID').first()

# check for weird numbers
for i in np.unique(df_LogicalMemoryScaleII.values):
    assert 0 <= i <= 25
# Merging with df including the mri paths
df = df.join(df_LogicalMemoryScaleII, how='inner')

# *** load RAVLT from neurobat file *** 
"""
Briefly, the RAVLT consists of presenting a list of 15 words across five consecutive trials. 
The list is read aloud to the participant, and then the participant is immediately asked to 
recall as many as words as he/she remembers. This procedure is repeated for 5 consecutive trials (Trials 1 to 5). 
After that, a new list (List B) of 15 new words is read to the participant, who then is immediately 
asked to recall the words. After the List B trial, the examiner asks participant to recall the words 
from the first list (Trial 6). After 30-minutes of interpolated testing (timed from the completion of List B recall), 
the participant is again asked to recall the words from the first list (delayed recall).
Different summary scores are derived from raw RAVLT scores. These include RAVLT Immediate 
(the sum of scores from 5 first trials (Trials 1 to 5)), RAVLT Learning (the score of Trial 5 minus the score of Trial 1), 
RAVLT Forgetting (the score of Trial 5 minus score of the delayed recall) and RAVLT Percent Forgetting 
(RAVLT Forgetting divided by the score of Trial 5)
"""
print('*** loading & merging RAVLT')
df_ravlt = df_neurobat[["AVTOT1", "AVTOT5", "AVTOT6", "AVDEL30MIN", "AVDELTOT"]]
df_ravlt.rename(columns = {'AVDELTOT':'ravlt_recognition', 'AVDEL30MIN':'ravlt_30minuteDelay', 'AVTOT6':'ravlt_shortDelay'}, inplace = True)
# drop nan
df_ravlt.dropna(inplace=True)
df_ravlt = df_ravlt[df_ravlt["AVTOT1"] != -1]
df_ravlt = df_ravlt[df_ravlt["AVTOT5"] != -1]
df_ravlt = df_ravlt[df_ravlt["ravlt_shortDelay"] != -1]
# check data make sense
for i in np.unique(df_ravlt["AVTOT1"].values):
    assert 0 <= i <= 15
for i in np.unique(df_ravlt["AVTOT5"].values):
    assert 0 <= i <= 15
for i in np.unique(df_ravlt["ravlt_shortDelay"].values):
    assert 0 <= i <= 15
df_ravlt["RAVLT_learning"] = df_ravlt["AVTOT1"] - df_ravlt["AVTOT5"]
df_ravlt["RAVLT_forgetting"] = df_ravlt["AVTOT5"] - df_ravlt["ravlt_shortDelay"]
# do not compute percentage forgetting cause need to divide by 0 for some values
# check data make sense
for i in np.unique(df_ravlt["RAVLT_learning"].values):
    assert -15 <= i <= 15
# drop score > 15 (cannot be >15) and == -1 (cannot be -1)
df_ravlt = df_ravlt[df_ravlt["ravlt_30minuteDelay"] <= 15]
df_ravlt = df_ravlt[df_ravlt["ravlt_30minuteDelay"] != -1]
# check data make sense
for i in np.unique(df_ravlt["ravlt_30minuteDelay"].values):
    assert 0 <= i <= 15
# check data make sense
for i in np.unique(df_ravlt["RAVLT_forgetting"].values):
    assert -15 <= i <= 15
# keep the first score per rid
df_ravlt = df_ravlt.groupby('RID').first()
# Merging with df including the mri paths
df = df.join(df_ravlt, how='inner')

# *** load Boston Naming test from neurobat file *** 
print('*** loading & merging BNT')
df_neurobat.rename(columns = {"BNTTOTAL":'BostonNamingTest'}, inplace = True)
df_BostonNamingTest = df_neurobat['BostonNamingTest']
# drop nan
df_BostonNamingTest.dropna(inplace=True)
df_BostonNamingTest = df_BostonNamingTest[df_BostonNamingTest != -1]
# check data make sense
for i in np.unique(df_BostonNamingTest.values):
    assert 0 <= i <= 30
# keep the first score per rid
df_BostonNamingTest = df_BostonNamingTest.groupby('RID').first()
# Merging with df including the mri paths
df = df.join(df_BostonNamingTest, how='inner')

# *** load Trail Making Test from neurobat file *** 
"""
Results for both TMT A and B are reported as the number of seconds required to complete the task;
therefore, higher scores reveal greater impairment. 
         Average     Deficient    Rule of Thumb 
Trail A 29 seconds > 78 seconds Most in 90 seconds 
Trail B 75 seconds > 273 seconds Most in 3 minutes 
"""
print('*** loading & merging TMT')
df_tmt =  df_neurobat[["TRAASCOR","TRABSCOR"]]
# drop nan
df_tmt.dropna(inplace=True)
df_tmt["TRAASCOR"][df_tmt["TRAASCOR"] >=150] = 150
df_tmt["TRABSCOR"][df_tmt["TRABSCOR"] >=300] = 300
# drop <10sec values (impossible) for A, and <50sec for b (impossible)
df_tmt = df_tmt[df_tmt["TRAASCOR"] > 10]
df_tmt = df_tmt[df_tmt["TRABSCOR"] > 50]
# check data make sense
for i in np.unique(df_tmt.values):
    assert 10 <= i <= 300
df_tmt["TMT_BminusA"] = df_tmt["TRABSCOR"] - df_tmt["TRAASCOR"]
# drop negative (BminusA) value, don't make sense
df_tmt = df_tmt[df_tmt["TMT_BminusA"] > 10]
# keep the first score per rid
df_tmt = df_tmt.groupby('RID').first()
# Merging with df including the mri paths
df = df.join(df_tmt, how='inner')

# *** load CAT ANIMAL from neurobat file *** 
"""
A score of under 17 indicates concern, although some practitioners use 14 as a cutoff. 
"""
print('*** loading & merging CAT Animals')
df_neurobat.rename(columns = {"CATANIMSC":'VFT_animals'}, inplace = True)
df_VFT_animals = df_neurobat['VFT_animals']
# keep only vamue > 0 (doesnt make sense otherwise)
df_VFT_animals = df_VFT_animals[df_VFT_animals >= 0]
# > 45 is already very good 
df_VFT_animals[df_VFT_animals >= 45] = 45
# drop nan
df_VFT_animals.dropna(inplace=True)
# check data make sense
for i in np.unique(df_VFT_animals.values):
    assert 0 <= i <= 45
# keep the first score per rid
df_VFT_animals = df_VFT_animals.groupby('RID').first()
# Merging with df including the mri paths
df = df.join(df_VFT_animals, how='inner')
# Only nan for CAT VEGETAL 

# *** load ANAR from neurobat file *** 
print('*** loading & merging ANART')
df_neurobat.rename(columns = {"ANARTERR":'ANART'}, inplace = True)
df_ANART = df_neurobat['ANART']
# drop nan
df_ANART.dropna(inplace=True)
# check data make sense
for i in np.unique(df_ANART.values):
    assert 0 <= i <= 50
# keep the first score per rid
df_ANART = df_ANART.groupby('RID').first()
# Merging with df including the mri paths
df = df.join(df_ANART, how='inner')

# only NAN = ['DSPANBLTH', 'DSPANFLTH', 'CATVEGESC', 'DIGITSCOR']

# *** load and add FAQ to df with mri path *** ***
"""
Sum scores (range 0-30). Cutpoint of 9 (dependent in 3 or more activities) is recommended 
to indicate impaired function and possible cognitive impairment.
"""
print('*** loading & merging FAQ')
df_faq = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/FAQ.csv")
df_faq = df_faq.astype({'RID': str})
df_faq = df_faq.set_index("RID")
df_faq.rename(columns = {"FAQTOTAL":'FAQ_tot'}, inplace = True)
df_faq = df_faq['FAQ_tot']
# drop nan
df_faq.dropna(inplace=True)
# get rid of -1 scores
df_faq = df_faq[df_faq >=0]
# check data make sense
for i in np.unique(df_faq.values):
    assert 0 <= i <= 30
# keep the first score per rid
df_faq = df_faq.groupby('RID').first()
# Merging with df including the mri paths
df = df.join(df_faq, how='inner')

# # *** load NPI-Q and add it to df with mri path *** ***
### LOSING 25 subjetcs // NOT INCLUDED IN FINAL ANALYSIS
# print('*** loading & merging NPI-Q')
# df_npiq = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/NPIQ.csv")
# df_npiq = df_npiq.astype({'RID': str})
# df_npiq = df_npiq.set_index("RID")
# df_npiq.rename(columns = {'NPISCORE':'NPI_tot'}, inplace = True)
# df_npiq = df_npiq['NPI_tot']
# # drop nan
# df_npiq.dropna(inplace=True)
# # check data make sense
# for i in np.unique(df_npiq.values):
#     assert 0 <= i <= 30
# # keep the first score per rid
# df_npiq = df_npiq.groupby('RID').first()
# # Merging with df including the mri paths
# df = df.join(df_npiq, how='inner')

# *** load GDS and add it to df with mri path *** ***
print('*** loading & merging GDS')
df_gds = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/GDSCALE.csv")
df_gds = df_gds.astype({'RID': str})
df_gds = df_gds.set_index("RID")
df_gds.rename(columns = {'GDTOTAL':'GDS_tot'}, inplace = True)
df_gds = df_gds['GDS_tot']
# drop nan
df_gds.dropna(inplace=True)
# get rid of negative scores (not possible)
df_gds = df_gds[df_gds >=0]
# check data make sense
for i in np.unique(df_gds.values):
    assert 0 <= i <= 15
# keep the first score per rid
df_gds = df_gds.groupby('RID').first()
# Merging with df including the mri paths
df = df.join(df_gds, how='inner')

# 1032 participants

executionTime = (time.time() - startTime)
print('Execution time to extract a score per test in seconds: ' + str(executionTime))
# script ran in 5 seconds


####################################
# get which participants turned AD #
####################################

df['TurnedAD'] = 0
df_diag_change = pd.read_csv('filesFromADNI/ADNI_cognitiveTests/ADNIMERGE.csv')
df_diag_change = df_diag_change.astype({'RID': str})
df_diag_change = df_diag_change.set_index('RID')
df_diag_change = df_diag_change['DX']


turned_AD = 0
for index in df.index:
    assert len(df_diag_change.loc[str(index)]) > 0
    if 'Dementia' in list(df_diag_change.loc[str(index)]):
        print(list(df_diag_change.loc[str(index)]), 'DIAG OFFICIAL = ', df.Group.loc[index])
        turned_AD += 1
        df['TurnedAD'].loc[index] = 1


##################
### load Apoe ####
##################
df['APOE4'] = 0
df_apoe = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/APOERES.csv")
df_apoe = df_apoe.astype({'RID': str})
df_apoe = df_apoe.set_index("RID")
df_apoe = df_apoe[['APGEN1', 'APGEN2']]

for index in df.index:
    try:
        df_apoe[['APGEN1', 'APGEN2']].loc[str(index)]
        genes = df_apoe[['APGEN1', 'APGEN2']].loc[str(index)].values
        assert len(df_apoe[['APGEN1', 'APGEN2']].loc[str(index)]) > 0
    except:
        print('no data for ', index)
    status = 0
    if genes[0] == genes[1] == 4:
        status = 2
    elif genes[0] == 4:
        status = 1
    elif genes[1] == 4:
        status = 1
    else:
        status = 0
    df['APOE4'].loc[index] = status


############################################
## RENAME VARIABLE NAME FOR PRETTIER ONES ##
############################################

df.rename(columns = {'Q3':'ADAS Q3 constr. praxis', 
    'Q5':'ADAS Q5 naming',  
    'Q6':'ADAS Q6 ideat. praxis',  
    'Q8':'ADAS Q8 word recognition', 
    'Q10':'ADAS Q10 compr. spoken lang.', 
    'Q11':'ADAS Q11 word finding difficuty',
    'ravlt_shortDelay': 'RAVLT short delay',
    'ravlt_30minuteDelay':'RAVLT long delay', 
    'ravlt_recognition':'RAVLT recognition', 
    'RAVLT_learning':'RAVLT learning', 
    'RAVLT_forgetting':'RAVLT forgetting',
    'BostonNamingTest':'Boston naming test', 
    'TRAASCOR':'TMT score A', 
    'TMT_BminusA':'TMT score B-A',  
    'VFT_animals':'VFT animals',
    'LogicalMemoryScaleII': 'Logical memory scale II'}, inplace = True)

df.to_excel('_createdDataframe/df_scores.xlsx') # 1032 subjects

