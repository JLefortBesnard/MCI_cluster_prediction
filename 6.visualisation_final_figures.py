"""
Script to make every figures from the ADNI analysis
2022
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from nilearn import datasets as ds

# visualisation
from matplotlib import pylab as plt
import seaborn as sns


# Plotting function for the x axis to be centered
def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
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







#############################################################
########################## FIGURE 1 #########################
#############################################################

#########################
# clustering figures
#########################

# reproducibility
np.random.seed(0)

# order for category (visuospatial, memory, language, executive)
col_for_clustering = ['ADAS Q3 constr. praxis', 'ADAS Q6 ideat. praxis',  # visuospatial
        'ADAS Q8 word recognition', 'Logical memory scale II','RAVLT short delay', 'RAVLT long delay', 'RAVLT recognition', 'RAVLT learning', 'RAVLT forgetting',# memory
        'VFT animals', 'Boston naming test', 'ADAS Q5 naming', # language
        'TMT score A', 'TMT score B-A'# excecutive
        ]


# get the participant raw scores
df_scores = pd.read_excel('_createdDataframe/df_scores.xlsx', index_col=0) # 1033 subjects
variance = np.var(df_scores[col_for_clustering], axis=0) # 1033 subjects

# get the participant standardized scores
df_scores_standardized = pd.read_excel('_createdDataframe/df_scores_std.xlsx')



#################################################
## Plot clustering analysis results as barplot ##
#################################################

# get MCI cluster and scores
sns.set(style="white", context="talk")
X_colnames = col_for_clustering
# Set up the matplotlib figure


# reformat df to allow for displaying standard deviation on the plot
# first extract subgroup scores
# then reshape to get columns of scores
# then add 1.5 to avoid negative scores (standardized b/w -1 and 1)

df_low = pd.melt(df_scores_standardized[df_scores_standardized['cluster'] == 'Low'][col_for_clustering])
df_low['value'] = df_low['value'] + 1.5

df_high = pd.melt(df_scores_standardized[df_scores_standardized['cluster'] == 'High'][col_for_clustering])
df_high['value'] = df_high['value'] + 1.5

df_middle = pd.melt(df_scores_standardized[df_scores_standardized['cluster'] == 'Middle'][col_for_clustering])
df_middle['value'] = df_middle['value'] + 1.5

assert df_low.mean(axis=0).mean() < df_middle.mean(axis=0).mean() < df_high.mean(axis=0).mean()

# get scores of group CN
df_cn = pd.melt(df_scores_standardized[df_scores_standardized['cluster'] == 'CN'][col_for_clustering])
df_cn['value'] = df_cn['value'] + 1.5


# Set up the matplotlib figure
f, axarr = plt.subplots(4, 1, figsize=(16, 12), sharex=True) 
# Take care of the colors
my_palette = ['#e74c3c'] * 2 + ['#3498db'] * 7 + ['purple'] * 3 + ['#00BB00'] * 2
sns.set_palette(my_palette)
for i_cl, df in enumerate([df_low, df_middle, df_high, df_cn]):
    df = df.rename(columns={'value':' '})
    n_subs = len(df.index)/len(col_for_clustering)
    ax = sns.barplot('variable', ' ', data=df, palette=my_palette, ci='sd', ax=axarr[i_cl])
    if i_cl == 0:
        ax.xaxis.set_ticks_position('top')
        rotateTickLabels(ax, 45, 'x')
        ax.xaxis.set_ticklabels(X_colnames, fontsize=24)
        cluster_name = "Mixed MCI"
    if i_cl == 1:
        cluster_name = "Amnestic MCI"
        plt.tick_params(axis='x', labelbottom=False)

    if i_cl == 2:
        cluster_name = "Cluster-derived normal"
        plt.tick_params(axis='x', labelbottom=False)

    if i_cl == 3:
        cluster_name = "Controls"
        plt.tick_params(axis='x', labelbottom=False)


    ax.set_ylim([0, 3])
    ax.set_xlabel("{} : {} patients".format(cluster_name, int(n_subs)), fontsize=24)
plt.tight_layout(h_pad=3)
plt.savefig('_figures/final_figs/clustering_barplot_allsub.png')
plt.close('all')

# gainsboro, darkgrey, dimgray

#################################################
## Plot clustering analysis results as polar ##
#################################################
from math import pi

# number of variable
categories = col_for_clustering.copy()
N = len(categories)
 
# prepare data to fit
# we need to repeat the first value to close the circular graph:
values0 = df_scores_standardized[df_scores_standardized['cluster'] == 'Low'][col_for_clustering].mean(axis=0) + 1
values0 = values0.values.tolist()
values0 += values0[:1]
values1 = df_scores_standardized[df_scores_standardized['cluster'] == 'Middle'][col_for_clustering].mean(axis=0) + 1
values1 = values1.values.tolist()
values1 += values1[:1]
values2 = df_scores_standardized[df_scores_standardized['cluster'] == 'High'][col_for_clustering].mean(axis=0) + 1
values2 = values2.values.tolist()
values2 += values2[:1]
values_cn = df_scores_standardized[df_scores_standardized['Group'] == 'CN'][col_for_clustering].mean(axis=0) + 1
values_cn = values_cn.values.tolist()
values_cn += values_cn[:1]


 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels
angles_lab = angles[:-1]
categories[3] = 'Log. mem. sca. II'
plt.xticks(angles_lab, [], color='black', size=14)
# plt.xticks(angles_lab, categories, color='black', size=14)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.5,1,1.5], ["0.5","1","1.5"], color="grey", size=12)
plt.ylim(0,2)

# Plot data
ax.plot(angles, values0, linewidth=3, linestyle='solid', color='dimgray', label='Mixed MCI')
ax.plot(angles, values1, linewidth=3, linestyle='solid', color='darkgrey', label='Amnestic MCI')
ax.plot(angles, values2, linewidth=3, linestyle='solid', color='gainsboro', label='Cluster-derived normal')
ax.plot(angles, values_cn, linewidth=2, linestyle='dotted', color='black', label='Controls')#, alpha=0.9)


# lightcoral, lightgreen,orange

# Fill area
# ax.fill(angles, values, 'b', alpha=0.1)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='lower left', borderaxespad=0, fontsize=12)
# plt.legend(loc='lower left', borderaxespad=0, fontsize=12)
# Save the graph
plt.tight_layout()
plt.savefig('_figures/final_figs/polarplot.png')
plt.close('all')










#############################################################
########################## FIGURE 2 #########################
#############################################################


#################
## Plot as PCA ##
#################

df_scores_standardized =  df_scores_standardized[ df_scores_standardized["cluster"] != 'SMC']

pca = PCA(n_components=2)
acp_coef = pca.fit(df_scores_standardized[col_for_clustering])
explained_variance =  acp_coef.explained_variance_ratio_
coefs = acp_coef.components_
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

pd.DataFrame(columns=[['composante 1', 'composante 2']], index=col_for_clustering, data=coefs.T)

acp = pca.transform(df_scores_standardized[col_for_clustering].values)
df_scores_standardized[['acp1', 'acp2']] = acp

acp_high = df_scores_standardized[['acp1', 'acp2']].loc[df_scores_standardized[df_scores_standardized.cluster == 'High'].index]
acp_middle = df_scores_standardized[['acp1', 'acp2']].loc[df_scores_standardized[df_scores_standardized.cluster == 'Middle'].index]
acp_low = df_scores_standardized[['acp1', 'acp2']].loc[df_scores_standardized[df_scores_standardized.cluster == 'Low'].index]
acp_cn = df_scores_standardized[['acp1', 'acp2']].loc[df_scores_standardized[df_scores_standardized.cluster == 'CN'].index]

acp_ad = df_scores_standardized[df_scores_standardized.TurnedAD == 1][['acp1', 'acp2']]

f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(acp_cn['acp1'].values, acp_cn['acp2'].values, c='orange', alpha=0.3, s=70, label="Controls") # 'antiquewhite'
plt.scatter(acp_low['acp1'].values, acp_low['acp2'].values, c='dimgray', s=100, label="Mixed MCI")
plt.scatter(acp_middle['acp1'].values, acp_middle['acp2'].values, c='darkgrey', s=100, label="Amnestic MCI")
plt.scatter(acp_high['acp1'].values, acp_high['acp2'].values, c='gainsboro', s=100, label="Cluster-derived normal") #
plt.scatter(acp_ad['acp1'].values, acp_ad['acp2'].values, s=5, facecolors='none', alpha=0.9, edgecolors='lime', label="Turned AD")

# plt.text(-2.5, 4.5, "Composant 1: {}".format(np.round(explained_variance[0]*100, 2)),fontsize=20)
# plt.text(-2.5, 4, "Composant 2: {}".format(np.round(explained_variance[1]*100, 2)),fontsize=20)

plt.legend()
plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.legend(loc='upper left', fontsize=22)
plt.tight_layout()
# plt.savefig('_figures/ACP_turnedAD_apoe4.png')
plt.savefig('_figures/final_figs/ACP_turnedAD.png')
plt.close('all')




#########################################
## PIE CHART PLOTTING TURNING ALZHEIMER #
#########################################
df = df_scores_standardized.copy()
df_CN = df_scores_standardized[df_scores_standardized['cluster'] == 'CN']
df_High = df_scores_standardized[df_scores_standardized['cluster'] == 'High']
df_Low = df_scores_standardized[df_scores_standardized['cluster'] == 'Low']
df_Middle = df_scores_standardized[df_scores_standardized['cluster'] == 'Middle']

# adapt cluster names:
df['cluster'][df['cluster'] == 'Low'] = 'Mixed MCI'
df['cluster'][df['cluster'] == 'Middle'] = 'Amnestic MCI'
df['cluster'][df['cluster'] == 'High'] = 'Cluster-derived normal'
df['cluster'][df['cluster'] == 'CN'] = 'Controls'
dfs = [df_CN, df_High, df_Low, df_Middle]
AD_per_gr = {}
rapport = {}
tot_per_gr = {}
for df_ in dfs:
    name = np.unique(df_.cluster)[0]
    nb_AD = df_.TurnedAD.sum()
    AD_per_gr[name] = nb_AD
    rapport[nb_AD] = df_.shape[0]

AD_per_gr["Controls"] = AD_per_gr.pop("CN")
AD_per_gr["Cluster-derived normal"] = AD_per_gr.pop("High")
AD_per_gr["Amnestic MCI"] = AD_per_gr.pop("Middle")
AD_per_gr["Mixed MC"] = AD_per_gr.pop("Low")

def func(pct, x):
    print(pct)
    if int(pct) == 9:
        absolute = 28
    elif int(pct) == 0:
        absolute = 1
    elif int(pct) == 4:
        absolute = 14
    elif int(pct) == 61:
        absolute = 190
    elif int(pct) == 25:
        absolute = 78

    return "{} / {} part. ".format(absolute, rapport[absolute])

plt.figure(figsize = (8, 8))
x = [v for v in AD_per_gr.values()]
values = [v for v in AD_per_gr.values()]
plt.pie(x, labels = [v for v in AD_per_gr.keys()], 
    colors = ['orange', 'gainsboro', 'dimgray', 'darkgrey'],
    # autopct= lambda pct: func(pct, rapport), 
    pctdistance = 0.7,
    textprops=dict(color="black", fontsize=18),
    wedgeprops = {'linewidth': 0},
    normalize = True)
# plt.legend(loc='lower left', fontsize=14)
# plt.title('AD turned participants per diagnosis group', fontsize=20)
plt.tight_layout()
plt.savefig('_figures/final_figs/TurnedAD_per_group_pie.png')
plt.close('all')



################################################
## double histogram PLOTTING TURNING ALZHEIMER #
################################################

df = df_scores_standardized.copy()
# adapt cluster names:
df['cluster'][df['cluster'] == 'Low'] = 'Mixed MCI'
df['cluster'][df['cluster'] == 'Middle'] = 'Amnestic MCI'
df['cluster'][df['cluster'] == 'High'] = 'Cluster-derived normal'
df['cluster'][df['cluster'] == 'CN'] = 'Controls'

df_hist = df[['TurnedAD', 'cluster']]
df_hist['N'] = 1

df_hist = df_hist.groupby('cluster').sum()
df_hist['cluster'] = df_hist.index



f, ax = plt.subplots(figsize=(10, 5))
# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="N", y="cluster", data=df_hist,
            order=['Controls', 'Cluster-derived normal', 'Amnestic MCI','Mixed MCI'],
            label="Number of subject total", color="b")
sns.set_color_codes("muted")
sns.barplot(x="TurnedAD", y="cluster", data=df_hist,
            order=['Controls', 'Cluster-derived normal', 'Amnestic MCI','Mixed MCI'],
            label="Number of subject developing AD", color="b")


# Add a legend and informative axis label
# ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 340), ylabel="",
       xlabel="Number of subjects")
sns.despine(left=True, bottom=True)
plt.legend(bbox_to_anchor=(1.05, 1), fontsize=14)
plt.tight_layout()
plt.savefig('_figures/final_figs/TurnedAD_per_group.png')
plt.close('all')



#############################
# RISK RATIO TURNING AD  ####
#############################

# get risk ratio per diagnosis
# chi square analysis
from scipy.stats import chisquare

df_chisquare = df_scores_standardized[['TurnedAD', 'cluster']]

nb_turnedAD = len(df_chisquare[df_chisquare['TurnedAD']==1][df_chisquare['cluster'] == "CN"])
tot = len(df_chisquare[df_chisquare['cluster'] == "CN"])
ratio_control = nb_turnedAD/tot


nb_turnedAD = len(df_chisquare[df_chisquare['TurnedAD']==1][df_chisquare['cluster'] == "High"])
tot = len(df_chisquare[df_chisquare['cluster'] == "High"])
ratio_high = nb_turnedAD/tot

nb_turnedAD = len(df_chisquare[df_chisquare['TurnedAD']==1][df_chisquare['cluster'] == 'Low'])
tot = len(df_chisquare[df_chisquare['cluster'] == 'Low'])
ratio_low = nb_turnedAD/tot

nb_turnedAD = len(df_chisquare[df_chisquare['TurnedAD']==1][df_chisquare['cluster'] == 'Middle'])
tot = len(df_chisquare[df_chisquare['cluster'] == 'Middle'])
ratio_middle = nb_turnedAD/tot



RR_high = ratio_high/ratio_control
RR_middle = ratio_middle/ratio_control
RR_low = ratio_low/ratio_control


import matplotlib.pyplot as plt
fig = plt.figure()
diags = ['Cluster-derived normal', 'Amnestic MCI', 'Mixed MCI']
RR = [RR_high,RR_middle,RR_low]
plt.bar(diags,RR, color=['gainsboro', 'darkgrey', 'dimgray'])
plt.xlabel('Diagnosis', fontsize=20)
plt.ylabel('Risk ratio', fontsize=20)
plt.xticks(fontsize=20, rotation=35)
plt.title('Risk ratio of turning AD per group')
plt.axhline(y=1, linestyle='--', color='red')
plt.tight_layout()
plt.savefig('_figures/final_figs/RR_per_group.png')
plt.close('all')













#############################################################
########################## FIGURE 3 #########################
#############################################################



################
## PLOTTING TAU et Amyloid Beta
################
np.random.seed(0)

df_ = pd.read_pickle('_pickles/csf_level_std')

# adapt cluster names:
df_['cluster'][df_['cluster'] == 'Low'] = 'Mixed MCI'
df_['cluster'][df_['cluster'] == 'Middle'] = 'Amnestic MCI'
df_['cluster'][df_['cluster'] == 'High'] = 'Cluster-derived normal'
df_['cluster'][df_['cluster'] == 'CN'] = 'Controls'

######
# TAU/PTAU + ABETA
######

import seaborn as sns
sns.set_theme(style="whitegrid")

df__ = pd.melt(df_[['cluster', 'ABETA', 'PTAU', 'TAU']], id_vars=['cluster'])


g = sns.catplot(
    data=df__, kind="point",
    order=['Controls', 'Cluster-derived normal', 'Amnestic MCI','Mixed MCI'],
    x="cluster", y="value", hue="variable",
    ci=95, palette="dark", alpha=.6, height=6, legend=False
)
g.despine(left=True)
g.set_axis_labels("", "CSF Level (std)", fontsize=20)
plt.legend(title="CSF", loc='center right', fontsize=9)

plt.xticks(fontsize=20, rotation=45)
plt.tight_layout()
plt.savefig('_figures/final_figs/csf_per_group_std.png')
plt.close('all')



############
# VISUALISATION PREDICTION WITH GREY MATTER  ####
############

# violin plot of accuracies

df_visu_gm = pd.read_pickle('_pickles/df_visu_gm')
df_visu_csf = pd.read_pickle('_pickles/df_visu_csf')


sns.set(style="white", context="talk")
df_visu_gm = df_visu_gm[['High_vs_CN', 'Middle_vs_CN', 'Low_vs_CN']]
df_visu_csf = df_visu_csf[['High_vs_CN', 'Middle_vs_CN', 'Low_vs_CN']]
df_visu_gm.rename(columns = {'High_vs_CN':'Cluster-derived normal', 'Middle_vs_CN':'Amnestic MCI', 'Low_vs_CN':'Mixed MCI'}, inplace = True)
df_visu_csf.rename(columns = {'High_vs_CN':'Cluster-derived normal', 'Middle_vs_CN':'Amnestic MCI', 'Low_vs_CN':'Mixed MCI'}, inplace = True)
df_gm = pd.melt(df_visu_gm)
df_csf = pd.melt(df_visu_csf)
df_gm = df_gm.rename(columns={'variable': 'Diagnosis group', 'value': 'Model accuracies gm'})
df_csf = df_csf.rename(columns={'variable': 'Diagnosis group', 'value': 'Model accuracies csf'})
df_mix = pd.concat((df_csf, df_gm))
df_mix['base'] = 0
df_mix['base'][df_mix['Model accuracies csf'].isna()] = 'Grey matter based prediction'
df_mix['base'][df_mix['Model accuracies gm'].isna()] = 'CSF based prediction'
df_mix['Model accuracies'] = 0
df_mix['Model accuracies'][df_mix['base'] == 'Grey matter based prediction'] = df_mix['Model accuracies gm'][df_mix['Model accuracies csf'].isna()]
df_mix['Model accuracies'][df_mix['base'] == 'CSF based prediction'] = df_mix['Model accuracies csf'][df_mix['Model accuracies gm'].isna()]
ax = sns.violinplot(x="Diagnosis group", y="Model accuracies", hue="base", data=df_mix, palette="muted")
plt.axhline(y=0.5, color="grey", linestyle='--')
plt.xticks(fontsize=10, rotation=45)
plt.legend(loc="lower right", fontsize=10) #bbox_to_anchor = (1.01, 0.6), 
plt.ylabel('Model accuracies', fontsize=14)
plt.xlabel('Diagnosis group', fontsize=14)
sns.despine(bottom=True)
plt.title("Prediction controls vs diagnosis group", fontsize=14)
plt.tight_layout()
plt.savefig('_figures/final_figs/accuracies_plot_gm_std.png')
plt.close('all')











#############################################################
########################## FIGURE 4 #########################
#############################################################

#############################################
# SIGNIFICANT ROI COEFFICIENT HISTOGRAM  ####
#############################################

# grey matter low level
df_sign = pd.read_pickle('_pickles/gm_predictions/df_significant_roi_Low_vs_CN')
df = pd.read_pickle('_pickles/gm_cleaned_atlas_harvard_oxford')
# group label
cluster = pd.read_excel('_createdDataframe/df_scores_std.xlsx', index_col=0)['cluster']
df['cluster'] = cluster
low = df[df['cluster'] == 'Low']
low.cluster = 'Cluster-derived mixte'
CN = df[df['cluster'] == 'CN']
CN.cluster = 'Controls'
df_ = pd.concat((low, CN))

sign_roi = [label for label in df_sign.index if df_sign.loc[label].values != 0]
sign_roi.append('cluster')
df_ = df_[sign_roi]
df_['cluster'][df_['cluster'] == 'Cluster-derived mixte'] = 'Mixed MCI'
df_ = df_.rename(columns={'cluster':'Group'})
# SCATTERPLOT
melted_df = pd.melt(df_, id_vars='Group')
melted_df = melted_df.rename(columns={'value':'Value (std)'})

# plotting
sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(8, 4))
sns.stripplot(x="Value (std)", y="variable", hue='Group', data=melted_df,
                        hue_order=['Controls', 'Mixed MCI'],
          size=2, color=".8", palette=['deeppink', 'lime'], alpha=0.8)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True, bottom=True)
plt.title('Grey matter volume distribution per significant ROI', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('_figures/final_figs/significant_stipplot.png')
plt.close('all')

import seaborn as sns
sns.set_style(style="whitegrid", rc={'grid.color': '0.1', 'axes.edgecolor':'1'})
melted_df = pd.melt(df_, id_vars='Group')
for roi in sign_roi[:-1]:
    print(roi)
    g = sns.catplot(
    data=melted_df.loc[melted_df['variable'] == roi], kind="boxen",
    x="variable", y="value", hue="Group",
    ci=95, palette=['lime', 'deeppink'], height=5, aspect=0.5, legend=False
    )
    g.despine(left=True)
    g.set_axis_labels("", "Volume (std)")
    plt.xticks([])
    plt.tight_layout()
    plt.savefig('_figures/final_figs/significant_repartition{}.png'.format(roi))
    plt.close('all')


plotting.plot_glass_brain('_niftiFiles/Low_vs_CN_coeffs.nii', '_figures/final_figs/coef_brain.png', threshold=0.01, colorbar=True,
                            plot_abs=False, display_mode='l', cmap='coolwarm')

fsaverage = ds.fetch_surf_fsaverage()
from nilearn import surface
texture = surface.vol_to_surf('_niftiFiles/Low_vs_CN_coeffs.nii', fsaverage.pial_right)
plotting.plot_surf_stat_map(fsaverage['pial_right'], texture,
                            hemi='right',  title='Pial surface - right hemisphere',
                            threshold=0.001, bg_map=fsaverage['sulc_right'],
                            view='medial', cmap='coolwarm',output_file='_figures/final_figs/coef_rightbrain.png')









#############################################################
########################## SUPP FIGURE 1 ####################
#############################################################


##########################################
# VISUALISATION OF OTHER VARIABLES #######
##########################################


col_to_check = ['cluster', 'Sex', 'Age',
       'Session', 'ID', 'TurnedAD', 'APOE4', 'GDS_tot', 'FAQ_tot', 'ANART']

df = df_scores_standardized.copy()
df_CN = df_scores_standardized[col_to_check][df_scores_standardized['cluster'] == 'CN']
df_High = df_scores_standardized[col_to_check][df_scores_standardized['cluster'] == 'High']
df_Low = df_scores_standardized[col_to_check][df_scores_standardized['cluster'] == 'Low']
df_Middle = df_scores_standardized[col_to_check][df_scores_standardized['cluster'] == 'Middle']
df_SMC = df_scores_standardized[col_to_check][df_scores_standardized['cluster'] == 'SMC']

# adapt cluster names:
df['cluster'][df['cluster'] == 'Low'] = 'Cluster-derived mixte'
df['cluster'][df['cluster'] == 'Middle'] = 'Cluster-derived amnesic'
df['cluster'][df['cluster'] == 'High'] = 'Cluster-derived normal'
df['cluster'][df['cluster'] == 'CN'] = 'Controls'

################
## PLOTTING session per group
################
import seaborn as sns
sns.set_theme(style="whitegrid")
# Draw a nested barplot by species and sex
sns.countplot(
    data=df,
    order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'],
    x="cluster", hue='Session',
    palette="dark")
plt.title('ADNI session per group')
sns.despine(trim=True, left=True)
plt.xticks(fontsize=12, rotation=45)
plt.xlabel('Diagnosis', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.savefig('_figures/final_figs/Session_per_group.png')
plt.close('all')


################
## PLOTTING sex per group
################
import seaborn as sns
sns.set_theme(style="whitegrid")
# Draw a nested barplot by species and sex
sns.countplot(
    data=df,
    order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'],
    x="cluster", hue='Sex',
    palette="dark")
plt.title('Male and female per group')
sns.despine(trim=True, left=True)
plt.xticks(fontsize=12, rotation=45)
plt.xlabel('Diagnosis', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.savefig('_figures/final_figs/Sex_per_group.png')
plt.close('all')


################
## PLOTTING APOE4 per group
################
import seaborn as sns
sns.set_theme(style="whitegrid")
# Draw a nested barplot by species and sex
sns.countplot(
    data=df,
    order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'],
    x="cluster", hue='APOE4',
    palette="dark")
plt.title('APOE4 allele per group')
sns.despine(trim=True, left=True)
plt.xticks(fontsize=12, rotation=45)
plt.xlabel('Diagnosis', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.savefig('_figures/final_figs/APOE4_per_group.png')
plt.close('all')


################
## PLOTTING FAQ 
################

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="FAQ_tot", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df[['cluster', 'FAQ_tot']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="FAQ_tot", y="cluster",  order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df[['cluster', 'FAQ_tot']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.xticks(fontsize=12)
plt.xlabel('FAQ scores', fontsize=14)
plt.title('FAQ scores per diagnosis group', fontsize=16)
plt.tight_layout()
plt.savefig('_figures/final_figs/FAQ_per_group.png')
plt.close('all')

################
## PLOTTING GDS 
################

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="GDS_tot", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic', 'Cluster-derived mixte'], data=df[['cluster', 'GDS_tot']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="GDS_tot", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df[['cluster', 'GDS_tot']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.xticks(fontsize=12)
plt.xlabel('GDS scores', fontsize=14)
plt.title('GDS scores per diagnosis group', fontsize=16)
plt.tight_layout()
plt.savefig('_figures/final_figs/GDS_per_group.png')
plt.close('all')

################
## PLOTTING GDS 
################

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="ANART", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df[['cluster', 'ANART']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="ANART", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df[['cluster', 'ANART']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.xticks(fontsize=12)
plt.xlabel('ANART scores', fontsize=14)
plt.title('ANART scores per diagnosis group', fontsize=16)
plt.tight_layout()
plt.savefig('_figures/final_figs/ANART_per_group.png')
plt.close('all')




#######
# Age
#######

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="Age", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df[['cluster', 'Age']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="Age", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df[['cluster', 'Age']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.xlabel('Age', fontsize=14)
plt.title('Age per diagnosis group', fontsize=16)
plt.tight_layout()
plt.savefig('_figures/final_figs/Age_per_group.png')
plt.close('all')


sns.set_theme(style="whitegrid")
df__ = pd.melt(df[['cluster', 'Age']], id_vars=['cluster'])
# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df__, kind="bar",
    order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'],
    x="cluster", y="value",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Mean age")
plt.title('Age per diagnosis group', fontsize=16)
plt.ylim([60, 85])
plt.xticks(fontsize=12, rotation=45)
plt.tight_layout()
plt.savefig('_figures/final_figs/age_per_group_histo.png')
plt.close('all')

################
## PLOTTING TURNING ALZHEIMER 
################

dfs = [df_CN, df_SMC, df_High, df_Low, df_Middle]
AD_per_gr = {}
rapport = {}
tot_per_gr = {}
for df_ in dfs:
    name = np.unique(df_.cluster)[0]
    nb_AD = df_.TurnedAD.sum()
    AD_per_gr[name] = nb_AD
    rapport[nb_AD] = df_.shape[0]

AD_per_gr["Controls"] = AD_per_gr.pop("CN")
AD_per_gr["Cluster-derived normal"] = AD_per_gr.pop("High")
AD_per_gr["Cluster-derived amnesic"] = AD_per_gr.pop("Middle")
AD_per_gr["Cluster-derived mixte"] = AD_per_gr.pop("Low")

def func(pct, x):
    print(pct)
    if int(pct) == 9:
        absolute = 28
    elif int(pct) == 0:
        absolute = 1
    elif int(pct) == 4:
        absolute = 14
    elif int(pct) == 61:
        absolute = 190
    elif int(pct) == 25:
        absolute = 78

    return "{} / {} part. ".format(absolute, rapport[absolute])

plt.figure(figsize = (8, 8))
x = [v for v in AD_per_gr.values()]
values = [v for v in AD_per_gr.values()]
plt.pie(x, labels = [v for v in AD_per_gr.keys()], 
    colors = ['antiquewhite', 'tan', 'gainsboro', 'dimgray', 'darkgrey'],
    autopct= lambda pct: func(pct, rapport), 
    pctdistance = 0.7,
    textprops=dict(color="black", fontsize=18),
    wedgeprops = {'linewidth': 0},
    normalize = True)
# plt.legend(loc='lower left', fontsize=14)
plt.title('AD turned participants per diagnosis group', fontsize=20)
plt.tight_layout()
plt.show()
plt.savefig('_figures/final_figs/TurnedAD_per_group.png')
plt.close('all')


