"""
Script to make every figures from the ADNI analysis
2022
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

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



# Set up the matplotlib figure
f, axarr = plt.subplots(3, 1, figsize=(12, 8), sharex=True) 
# Take care of the colors
my_palette = ['#e74c3c'] * 2 + ['#3498db'] * 7 + ['purple'] * 3 + ['#00BB00'] * 2
sns.set_palette(my_palette)
for i_cl, df in enumerate([df_low, df_middle, df_high]):
    n_subs = len(df.index)/13
    ax = sns.barplot('variable', 'value', data=df, palette=my_palette, ci='sd', ax=axarr[i_cl])
    if i_cl == 0:
        ax.xaxis.set_ticks_position('top')
        rotateTickLabels(ax, 45, 'x')
        ax.xaxis.set_ticklabels(X_colnames, fontsize=16)
        cluster_name = "Mixed MCI"

    if i_cl == 1:
        cluster_name = "Amnestic MCI"
        plt.tick_params(axis='x', labelbottom=False)

    if i_cl == 2:
        cluster_name = "Cluster-derived normal"
        plt.tick_params(axis='x', labelbottom=False)

    ax.set_ylim([0, 3])
    ax.set_xlabel("{} : {} patients".format(cluster_name, n_subs), fontsize=9)
plt.text(0, -1, "Visio-spatial", c='#e74c3c')
plt.text(4, -1, "Memory", c='#3498db')
plt.text(9, -1, "Language", c='purple')
plt.text(12, -1, "Executive", c='#00BB00')
plt.tight_layout(h_pad=3)
plt.show()
plt.savefig('_figures/final_figs/clustering_barplot_MCI.png')
plt.close('all')



###################################
## same but with High CN and SMC ##
###################################

# get scores of group CN
df_cn = pd.melt(df_scores_standardized[df_scores_standardized['cluster'] == 'CN'][col_for_clustering])
df_cn['value'] = df_cn['value'] + 1.5

# get scores of group SMC
df_smc = pd.melt(df_scores_standardized[df_scores_standardized['cluster'] == 'SMC'][col_for_clustering])
df_smc['value'] = df_smc['value'] + 1.5

# Set up the matplotlib figure
f, axarr = plt.subplots(3, 1, figsize=(12, 8), sharex=True) 
# Take care of the colors
my_palette = ['#e74c3c'] * 2 + ['#3498db'] * 7 + ['purple'] * 3 + ['#00BB00'] * 2
sns.set_palette(my_palette)
for i_cl, df in enumerate([df_high, df_smc, df_cn]):
    n_subs = len(df.index)/13
    ax = sns.barplot('variable', 'value', data=df, palette=my_palette, ci='sd', ax=axarr[i_cl])
    if i_cl == 0:
        ax.xaxis.set_ticks_position('top')
        rotateTickLabels(ax, 45, 'x')
        ax.xaxis.set_ticklabels(X_colnames, fontsize=16)
        ax.set_ylabel("Cluster-derived normal", fontsize=16, rotation=85)

    if i_cl == 1:
        ax.set_ylabel("SMC", fontsize=16, rotation=85)
        plt.tick_params(axis='x', labelbottom=False)

    if i_cl == 2:
        ax.set_ylabel("Controls", fontsize=16, rotation=85)
        plt.tick_params(axis='x', labelbottom=False)

    ax.set_ylim([0, 3])
    ax.set_xlabel('%i patients' % n_subs, fontsize=9)
plt.text(0, -1, "Visio-spatial", c='#e74c3c')
plt.text(4, -1, "Memory", c='#3498db')
plt.text(9, -1, "Language", c='purple')
plt.text(12, -1, "Executive", c='#00BB00')
plt.tight_layout(h_pad=3)
plt.savefig('_figures/final_figs/clustering_barplot.png')
plt.close('all')

###################################
## same but with all groups in ##
###################################

# Set up the matplotlib figure
f, axarr = plt.subplots(5, 1, figsize=(16, 12), sharex=True) 
# Take care of the colors
my_palette = ['#e74c3c'] * 2 + ['#3498db'] * 7 + ['purple'] * 3 + ['#00BB00'] * 2
sns.set_palette(my_palette)
for i_cl, df in enumerate([df_low, df_middle, df_high, df_smc, df_cn]):
    n_subs = len(df.index)/13
    ax = sns.barplot('variable', 'value', data=df, palette=my_palette, ci='sd', ax=axarr[i_cl])
    if i_cl == 0:
        ax.xaxis.set_ticks_position('top')
        rotateTickLabels(ax, 45, 'x')
        ax.xaxis.set_ticklabels(X_colnames, fontsize=20)
        ax.set_ylabel("Cluster-derived mixte", fontsize=20, rotation=85)

    if i_cl == 1:
        ax.set_ylabel("Cluster-derived amnesic", fontsize=20, rotation=85)
        plt.tick_params(axis='x', labelbottom=False)

    if i_cl == 2:
        ax.set_ylabel("Cluster-derived normal", fontsize=20, rotation=85)
        plt.tick_params(axis='x', labelbottom=False)

    if i_cl == 3:
        ax.set_ylabel("SMC", fontsize=20, rotation=85)
        plt.tick_params(axis='x', labelbottom=False)

    if i_cl == 4:
        ax.set_ylabel("Controls", fontsize=20, rotation=85)
        plt.tick_params(axis='x', labelbottom=False)

    ax.set_ylim([0, 3])
    ax.set_xlabel('%i patients' % n_subs, fontsize=9)
plt.text(0, -1.3, "Visio-spatial", c='#e74c3c')
plt.text(4, -1.3, "Memory", c='#3498db')
plt.text(9, -1.3, "Language", c='purple')
plt.text(12, -1.3, "Executive", c='#00BB00')
plt.tight_layout(h_pad=3)
plt.savefig('_figures/final_figs/clustering_barplot_allsub.png')
plt.close('all')



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
values_smc = df_scores_standardized[df_scores_standardized['Group'] == 'SMC'][col_for_clustering].mean(axis=0) + 1
values_smc = values_smc.values.tolist()
values_smc += values_smc[:1]
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
plt.xticks(angles_lab, categories, color='grey', size=8)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.5,1,1.5], ["0.5","1","1.5"], color="grey", size=7)
plt.ylim(0,2)
 
# Plot data
ax.plot(angles, values0, linewidth=1.5, linestyle='solid', color='orange', label='Cluster-derived mixte')
ax.plot(angles, values1, linewidth=1.5, linestyle='solid', color='lightgreen', label='Cluster-derived amnesic')
ax.plot(angles, values2, linewidth=1.5, linestyle='solid', color='lightcoral', label='Cluster-derived normal')
ax.plot(angles, values_smc, linewidth=1, linestyle='dotted', color='grey', label='SMC', alpha=0.9)
ax.plot(angles, values_cn, linewidth=1, linestyle='dotted', color='black', label='Controls', alpha=0.9)


# Fill area
# ax.fill(angles, values, 'b', alpha=0.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
# Save the graph
plt.tight_layout()
plt.savefig('_figures/final_figs/polarplot.png')
plt.close('all')




#################
## Plot as PCA ##
#################

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
acp_smc = df_scores_standardized[['acp1', 'acp2']].loc[df_scores_standardized[df_scores_standardized.cluster == 'SMC'].index]

f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(acp_low['acp1'].values, acp_low['acp2'].values, c='dimgray', s=100, label="Cluster-derived mixte scores")
plt.scatter(acp_middle['acp1'].values, acp_middle['acp2'].values, c='darkgrey', s=100, label="Cluster-derived amnesic scores")
plt.scatter(acp_high['acp1'].values, acp_high['acp2'].values, c='gainsboro', s=100, label="Cluster-derived normal")
# plt.scatter(acp_smc['acp1'].values, acp_smc['acp2'].values, c='tan', alpha=0.3, s=70, label="SMC")
# plt.scatter(acp_cn['acp1'].values, acp_cn['acp2'].values, c='antiquewhite', alpha=0.3, s=70, label="controls")
plt.text(-2.5, 4.5, "Composant 1: {}".format(np.round(explained_variance[0]*100, 2)),fontsize=20)
plt.text(-2.5, 4, "Composant 2: {}".format(np.round(explained_variance[1]*100, 2)),fontsize=20)
plt.legend(loc='upper left', fontsize=20)
plt.xlabel('Composant 1')
plt.ylabel('Composant 2')
plt.tight_layout()
plt.savefig('_figures/final_figs/ACP.png')
plt.close('all')



######################################
## RUN PCA analysis with 2 components to 
## get a better visualisation 
## with turned AD and APOE4 
######################################

acp_ad = df_scores_standardized[df_scores_standardized.TurnedAD == 1][['acp1', 'acp2']]

f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(acp_low['acp1'].values, acp_low['acp2'].values, c='dimgray', s=100, label="Cluster-derived mixte scores")
plt.scatter(acp_middle['acp1'].values, acp_middle['acp2'].values, c='darkgrey', s=100, label="Cluster-derived amnesic scores")
plt.scatter(acp_smc['acp1'].values, acp_smc['acp2'].values, c='orange', alpha=0.2, s=70, label="SMC") # tan
plt.scatter(acp_cn['acp1'].values, acp_cn['acp2'].values, c='yellow', alpha=0.2, s=70, label="controls") # 'antiquewhite'
plt.scatter(acp_high['acp1'].values, acp_high['acp2'].values, c='gainsboro', s=100, label="Cluster-derived normal") #
plt.scatter(acp_ad['acp1'].values, acp_ad['acp2'].values, s=5, facecolors='none', alpha=0.9, edgecolors='lime', label="Turned AD")

plt.text(-2.5, 4.5, "Composant 1: {}".format(np.round(explained_variance[0]*100, 2)),fontsize=20)
plt.text(-2.5, 4, "Composant 2: {}".format(np.round(explained_variance[1]*100, 2)),fontsize=20)

plt.legend()
plt.xlabel('Composant 1')
plt.ylabel('Composant 2')
plt.legend(loc='upper left', fontsize=20)
plt.tight_layout()
# plt.savefig('_figures/ACP_turnedAD_apoe4.png')
plt.savefig('_figures/final_figs/ACP_turnedAD.png')
plt.close('all')



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
plt.savefig('_figures/final_figs/TurnedAD_per_group.png')
plt.close('all')



################
## PLOTTING TAU et Amyloid Beta
################
np.random.seed(0)

df_ = pd.read_pickle('_pickles/csf_level_std')


#######
# TAU
#######

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="TAU", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df_[['cluster', 'TAU']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="TAU", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df[['cluster', 'TAU']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title('TAU quantity per diagnosis group', fontsize=16)
plt.tight_layout()
plt.savefig('_figures/final_figs/TAU_per_group.png')
plt.close('all')

#######
# PTAU
#######

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="PTAU", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df_[['cluster', 'PTAU']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="PTAU", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df[['cluster', 'PTAU']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title('PTAU quantity per diagnosis group', fontsize=16)
plt.tight_layout()
plt.savefig('_figures/final_figs/PTAU_per_group.png')
plt.close('all')

#######
# ABETA
#######

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="ABETA", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df_[['cluster', 'ABETA']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="ABETA", y="cluster", order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'], data=df[['cluster', 'ABETA']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title('ABETA quantity per diagnosis group', fontsize=16)
plt.tight_layout()
plt.savefig('_figures/final_figs/ABETA_per_group.png')
plt.close('all')


######
# TAU/PTAU + ABETA
######

import seaborn as sns
sns.set_theme(style="whitegrid")

df__ = pd.melt(df_[['cluster', 'ABETA', 'PTAU', 'TAU']], id_vars=['cluster'])

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df__, kind="bar",
    order=['Controls', 'SMC', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'],
    x="cluster", y="value", hue="variable",
    ci="sd", palette="dark", alpha=.6, height=6, legend=False
)
g.despine(left=True)
g.set_axis_labels("", "Level")
plt.legend(title="CSF rates", loc='upper right')

plt.xticks(fontsize=14, rotation=45)
plt.tight_layout()
plt.savefig('_figures/final_figs/csf_per_group.png')
plt.close('all')


######
# TAU + ABETA
######

import seaborn as sns
sns.set_theme(style="whitegrid")

df__ = pd.melt(df_[['cluster', 'ABETA', 'TAU']], id_vars=['cluster'])

g = sns.catplot(
    data=df__, kind="point",
    order=['Controls', 'Cluster-derived normal', 'Cluster-derived amnesic','Cluster-derived mixte'],
    x="cluster", y="value", hue="variable",
    ci=95, palette="dark", alpha=.6, height=6, legend=False
)
g.despine(left=True)
g.set_axis_labels("", "Level")
plt.legend(title="CSF rates", loc='upper right')

plt.xticks(fontsize=14, rotation=45)
plt.tight_layout()
plt.savefig('_figures/final_figs/csf_per_group.png')
plt.close('all')



######
# TAU/PTAU against ABETA
######

# plot tau against amyloid per group per ad
df_ = df.loc[(df[['TAU', 'PTAU', 'ABETA']]!=0).any(axis=1)]

# TAU 
f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(df_['TAU'][df_.cluster == 'Cluster-derived mixte'].values, df_['ABETA'][df_.cluster == 'Cluster-derived mixte'].values, c='dimgray', s=100, alpha=0.9, label="Cluster-derived mixte")
plt.scatter(df_['TAU'][df_.cluster == 'Cluster-derived amnesic'].values, df_['ABETA'][df_.cluster == 'Cluster-derived amnesic'].values, c='darkgrey',alpha=0.8, s=100, label="Cluster-derived amnesic")
plt.scatter(df_['TAU'][df_.cluster == 'Cluster-derived normal'].values, df_['ABETA'][df_.cluster == 'Cluster-derived normal'].values, c='gainsboro',alpha=0.8, s=100, label="Cluster-derived normal")
plt.scatter(df_['TAU'][df_.cluster == 'SMC'].values, df_['ABETA'][df_.cluster == 'SMC'].values, c='tan', alpha=0.3, s=120, label="smc")
plt.scatter(df_['TAU'][df_.cluster == 'Controls'].values, df_['ABETA'][df_.cluster == 'Controls'].values, c='antiquewhite', alpha=0.6, s=90, label="Controls")
plt.scatter(df_['TAU'][df_.TurnedAD == 1].values, df_['ABETA'][df_.TurnedAD == 1].values, s=100, facecolors='none', alpha=0.3, edgecolors='green', label="Turned AD")
plt.legend(fontsize=20)
plt.xlabel('TAU',fontsize=20)
plt.ylabel('ABETA',fontsize=20)
plt.tight_layout()
plt.savefig('_figures/final_figs/Tau_amyloid_AD.png')
plt.close('all')

# PTAU
f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(df_['PTAU'][df_.cluster == 'Cluster-derived mixte'].values, df_['ABETA'][df_.cluster == 'Cluster-derived mixte'].values, c='dimgray', s=100, alpha=0.9, label="Cluster-derived mixte")
plt.scatter(df_['PTAU'][df_.cluster == 'Cluster-derived amnesic'].values, df_['ABETA'][df_.cluster == 'Cluster-derived amnesic'].values, c='darkgrey',alpha=0.8, s=100, label="Cluster-derived amnesic")
plt.scatter(df_['PTAU'][df_.cluster == 'Cluster-derived normal'].values, df_['ABETA'][df_.cluster == 'Cluster-derived normal'].values, c='gainsboro',alpha=0.8, s=100, label="Cluster-derived normal")
plt.scatter(df_['PTAU'][df_.cluster == 'SMC'].values, df_['ABETA'][df_.cluster == 'SMC'].values, c='tan', alpha=0.3, s=120, label="smc")
plt.scatter(df_['PTAU'][df_.cluster == 'Controls'].values, df_['ABETA'][df_.cluster == 'Controls'].values, c='antiquewhite', alpha=0.6, s=90, label="Controls")
plt.scatter(df_['PTAU'][df_.TurnedAD == 1].values, df_['ABETA'][df_.TurnedAD == 1].values, s=100, facecolors='none', alpha=0.3, edgecolors='green', label="Turned AD")
plt.legend(fontsize=20)
plt.xlabel('PTAU',fontsize=20)
plt.ylabel('ABETA',fontsize=20)
plt.tight_layout()
plt.savefig('_figures/final_figs/PTau_amyloid_AD.png')
plt.close('all')


# just CN and LOW PTAU
f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(df_['PTAU'][df_.cluster == 'Cluster-derived mixte'].values, df_['ABETA'][df_.cluster == 'Cluster-derived mixte'].values, c='dimgray', s=100, alpha=0.9, label="Cluster-derived mixte")
plt.scatter(df_['PTAU'][df_.cluster == 'Controls'].values, df_['ABETA'][df_.cluster == 'Controls'].values, c='antiquewhite', alpha=0.6, s=90, label="Controls")
plt.scatter(df_['PTAU'][df_.TurnedAD == 1].values, df_['ABETA'][df_.TurnedAD == 1].values, s=100, facecolors='none', alpha=0.3, edgecolors='green', label="Turned AD")
plt.legend(fontsize=20)
plt.xlabel('PTAU',fontsize=20)
plt.ylabel('ABETA',fontsize=20)
plt.tight_layout()
plt.savefig('_figures/PTau_amyloid_AD_CNLow.png')
plt.close('all')

# just CN and LOW TAU
f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(df_['TAU'][df_.cluster == 'Cluster-derived mixte'].values, df_['ABETA'][df_.cluster == 'Cluster-derived mixte'].values, c='dimgray', s=100, alpha=0.9, label="Cluster-derived mixte")
plt.scatter(df_['TAU'][df_.cluster == 'Controls'].values, df_['ABETA'][df_.cluster == 'Controls'].values, c='antiquewhite', alpha=0.6, s=90, label="Controls")
plt.scatter(df_['TAU'][df_.TurnedAD == 1].values, df_['ABETA'][df_.TurnedAD == 1].values, s=100, facecolors='none', alpha=0.3, edgecolors='green', label="Turned AD")
plt.legend(fontsize=20)
plt.xlabel('TAU',fontsize=20)
plt.ylabel('ABETA',fontsize=20)
plt.tight_layout()
plt.savefig('_figures/Tau_amyloid_AD_CNLow.png')
plt.close('all')


#################################################
# VISUALISATION PREDICTION WITH GREY MATTER  ####
#################################################

# violin plot of accuracies

df_visu_gm = pd.read_pickle('_pickles/df_visu_gm')
df_visu_csf = pd.read_pickle('_pickles/df_visu_csf')


sns.set(style="white", context="talk")
df_visu_gm = df_visu_gm[['High_vs_CN', 'Middle_vs_CN', 'Low_vs_CN']]
df_visu_csf = df_visu_csf[['High_vs_CN', 'Middle_vs_CN', 'Low_vs_CN']]
df_visu_gm.rename(columns = {'High_vs_CN':'Cluster-derived normal', 'Middle_vs_CN':'Cluster-derived amnesic', 'Low_vs_CN':'Cluster-derived mixte'}, inplace = True)
df_visu_csf.rename(columns = {'High_vs_CN':'Cluster-derived normal', 'Middle_vs_CN':'Cluster-derived amnesic', 'Low_vs_CN':'Cluster-derived mixte'}, inplace = True)
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
plt.ylabel('Model scores', fontsize=14)
plt.xlabel('Diagnosis group', fontsize=14)
sns.despine(bottom=True)
plt.title("Prediction controls vs diagnosis group", fontsize=14)
plt.tight_layout()
plt.savefig('_figures/final_figs/accuracies_plot_gm.png')
plt.close('all')


#################################################
# RISK RATIO TURNING AD  ####
#################################################

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


nb_turnedAD = len(df_chisquare[df_chisquare['TurnedAD']==1][df_chisquare['cluster'] == 'SMC'])
tot = len(df_chisquare[df_chisquare['cluster'] == 'SMC'])
ratio_smc = nb_turnedAD/tot


RR_high = ratio_high/ratio_control
RR_middle = ratio_middle/ratio_control
RR_low = ratio_low/ratio_control
RR_smc = ratio_smc/ratio_control

import matplotlib.pyplot as plt
fig = plt.figure()
diags = ['SMC', 'Cluster-derived normal', 'Cluster-derived amnesic', 'Cluster-derived mixte']
RR = [RR_smc,RR_high,RR_middle,RR_low]
plt.bar(diags,RR, color=['tan', 'gainsboro', 'darkgrey', 'dimgray'])
plt.xlabel('Diagnosis', fontsize=20)
plt.ylabel('Risk ratio', fontsize=20)
plt.xticks(fontsize=20, rotation=35)
plt.title('Risk ratio of turning AD per group')
plt.axhline(y=1, linestyle='--', color='red')
plt.tight_layout()
plt.savefig('_figures/final_figs/RR_per_group.png')
plt.close('all')


#############################################
# SIGNIFICANT biomarkers   ####
#############################################
df_sign = pd.read_pickle('_pickles/csf_predictions/df_coefs_csf_Low_vs_CN')
f, ax = plt.subplots(figsize=(5, 3))
weights = np.array(df_sign.coef.values).astype('float64')
sns.heatmap(weights.reshape(1, 3),vmin=-1.1, vmax=1.1,  cmap='coolwarm', square=True, fmt='.2f', annot=True, cbar=False,linecolor='White')
# plt.xticks(range(len(X_colnames)), X_colnames)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_ticklabels(['Tau', 'PTau', 'ABeta'], fontsize=16)
rotateTickLabels(ax, 45, 'x')
y_ticks = ["Coefficients"]
ax.yaxis.set_ticklabels(y_ticks, fontsize=16, rotation=0)
plt.tight_layout()
plt.savefig('_figures/final_figs/significant_biomarkers_low.png')
plt.close('all')

df_sign = pd.read_pickle('_pickles/csf_predictions/df_coefs_csf_Middle_vs_CN')
f, ax = plt.subplots(figsize=(5, 3))
weights = np.array(df_sign.coef.values).astype('float64')
sns.heatmap(weights.reshape(1, 3),center=0, vmin=-1.1, vmax=1.1,  cmap='coolwarm', square=True, fmt='.2f', annot=True, cbar=False,linecolor='White')
# plt.xticks(range(len(X_colnames)), X_colnames)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_ticklabels(['Tau', 'PTau', 'ABeta'], fontsize=16)
rotateTickLabels(ax, 45, 'x')
y_ticks = ["Coefficients"]
ax.yaxis.set_ticklabels(y_ticks, fontsize=16, rotation=0)
plt.tight_layout()
plt.savefig('_figures/final_figs/significant_biomarkers_middle.png')
plt.close('all')



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


# SCATTERPLOT
melted_df = pd.melt(df_, id_vars='cluster')

# plotting
sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(10, 6))
sns.stripplot(x="value", y="variable", hue='cluster', data=melted_df,
                        hue_order=['Controls', 'Cluster-derived mixte'],
          size=4, color=".3", palette="vlag")

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True, bottom=True)
plt.title('singificant ROI differences', fontsize=16)
plt.tight_layout()
plt.show()


import seaborn as sns
sns.set_theme(style="whitegrid")

for roi in sign_roi[:-1]:
    print(roi)
    g = sns.catplot(
    data=melted_df.loc[melted_df['variable'] == roi], kind="boxen",
    x="variable", y="value", hue="cluster",
    ci=95, palette="dark", height=5, aspect=0.5, legend=False
    )
    g.despine(left=True)
    g.set_axis_labels("", "Level")
    plt.xticks(fontsize=3, rotation=45)
    plt.tight_layout()
    plt.savefig('_figures/final_figs/significant_repartition{}.png'.format(roi))
    plt.close('all')






