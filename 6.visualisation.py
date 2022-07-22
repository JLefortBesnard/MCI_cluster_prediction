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


####################################
## CHECK VARIANCE SUR CHAQUE TEST ##
####################################

df_melted = pd.melt(df_scores_standardized[col_for_clustering])

# df_melted = pd.melt(df_score_originaux)
sns.set_theme(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))

# Plot the orbital period with horizontal boxes
sns.boxplot(x="value", y="variable", data=df_melted,
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="value", y="variable", data=df_melted,
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
ys = [i for i in range(len(col_for_clustering))]
for y, var in zip(ys, variance):
       plt.text(5, y, np.round_(var, 2))
plt.tight_layout()
plt.savefig('_figures/neuropsychotest_variance.png')
plt.close('all')



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
        ax.xaxis.set_ticklabels(X_colnames, fontsize=9)
        ax.set_ylabel("Low", fontsize=12)

    if i_cl == 1:
        ax.set_ylabel("Middle", fontsize=12)
        plt.tick_params(axis='x', labelbottom=False)

    if i_cl == 2:
        ax.set_ylabel("High", fontsize=12)
        plt.tick_params(axis='x', labelbottom=False)

    ax.set_ylim([0, 3])
    ax.set_xlabel('%i patients' % n_subs, fontsize=9)
plt.text(0, -1, "Visio-spatial", c='#e74c3c')
plt.text(4, -1, "Memory", c='#3498db')
plt.text(9, -1, "Language", c='purple')
plt.text(12, -1, "Executive", c='#00BB00')
plt.tight_layout(h_pad=3)
plt.savefig('_figures/clustering_barplot_MCI.png')
plt.close('all')



###################################
## same but including CN and SMC ##
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
        ax.set_ylabel("HighMCI", fontsize=12)

    if i_cl == 1:
        ax.set_ylabel("SMC", fontsize=12)
        plt.tick_params(axis='x', labelbottom=False)

    if i_cl == 2:
        ax.set_ylabel("CN", fontsize=12)
        plt.tick_params(axis='x', labelbottom=False)

    ax.set_ylim([0, 3])
    ax.set_xlabel('%i patients' % n_subs, fontsize=9)
plt.text(0, -1, "Visio-spatial", c='#e74c3c')
plt.text(4, -1, "Memory", c='#3498db')
plt.text(9, -1, "Language", c='purple')
plt.text(12, -1, "Executive", c='#00BB00')
plt.tight_layout(h_pad=3)
plt.savefig('_figures/clustering_barplot.png')
plt.close('all')

###########################################################
## Plot clustering analysis results as barplot per tests ##
###########################################################

df_ = df_scores_standardized[col_for_clustering] + 1
df_['cluster'] = df_scores_standardized['cluster']
for ind, test in enumerate(col_for_clustering):
    df__ = df_[[test, 'cluster']]
    # Set up the matplotlib figure
    plt.subplots(figsize=(5, 5))
    colors = [my_palette[ind]] * 5
    sns.set_palette(colors)
    sns.barplot('cluster', test, data=df_, palette=colors, ci=None)
    plt.ylim(0, 3)
    plt.xticks(fontsize = 20)
    plt.tight_layout(h_pad=3)
    plt.savefig('_figures/barplot_per_test/barplot_{}.png'.format(test))
    plt.close('all')



#################################################
## Plot clustering analysis results as polar ##
#################################################
from math import pi

# number of variable
categories = col_for_clustering
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
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.5,1,1.5], ["0.5","1","1.5"], color="grey", size=7)
plt.ylim(0,2)
 
# Plot data
ax.plot(angles, values0, linewidth=1.5, linestyle='solid', color='orange', label='Low')
ax.plot(angles, values1, linewidth=1.5, linestyle='solid', color='lightgreen', label='Middle')
ax.plot(angles, values2, linewidth=1.5, linestyle='solid', color='lightcoral', label='High')
ax.plot(angles, values_smc, linewidth=1, linestyle='dotted', color='grey', label='SMC', alpha=0.9)
ax.plot(angles, values_cn, linewidth=1, linestyle='dotted', color='black', label='Controls', alpha=0.9)


# Fill area
# ax.fill(angles, values, 'b', alpha=0.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# Save and show the graph
plt.tight_layout()
plt.savefig('_figures/polarplot.png')
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
print("    composante 1 | composante 2")
for ind, col in enumerate(col_for_clustering):
    print(col, ' : ',coefs[0][ind], ' | ', coefs[1][ind])

# plot composant 1 et 2
sns.set_palette("Paired")
f, axarr = plt.subplots(figsize=(12, 12))
for ind, col in enumerate(col_for_clustering):
    plt.plot([0, coefs[0][ind]],[0, coefs[1][ind]], label=col)
plt.legend(prop={'size': 20})
plt.xlabel('Composant 1')
plt.ylabel('Composant 2')
plt.tight_layout()
plt.savefig('_figures/PCA_composante.png')
plt.close('all')



acp = pca.transform(df_scores_standardized[col_for_clustering].values)
df_scores_standardized[['acp1', 'acp2']] = acp

acp_high = df_scores_standardized[['acp1', 'acp2']].loc[df_scores_standardized[df_scores_standardized.cluster == 'High'].index]
acp_middle = df_scores_standardized[['acp1', 'acp2']].loc[df_scores_standardized[df_scores_standardized.cluster == 'Middle'].index]
acp_low = df_scores_standardized[['acp1', 'acp2']].loc[df_scores_standardized[df_scores_standardized.cluster == 'Low'].index]
acp_cn = df_scores_standardized[['acp1', 'acp2']].loc[df_scores_standardized[df_scores_standardized.cluster == 'CN'].index]
acp_smc = df_scores_standardized[['acp1', 'acp2']].loc[df_scores_standardized[df_scores_standardized.cluster == 'SMC'].index]

f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(acp_low['acp1'].values, acp_low['acp2'].values, c='dimgray', s=100, label="low scores")
plt.scatter(acp_middle['acp1'].values, acp_middle['acp2'].values, c='darkgrey', s=100, label="middle scores")
plt.scatter(acp_high['acp1'].values, acp_high['acp2'].values, c='gainsboro', s=100, label="high scores")
plt.scatter(acp_smc['acp1'].values, acp_smc['acp2'].values, c='tan', alpha=0.3, s=70, label="smc")
plt.scatter(acp_cn['acp1'].values, acp_cn['acp2'].values, c='antiquewhite', alpha=0.3, s=70, label="controls")
plt.legend()
plt.xlabel('Composant 1')
plt.ylabel('Composant 2')
plt.tight_layout()
plt.savefig('_figures/ACP.png')
plt.close('all')



######################################
## RUN PCA analysis with 2 components to 
## get a better visualisation 
## with turned AD and APOE4 
######################################

acp_1ap4 = df_scores_standardized[df_scores_standardized.APOE4 == 1][['acp1', 'acp2']]
acp_2ap4 = df_scores_standardized[df_scores_standardized.APOE4 == 2][['acp1', 'acp2']]
acp_ad = df_scores_standardized[df_scores_standardized.TurnedAD == 1][['acp1', 'acp2']]

f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(acp_low['acp1'].values, acp_low['acp2'].values, c='dimgray', s=100, label="low scores")
plt.scatter(acp_middle['acp1'].values, acp_middle['acp2'].values, c='darkgrey', s=100, label="middle scores")
plt.scatter(acp_high['acp1'].values, acp_high['acp2'].values, c='gainsboro', s=100, label="high scores")
plt.scatter(acp_smc['acp1'].values, acp_smc['acp2'].values, c='tan', alpha=0.3, s=70, label="smc")
plt.scatter(acp_cn['acp1'].values, acp_cn['acp2'].values, c='antiquewhite', alpha=0.3, s=70, label="controls")
# plt.scatter(acp_1ap4['acp1'].values, acp_1ap4['acp2'].values, c='deeppink', s=2, alpha=0.7, label="1 gene APOE4")
# plt.scatter(acp_2ap4['acp1'].values, acp_2ap4['acp2'].values, c='lime', s=2, alpha=0.7, label="2 genes APOE4")
plt.scatter(acp_ad['acp1'].values, acp_ad['acp2'].values, s=100, facecolors='none', alpha=0.7, edgecolors='yellow', label="Turned AD")

plt.text(-3, 4, "Composant 1: {}".format(np.round(explained_variance[0]*100, 2)),fontsize=9)
plt.text(-3, 3.5, "Composant 2: {}".format(np.round(explained_variance[1]*100, 2)),fontsize=9)

plt.legend()
plt.xlabel('Composant 1')
plt.ylabel('Composant 2')
plt.tight_layout()
# plt.savefig('_figures/ACP_turnedAD_apoe4.png')
plt.savefig('_figures/ACP_turnedAD.png')
plt.close('all')



##########################################
# VISUALISATION OF OTHER VARIABLES #######
##########################################


col_to_check = ['cluster', 'Sex', 'Age',
       'Session', 'ID', 'TurnedAD', 'APOE4', 'GDS_tot', 'FAQ_tot', 'ANART']

df = df_scores_standardized
df_CN = df_scores_standardized[col_to_check][df_scores_standardized['cluster'] == 'CN']
df_High = df_scores_standardized[col_to_check][df_scores_standardized['cluster'] == 'High']
df_Low = df_scores_standardized[col_to_check][df_scores_standardized['cluster'] == 'Low']
df_Middle = df_scores_standardized[col_to_check][df_scores_standardized['cluster'] == 'Middle']
df_SMC = df_scores_standardized[col_to_check][df_scores_standardized['cluster'] == 'SMC']



################
## PLOTTING session per group
################
import seaborn as sns
sns.set_theme(style="whitegrid")
# Draw a nested barplot by species and sex
sns.countplot(
    data=df,
    order=['CN', 'SMC', 'High', 'Middle', 'Low'],
    x="cluster", hue='Session',
    palette="dark")
plt.title('ADNI session per group')
sns.despine(trim=True, left=True)
plt.savefig('_figures/Session_per_group.png')
plt.close('all')


################
## PLOTTING sex per group
################
import seaborn as sns
sns.set_theme(style="whitegrid")
# Draw a nested barplot by species and sex
sns.countplot(
    data=df,
    order=['CN', 'SMC', 'High', 'Middle', 'Low'],
    x="cluster", hue='Sex',
    palette="dark")
plt.title('Male and female per group')
sns.despine(trim=True, left=True)
plt.savefig('_figures/Sex_per_group.png')
plt.close('all')

################
## PLOTTING FAQ 
################

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="FAQ_tot", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'FAQ_tot']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="FAQ_tot", y="cluster",  order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'FAQ_tot']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title('FAQ scores per diagnosis group', fontsize=16)
plt.savefig('_figures/FAQ_per_group.png')
plt.close('all')

################
## PLOTTING GDS 
################

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="GDS_tot", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'GDS_tot']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="GDS_tot", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'GDS_tot']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title('GDS scores per diagnosis group', fontsize=16)
plt.savefig('_figures/GDS_per_group.png')
plt.close('all')

################
## PLOTTING GDS 
################

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="ANART", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'ANART']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="ANART", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'ANART']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title('ANART scores per diagnosis group', fontsize=16)
plt.savefig('_figures/ANART_per_group.png')
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
    textprops=dict(color="black", fontsize=14),
    wedgeprops = {'linewidth': 0},
    normalize = True)
plt.legend(loc='lower left', fontsize=14)
plt.title('AD turned participants per diagnosis group', fontsize=16)
plt.savefig('_figures/TurnedAD_per_group.png')
plt.close('all')


# as histogram
import seaborn as sns
sns.set_theme(style="whitegrid")
# Draw a nested barplot by species and sex
sns.countplot(
    data=df,
    order=['CN', 'SMC', 'High', 'Middle', 'Low'],
    x="cluster", hue='TurnedAD',
    palette="dark")
plt.title('Turned AD per group')
sns.despine(trim=True, left=True)
plt.savefig('_figures/TurnedAD_histogram_per_group.png')
plt.close('all')


################
## PLOTTING APOE4
################


dfs = [df_CN, df_SMC, df_High, df_Low, df_Middle]

for df_ in dfs:
    name = np.unique(df_.cluster)[0]
    zero = df_['APOE4'][df_['APOE4'] ==0].shape[0]
    un = df_['APOE4'][df_['APOE4'] ==1].shape[0]
    deux = df_['APOE4'][df_['APOE4'] ==2].shape[0]
    apoe4_values = [zero, un, deux]
    labels = ("0 allele APOE4", "1 allele APOE4","2 alleles APOE4")
    plt.figure(figsize = (8, 8))
    plt.pie(apoe4_values, labels = [v for v in labels], 
        colors = ['gainsboro', 'darkgrey', 'dimgray'],
        autopct= lambda pct: str(int(pct)) + ' %', 
        pctdistance = 0.7,
        textprops=dict(color="black", fontsize=14),
        wedgeprops = {'linewidth': 0},
        normalize = True)
    plt.legend(loc='lower left', fontsize=14)
    plt.title('APOE4 participants for {} participants'.format(name), fontsize=16)
    plt.tight_layout()
    plt.savefig('_figures/APOE4_{}.png'.format(name))
    plt.close('all')



#######
# Age
#######

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="Age", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'Age']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="Age", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'Age']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title('Age per diagnosis group', fontsize=16)
plt.savefig('_figures/Age_per_group.png')
plt.close('all')

import seaborn as sns
sns.set_theme(style="whitegrid")

df__ = pd.melt(df[['cluster', 'Age']], id_vars=['cluster'])

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df__, kind="bar",
    order=['CN', 'SMC', 'High', 'Middle', 'Low'],
    x="cluster", y="value",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Mean age")
plt.title('Age per diagnosis group', fontsize=16)
plt.ylim([60, 85])
plt.savefig('_figures/age_per_group_histo.png')
plt.close('all')


################
## PLOTTING TAU et Amyloid Beta
################
df[['TAU', 'PTAU', 'ABETA']] = 0
df_csf1 = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/CSF.csv")
df_csf2 = pd.read_csv("filesFromADNI/ADNI_cognitiveTests/csf_adni1_go_2.csv")
col_csf_to_keep = ['RID', 'ABETA', 'TAU', 'PTAU']
df_csf1 = df_csf1[col_csf_to_keep]
df_csf2 = df_csf2[col_csf_to_keep]
df_csf = pd.concat((df_csf1, df_csf2))
df_csf = df_csf.astype({'RID': str})
df_csf = df_csf.set_index("RID")

missing=[]
for index in df.index:
    success = 0
    try:
        df_csf[['TAU', 'PTAU', 'ABETA']].loc[str(index)]
        success = 1
    except:
        print('no data for ', index)
        missing.append(index)
    if success == 1:
        df.loc[index, ['TAU', 'PTAU', 'ABETA']] = df_csf[['TAU', 'PTAU', 'ABETA']].loc[str(index)].mean().values
print('missing {} participants'.format(len(missing)))
np.unique(df.loc[missing]['cluster'], return_counts=True)
np.unique(df.loc[missing]['Session'], return_counts=True)



df_ = df.loc[(df[['TAU', 'PTAU', 'ABETA']]!=0).any(axis=1)]
df_ = df_.dropna
#######
# TAU
#######

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="TAU", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df_[['cluster', 'TAU']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="TAU", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'TAU']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title('TAU quantity per diagnosis group', fontsize=16)
plt.savefig('_figures/TAU_per_group.png')
plt.close('all')

#######
# PTAU
#######

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="PTAU", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df_[['cluster', 'PTAU']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="PTAU", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'PTAU']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title('PTAU quantity per diagnosis group', fontsize=16)
plt.savefig('_figures/PTAU_per_group.png')
plt.close('all')

#######
# ABETA
#######

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
# Plot the orbital period with horizontal boxes
sns.boxplot(x="ABETA", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df_[['cluster', 'ABETA']],
            whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(x="ABETA", y="cluster", order=['CN', 'SMC', 'High', 'Middle', 'Low'], data=df[['cluster', 'ABETA']],
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title('ABETA quantity per diagnosis group', fontsize=16)
plt.savefig('_figures/ABETA_per_group.png')
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
    order=['CN', 'SMC', 'High', 'Middle', 'Low'],
    x="cluster", y="value", hue="variable",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Level")
g.legend.set_title("CSF rates")
plt.savefig('_figures/csf_per_group.png')
plt.close('all')


######
# TAU/PTAU against ABETA
######

# plot tau against amyloid per group per ad
df_ = df.loc[(df[['TAU', 'PTAU', 'ABETA']]!=0).any(axis=1)]

# TAU 
f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(df_['TAU'][df_.cluster == 'Middle'].values, df_['ABETA'][df_.cluster == 'Middle'].values, c='darkgrey',alpha=0.8, s=100, label="middle scores")
plt.scatter(df_['TAU'][df_.cluster == 'High'].values, df_['ABETA'][df_.cluster == 'High'].values, c='gainsboro',alpha=0.8, s=100, label="high scores")
plt.scatter(df_['TAU'][df_.cluster == 'SMC'].values, df_['ABETA'][df_.cluster == 'SMC'].values, c='tan', alpha=0.3, s=120, label="smc")
plt.scatter(df_['TAU'][df_.cluster == 'CN'].values, df_['ABETA'][df_.cluster == 'CN'].values, c='antiquewhite', alpha=0.6, s=90, label="controls")
plt.scatter(df_['TAU'][df_.cluster == 'Low'].values, df_['ABETA'][df_.cluster == 'Low'].values, c='dimgray', s=100, alpha=0.6, label="low scores")
plt.scatter(df_['TAU'][df_.TurnedAD == 1].values, df_['ABETA'][df_.TurnedAD == 1].values, s=100, facecolors='none', alpha=0.7, edgecolors='green', label="Turned AD")
plt.legend()
plt.xlabel('TAU')
plt.ylabel('ABETA')
plt.tight_layout()
plt.savefig('_figures/Tau_amyloid_AD.png')
plt.close('all')

# PTAU
f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(df_['PTAU'][df_.cluster == 'Middle'].values, df_['ABETA'][df_.cluster == 'Middle'].values, c='darkgrey',alpha=0.8, s=100, label="middle scores")
plt.scatter(df_['PTAU'][df_.cluster == 'High'].values, df_['ABETA'][df_.cluster == 'High'].values, c='gainsboro',alpha=0.8, s=100, label="high scores")
plt.scatter(df_['PTAU'][df_.cluster == 'SMC'].values, df_['ABETA'][df_.cluster == 'SMC'].values, c='tan', alpha=0.3, s=120, label="smc")
plt.scatter(df_['PTAU'][df_.cluster == 'CN'].values, df_['ABETA'][df_.cluster == 'CN'].values, c='antiquewhite', alpha=0.6, s=90, label="controls")
plt.scatter(df_['PTAU'][df_.cluster == 'Low'].values, df_['ABETA'][df_.cluster == 'Low'].values, c='dimgray', s=100, alpha=0.6, label="low scores")
plt.scatter(df_['PTAU'][df_.TurnedAD == 1].values, df_['ABETA'][df_.TurnedAD == 1].values, s=100, facecolors='none', alpha=0.7, edgecolors='green', label="Turned AD")
plt.legend()
plt.xlabel('PTAU')
plt.ylabel('ABETA')
plt.tight_layout()
plt.savefig('_figures/PTau_amyloid_AD.png')
plt.close('all')

# just CN and LOW PTAU
f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(df_['PTAU'][df_.cluster == 'CN'].values, df_['ABETA'][df_.cluster == 'CN'].values, c='antiquewhite', alpha=0.8, s=100, label="controls")
plt.scatter(df_['PTAU'][df_.cluster == 'Low'].values, df_['ABETA'][df_.cluster == 'Low'].values, c='dimgray', s=100, alpha=0.6, label="low scores")
plt.scatter(df_['PTAU'][df_.TurnedAD == 1].values, df_['ABETA'][df_.TurnedAD == 1].values, s=100, facecolors='none', alpha=0.7, edgecolors='green', label="Turned AD")
plt.legend()
plt.xlabel('PTAU')
plt.ylabel('ABETA')
plt.tight_layout()
plt.savefig('_figures/PTau_amyloid_AD_CNLow.png')
plt.close('all')

# just CN and LOW TAU
f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(df_['TAU'][df_.cluster == 'CN'].values, df_['ABETA'][df_.cluster == 'CN'].values, c='antiquewhite', alpha=0.8, s=100, label="controls")
plt.scatter(df_['TAU'][df_.cluster == 'Low'].values, df_['ABETA'][df_.cluster == 'Low'].values, c='dimgray', s=100, alpha=0.6, label="low scores")
plt.scatter(df_['TAU'][df_.TurnedAD == 1].values, df_['ABETA'][df_.TurnedAD == 1].values, s=100, facecolors='none', alpha=0.7, edgecolors='green', label="Turned AD")
plt.legend()
plt.xlabel('TAU')
plt.ylabel('ABETA')
plt.tight_layout()
plt.savefig('_figures/Tau_amyloid_AD_CNLow.png')
plt.close('all')


###################################
# VISUALISATION GREY MATTER LEVEL #
###################################

np.random.seed(0)

# load required dataframe
# grey matter quantity per roi
df_gm = pd.read_pickle('_pickles/gm_cleaned_atlas_harvard_oxford')
# add cluster label
df = df.set_index('RID')
df_gm = df_gm.join(df['cluster'], on=df.index)
# extract grey matter per group for later analysis


# SCATTERPLOT
for roi in range(24, len(df_gm.columns), 24): # 96 rois + column 'cluster'
    df_ = df_gm[list(df_gm.columns[roi-24:roi]) + ['cluster']]
    melted_df = pd.melt(df_, id_vars='cluster')

    # plotting
    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(10, 6))
    sns.stripplot(x="value", y="variable", hue='cluster', data=melted_df,
                            hue_order=['CN', 'SMC', 'High', 'Middle', 'Low'],
              size=4, color=".3", palette="vlag")

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True, bottom=True)
    plt.title('rois nb {} to {}'.format(roi-24, roi), fontsize=16)
    plt.tight_layout()
    plt.savefig('_figures/variation_gm_{}-{}.png'.format(roi-24, roi))
    plt.close('all')

# PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
acp_coef = pca.fit(df_gm[df_gm.columns[:-1]].values)
explained_variance =  acp_coef.explained_variance_ratio_
coefs = acp_coef.components_

# print components
pd.DataFrame(columns=[['composante 1', 'composante 2']], index=df_gm.columns[:-1], data=coefs.T)
print("composante 1 |    composante 2")
for ind, col in enumerate(df_gm.columns[:-1]):
    print(np.round(coefs[0][ind], 4), ' | ', np.round(coefs[1][ind], 4), ' | ', col)

acp = pca.transform(df_gm[df_gm.columns[:-1]].values)
df_gm[['acp1', 'acp2']] = acp

acp_high = df_gm[['acp1', 'acp2']].loc[df_gm[df_gm.cluster == 'High'].index]
acp_middle = df_gm[['acp1', 'acp2']].loc[df_gm[df_gm.cluster == 'Middle'].index]
acp_low = df_gm[['acp1', 'acp2']].loc[df_gm[df_gm.cluster == 'Low'].index]
acp_cn = df_gm[['acp1', 'acp2']].loc[df_gm[df_gm.cluster == 'CN'].index]
acp_smc = df_gm[['acp1', 'acp2']].loc[df_gm[df_gm.cluster == 'SMC'].index]


f, axarr = plt.subplots(figsize=(12, 12))
plt.scatter(acp_low['acp1'].values, acp_low['acp2'].values, c='dimgray', s=150, label="low scores")
plt.scatter(acp_middle['acp1'].values, acp_middle['acp2'].values, c='darkgrey', s=150, label="middle scores")
plt.scatter(acp_high['acp1'].values, acp_high['acp2'].values, c='gainsboro', s=150, label="high scores")
plt.scatter(acp_smc['acp1'].values, acp_smc['acp2'].values, c='tan', alpha=0.3, s=120, label="smc")
plt.scatter(acp_cn['acp1'].values, acp_cn['acp2'].values, c='antiquewhite', alpha=0.3, s=120, label="controls")
plt.legend()
plt.xlabel('Composant 1')
plt.ylabel('Composant 2')
plt.title("ACP_greyMatter")
plt.tight_layout()
plt.savefig('_figures/ACP_greyMatter.png')
plt.close('all')


#################################################
# VISUALISATION PREDICTION WITH GREY MATTER  ####
#################################################

# violin plot of accuracies

df_visu_gm = pd.read_pickle('_createdDataframe/df_visu_gm')
df_visu_csf = pd.read_pickle('_createdDataframe/df_visu_csf')


sns.set(style="white", context="talk")
df_visu_gm = df_visu_gm[['High_vs_CN', 'Middle_vs_CN', 'Low_vs_CN']]
df_visu_csf = df_visu_csf[['High_vs_CN', 'Middle_vs_CN', 'Low_vs_CN']]
df_gm = pd.melt(df_visu_gm)
df_csf = pd.melt(df_visu_csf)
df_gm = df_gm.rename(columns={'variable': 'Diagnosis group', 'value': 'Model accuracies gm'})
df_csf = df_csf.rename(columns={'variable': 'Diagnosis group', 'value': 'Model accuracies csf'})
df_mix = pd.concat((df_csf, df_gm))
df_mix['base'] = 0
df_mix['base'][df_mix['Model accuracies csf'].isna()] = 'Grey matter based'
df_mix['base'][df_mix['Model accuracies gm'].isna()] = 'CSF based'
df_mix['Model accuracies'] = 0
df_mix['Model accuracies'][df_mix['base'] == 'Grey matter based'] = df_mix['Model accuracies gm'][df_mix['Model accuracies csf'].isna()]
df_mix['Model accuracies'][df_mix['base'] == 'CSF based'] = df_mix['Model accuracies csf'][df_mix['Model accuracies gm'].isna()]
ax = sns.violinplot(x="Diagnosis group", y="Model accuracies", hue="base", data=df_mix, palette="muted")
plt.axhline(y=0.5, color="grey", linestyle='--')
plt.xticks(fontsize=14, rotation=45)
# plt.title("Prediction accuracy of diagnosis group vs controls", fontsize=14)
plt.legend(loc="lower right", fontsize=14) #bbox_to_anchor = (1.01, 0.6), 
plt.xlabel('Model scores', fontsize=14)
plt.xlabel('Diagnosis group', fontsize=14)
plt.tight_layout()
plt.savefig('_figures/accuracies_plot_gm.png')
plt.show()
