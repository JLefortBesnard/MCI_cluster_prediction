############################################
## All functions needed for ADNI analysis ##
############################################

# data manipulation
import pandas as pd
import glob
import numpy as np
import psutil
import csv

# math
from scipy import stats #v1.5.2
from scipy.stats import percentileofscore
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

# neuroimaging
import nibabel as nib
from nilearn import datasets as ds
from nilearn.image import resample_img
from nilearn.input_data import NiftiLabelsMasker
from nilearn.signal import clean
from nilearn import plotting, image

# visualisation
from matplotlib import pylab as plt
import seaborn as sns


np.random.seed(0)


##########################
# extracting grey matter #
##########################


def niftiise_array(array, saving_path):
    # prepare nifti for saving coefs as .nii file
    MRI_path = 'C:\\Users\\lefortb211\\Downloads\\ADNI_CN_smwc\\smwc1ADNI_002_S_4213_MR_MPRAGE_br_raw_20110903090753647_129_S121168_I254582.nii'
    atlas = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',  symmetric_split=True)
    tmp_nii = nib.load(MRI_path)
    ratlas_nii = resample_img(
        atlas.maps, target_affine=tmp_nii.affine, interpolation='nearest')
    nii = nib.load(MRI_path)
    masker = NiftiLabelsMasker(labels_img=ratlas_nii, standardize=False, strategy='sum')
    cur_FS = masker.fit_transform(nii)
    # save array
    niftiised_array = masker.inverse_transform(np.array(array).reshape(1, 96))
    niftiised_array.to_filename(saving_path + ".nii") # transform as nii and save
    return "Done"
    


####################################################################
####### Function running the non-parametric hypothesis testing #####
####################################################################

# Non-parameric hypothesis test 
# run the CV 5 fold logistic regression with permutated Y
def hypothesis_testing(X, Y, final_coefficients, final_acc, rois_name):
    clf = LogisticRegression(penalty='l2', class_weight='balanced')
    n_permutations = 100
    perm_rs = np.random.RandomState(0)
    permutation_accs = []
    permutation_coefs = []
    print("running 100 permutations")
    for i_iter in range(n_permutations):
        Y_perm = perm_rs.permutation(Y)
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=i_iter)
        sss.get_n_splits(X, Y_perm)
        permutation_accs_cv = []
        permutation_coefs_cv = []
        for train_index, test_index in sss.split(X, Y_perm):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y_perm[train_index], Y_perm[test_index]
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            permutation_accs_cv.append(acc)
            permutation_coefs_cv.append(clf.coef_[0, :])
        permutation_accs.append(np.mean(permutation_accs_cv))
        permutation_coefs.append(np.mean(permutation_coefs_cv, axis=0))

    # extract permutation weigth per ROI for hypothesis testing
    weight_per_roi = []
    for roi_nb in range(len(rois_name)):
        weight_roi = []
        for coef_iter in permutation_coefs:
            weight_roi.append(coef_iter[roi_nb])
        weight_per_roi.append(weight_roi)

    # extract 95 and 5% percentile and check if original weight outside these limits to check for significance
    pvals = []
    for n_roi in range(len(rois_name)):
        ROI_weights = weight_per_roi[n_roi] 
        above = stats.scoreatpercentile(ROI_weights, 95)
        below = stats.scoreatpercentile(ROI_weights, 5)
        if final_coefficients[n_roi] < below or final_coefficients[n_roi] > above:
            pvals.append(1)
        else:
            pvals.append(0)
    pvals = np.array(pvals)
    print('{} ROIs are significant at p<0.05'.format(np.sum(pvals > 0)))

    # check if accuracy is significant
    above = stats.scoreatpercentile(permutation_accs, 95)
    percentile_of_acc = percentileofscore(permutation_accs, final_acc)
    if final_acc > above:
        print("accuracy ({}) is significant. Percentile: ".format(np.round(final_acc, 2)), percentile_of_acc)
        significance_acc = True
    else:
        print("accuracy ({}) is NOT significant. Percentile: ".format(np.round(final_acc, 2)), percentile_of_acc)
        significance_acc = False

    # get significant ROI name + significant weights
    significant_weights = []
    dic = {}
    for i, roi in enumerate(rois_name):
        if pvals[i] == 1:
            significant_weights.append(final_coefficients[i])
            dic[roi] = final_coefficients[i]
        else:
            significant_weights.append(0)
    return dic, significant_weights, significance_acc, percentile_of_acc

##########################################################
####### Function to fit a regression model 100 times #####
##########################################################

def run_regression_100times(X, Y):
    # keep information for the confusion matrix
    all_y_pred = []
    all_y_true = []

    # save accuracies and model coefficients
    accs = []
    coefs = []
    print("Running 100 subsamples")
    # run 100 times to ensure stability
    for i_subsample in range(100):
        folder = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=i_subsample)
        folder.get_n_splits(X, Y)
        sample_acc = []
        sample_coefs = []
        for train_inds, test_inds in folder.split(X, Y):
            # define algorithm
            clf = LogisticRegression(penalty='l2', class_weight='balanced')
            # clf_l1 = LogisticRegression(penalty='l1', solver='liblinear')
            # fit algorithm on training data
            clf.fit(X[train_inds], Y[train_inds])
            # compute accuracies on testing data
            acc = clf.score(X[test_inds], Y[test_inds])
            y_pred = clf.predict(X[test_inds])
            # save accuracies
            sample_acc.append(acc)
            # save coefficients
            sample_coefs.append(clf.coef_[0, :]) # squeeze
            # extract results for the confusion matrix
            all_y_pred.append(y_pred)
            all_y_true.append(Y[test_inds])
        accs.append(np.mean(sample_acc))
        coefs.append(np.mean(sample_coefs, axis=0))
    return coefs, accs, all_y_pred, all_y_true


#######################################################
####### Function to run the script + save results #####
#######################################################

def run_and_save(X, Y, saving_path, rois_name):

    # run the regression
    coefs, accs, all_y_pred, all_y_true = run_regression_100times(X, Y)
    np.save(saving_path[:-12] + "_confusion_matrix.npy", [all_y_pred, all_y_true])
    np.save(saving_path[:-12] + "all_coefs.npy", coefs)
    np.save(saving_path[:-12] + "all_accs.npy", accs)
    averaged_coefs = np.mean(coefs, axis=0)
    averaged_acc = np.mean(accs)
    acc_std = np.std(accs)
    print("Accuracy : ", averaged_acc, "  +/- ", acc_std)

    dic, significant_weights, significance_acc, percentile_of_acc = hypothesis_testing(X, Y, np.squeeze(averaged_coefs), averaged_acc, rois_name)

    dic["accuracy"] = [averaged_acc, acc_std, significance_acc, percentile_of_acc]
    
    ##############
    # saving significant and averaged weights as nifti image
    ##############
    # prepare nifti for saving coefs as .nii file
    MRI_path = 'C:\\Users\\lefortb211\\Downloads\\ADNI_CN_smwc\\smwc1ADNI_002_S_4213_MR_MPRAGE_br_raw_20110903090753647_129_S121168_I254582.nii'
    atlas = ds.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',  symmetric_split=True)
    tmp_nii = nib.load(MRI_path)
    ratlas_nii = resample_img(
        atlas.maps, target_affine=tmp_nii.affine, interpolation='nearest')
    nii = nib.load(MRI_path)
    masker = NiftiLabelsMasker(labels_img=ratlas_nii, standardize=False, strategy='sum')
    cur_FS = masker.fit_transform(nii)
    # save significant weigths
    significant_coefficients_nii3D = masker.inverse_transform(np.array(significant_weights).reshape(1, 96))
    significant_coefficients_nii3D.to_filename(saving_path + ".nii") # transform as nii and save
    # save averaged weigths
    averaged_coefficients_nii3D = masker.inverse_transform(averaged_coefs.reshape(1, 96))
    averaged_coefficients_nii3D.to_filename(saving_path[:-12] + "averaged_weights.nii") # transform as nii and save

    ##############
    # saving significant weights as csv file
    ##############
    # open file for writing, "w" is writing
    w = csv.writer(open(saving_path + ".csv", "w"))
    # loop over dictionary keys and values
    for key, val in dic.items():
        # write every key and value to file
        w.writerow([key, val])

    ##############
    # saving averaged weights as csv file
    ##############
    ave_weights = {}
    for ind, weight in enumerate(averaged_coefs):
        ave_weights[list(rois_name)[ind]] = weight
    # open file for writing, "w" is writing
    w = csv.writer(open(saving_path[:-12] + "averaged_weights.csv", "w"))
    # loop over dictionary keys and values
    for key, val in ave_weights.items():
        # write every key and value to file
        w.writerow([key, val])
    return dic


#####################################
####### Visualisation functions #####
#####################################

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



