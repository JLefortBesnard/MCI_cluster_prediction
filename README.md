### Code for reproducing the key results of our paper: <br>_Grey matter volume and CSF biomarkers predict neuropsychological subtypes of MCI_

Lefort-Besnard J, Naveau M, Delcroix N, Decker L, Cignetti F (for the ADNI)

#### fit_atlas.py: 
Extract grey matter volume per ROI using Yeo atlas.

#### clustering.py:
Run clustering algorithm (n=3 according to R package nbclust) using MCI neuropsychological scores

#### regression_analyses.py:
Compute regression analyses for each MCI subgroup versus controls using GM or CSF level

#### Benchmark.py: 
Probing complex relationships among the grey matter rois

#### multinomial_analyses.py
Compute multiclass (One versus Rest) regression analyses for MCI subgroups using GM or CSF level

---

Please cite this paper when using the code for your research.

You can clone this repository by:

git clone https://github.com/JLefortBesnard/MCI_cluster_prediction.git
