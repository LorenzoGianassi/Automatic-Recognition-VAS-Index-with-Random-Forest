import numpy as np
""" Configuration file """

# PARAMETERS USED BY ALL SCRIPTS

# Define if the GMM must be fitted minimized BIC from more than one kernels number
fit_by_bic = False
# Number of clusters of the GMM (if fit_by_bic = True set a list of number of kernels, otherwise set an integer value)
n_kernels_GMM = 16

# Covariance type to use for the GMM fitting
covariance_type = "full"
"""covariance_type: 'diag' or 'full' """

# Indexes of the landmarks to use for fitting GMM and description sequences

type_of_landmark = "improved"
"""type_of_landmark:  'improved' or 'original' or 'eyes_and_eyebrows' or 'eyes' or 'nose' or 'mouth' or 'all'"""

improved_selected_lndks_idx = [5, 11, 19, 24, 30, 37, 41, 44, 46, 50, 52, 56, 58]
#selected_lndks_idx = np.arange(66)

# Original Arezzo, senza 30
original_selected_lndks_idx = [5, 11, 19, 24, 30, 37, 41, 44, 46, 50, 52, 56, 58]

# Occhi+sopracciglia
eyes_and_eyebrows_selected_lndks_idx = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Occhi
eyes_selected_lndks_idx = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Naso
nose_selected_lndks_idx = [27, 28, 29, 30, 31, 32, 33, 34, 35]

# Bocca
mouth_selected_lndks_idx = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]

# Occhi sopracciglia naso bocca
all_selected_lndks_idx = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                      41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                      65]

n_jobs = 8  # Number of threads to use to perform SVR training

# Type of protocol to be used to evaluate the performance of the models
cross_val_protocol = "5-fold-cross-validation"
"""cross_val_protocol:  'Leave-One-Subject-Out' or '5-fold-cross-validation' or 'Leave-One-Sequence-Out'"""


# PARAMETERS USED BY THE SCRIPT generate_model_predictor.py

# Threshold of the neutral configurations (if fit_by_bic = True set a list of thresholds of the same length defined
# in the n_kernels_GMM list, otherwise set a float value between 0 and 1)
threshold_neutral = 0.25
""" For example if the threshold is 0.3: all those configurations that occur within the sequences with vas equal to 0 
 with a frequency greater than 0.3 will be considered neutral. """

# Defines if the histograms of the dataset sequences must be saved in their respective files
save_histo_figures = False
"""If save_histo_figures = True, the histograms are saved in the project folder
 'data/classifier/n_kernels/figures/histograms/' with n=number of kernels of GMM
 (make sure that this file exists)"""

# Defines if the samples must be weighted for training
weighted_samples = True

# Thresholds to test
thresholds_neutral_to_test = np.arange(0.1, 0.5, 0.05)