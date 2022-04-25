from os import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from PreliminaryClustering import PreliminaryClustering
from ModelRFR import ModelRFR
from configuration import config
from utils import get_training_and_test_idx, check_existing_paths, plot_matrix, save_data_on_csv, save_GMM_mean_info, plot_all_graphs

"""Script that allows you to train an SVR using a given number of kernels for preliminary 
clustering."""

# Type of execution info
fit_by_bic = config.fit_by_bic

# Dataset info
if config.type_of_database == 'BioVid':
    coord_df_path ="data/dataset/BioVid_coords.csv"
    seq_df_path = "data/dataset/BioVid_sequence.csv"
    num_lndks = 67
    num_videos = 500
else:
    coord_df_path = "data/dataset/2d_skeletal_data_unbc_coords.csv"
    seq_df_path = "data/dataset/2d_skeletal_data_unbc_sequence.csv"
    num_lndks = 66
    num_videos = 200
weighted_samples = config.weighted_samples
# Features info
selected_lndks_idx = config.improved_selected_lndks_idx

cross_val_protocol = config.cross_val_protocol

train_video_idx, test_video_idx = get_training_and_test_idx(num_videos, cross_val_protocol, seq_df_path)


# Preliminary clustering info and paths
n_kernels_GMM = config.n_kernels_GMM
threshold_neutral = config.threshold_neutral
if fit_by_bic:
    assert isinstance(n_kernels_GMM, list) and isinstance(threshold_neutral, list) and min(n_kernels_GMM) > 0 \
           and min(threshold_neutral) > 0 and max(threshold_neutral) < 1 and len(n_kernels_GMM) == len(threshold_neutral)
    sub_directory = "fit_by_bic"
else:
    assert isinstance(n_kernels_GMM, int) and isinstance(threshold_neutral, float) and n_kernels_GMM > 0 and \
           0 < threshold_neutral < 1
    sub_directory = str(n_kernels_GMM) + "_kernels"
covariance_type = config.covariance_type
save_histo_figures = config.save_histo_figures
path_histo_figures = "data/classifier/" + sub_directory + "/histo_figures/"
preliminary_clustering_path = "data/classifier/" + sub_directory + "/preliminary_clustering.pickle"
# Model classifier info and paths
path_results = "data/classifier/" + sub_directory + "/"
path_errors = path_results + "errors_tests/"
path_gmm_means = path_results + "/gmm_means/"
path_confusion_matrices = path_results + "confusion_matrices/"
path_results_csv = path_results + "results.csv"
path_conf_matrix_csv = path_results + "confusion_matrix.csv"
n_jobs = config.n_jobs

if __name__ == '__main__':
    dir_paths = [path_results, path_errors, path_confusion_matrices, path_gmm_means]
    gc.collect()
    if save_histo_figures:
        dir_paths.append(path_histo_figures)
    file_paths = [coord_df_path, seq_df_path]
    check_existing_paths(dir_paths=dir_paths, file_paths=file_paths)
    out_df_scores = pd.DataFrame(columns=['Round', 'N kernels GMM', 'Threshold', 'Relevant Clusters', 'Mean Absolute Error'])
    n_test = len(train_video_idx)
    test_errors = []
    train_errors = []
    base_errors_list = []
    best_errors_list = []
    confusion_test_matrix = np.zeros(shape=(11, 11))
    confusion_train_matrix = np.zeros(shape=(11, 11))
    confusion_test_BioVid_matrix = np.zeros(shape=(5, 5))
    confusion_train_BioVid_matrix = np.zeros(shape=(5, 5))
    confusion_matrix_test_pain_levels = np.zeros(shape=(3, 3))
    confusion_matrix_train_pain_levels = np.zeros(shape=(3, 3))
    if fit_by_bic:
        print("Generate and test models with fitting GMM by BIC using "+str(n_kernels_GMM)+" kernels, "+covariance_type+" covariance and "+cross_val_protocol )
    else:
        print("Generate and test models with "+str(n_kernels_GMM)+" kernels GMM, "+covariance_type+" covariance, threshold = "+str(threshold_neutral)+ " and using "+cross_val_protocol )
    for test_idx in np.arange(0, n_test):
        print("- Round "+str(test_idx+1)+"/"+str(n_test)+" -")
        test_videos = test_video_idx[test_idx]
        train_videos = train_video_idx[test_idx]
        path_histo_current = None
        if fit_by_bic:
            print("-- Execute preliminary clustering fitting GMM by BIC... --")
        else:
            print("-- Execute preliminary clustering using " + str(n_kernels_GMM) + " kernels GMM... --")
        preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                       seq_df_path=seq_df_path,
                                                       num_lndks=num_lndks,
                                                       selected_lndks_idx=selected_lndks_idx,
                                                       train_video_idx=train_videos,
                                                       n_kernels=n_kernels_GMM,
                                                       covariance_type=covariance_type,
                                                       threshold_neutral=threshold_neutral,
                                                       fit_by_bic=fit_by_bic)
        if save_histo_figures == True:
            path_histo_current = path_histo_figures + "test_"+str(test_idx)+"_"
        preliminary_clustering.execute_preliminary_clustering(histo_figures_path=path_histo_current)
        n_kernels_current_GMM = preliminary_clustering.n_kernels
        threshold_current_clustering = preliminary_clustering.threshold_neutral
        num_relevant_config = len(preliminary_clustering.index_relevant_configurations)
        if num_relevant_config == 0:
            print("-- No relevant configurations were found using "+str(n_kernels_current_GMM)+" kernels and "+str(threshold_current_clustering)+" for the threshold of neutral configurations "
                  "(try to lower the threshold by analyzing the histograms produced by clustering in the test module )--")
            current_error = current_accuracy = "None"
        else:
            print("-- Preliminary clustering ended: "+str(num_relevant_config)+" relevant clusters founded --")
            model_rfr = ModelRFR(seq_df_path=seq_df_path,
                                 train_video_idx=train_videos,
                                 test_video_idx=test_videos,
                                 preliminary_clustering=preliminary_clustering,
                                 weighted_samples=weighted_samples)
            print("-- Train and save RFR model... --")
            model_rfr.train_RFR(n_jobs=n_jobs, path_tree_fig=path_results, threshold = threshold_neutral, train_by_max_score= False)
            print("-- Calculate scores for trained RFR... --")
            current_test_path_error = path_errors+"errors_test_"+str(test_idx)+".csv"
            current_path_cm = path_confusion_matrices + "conf_matrix_test_" + str(test_idx) + ".png"
            current_test_error, current_train_error, current_test_confusion_matrix, current_train_confusion_matrix = model_rfr.evaluate_performance(path_scores_parameters=current_test_path_error,path_scores_cm=current_path_cm)
            current_cm_pain_level = model_rfr.evaluate_performance_on_scaled_pain()
            test_errors.append(current_test_error)
            train_errors.append(current_train_error)
            print("-- Mean Absolute Test Error: " + str(current_test_error)+" --")
            if config.type_of_database == "BioVid":
                confusion_test_BioVid_matrix += current_test_confusion_matrix
                confusion_matrix_test_pain_levels += current_cm_pain_level
                print("-- Mean Absolute Train Error: " + str(current_train_error)+" --")
                confusion_train_BioVid_matrix += current_train_confusion_matrix
                confusion_matrix_train_pain_levels += current_cm_pain_level
            else:
                confusion_test_matrix += current_test_confusion_matrix
                confusion_matrix_test_pain_levels += current_cm_pain_level
                print("-- Mean Absolute Train Error: " + str(current_train_error)+" --")
                confusion_train_matrix += current_train_confusion_matrix
                confusion_matrix_train_pain_levels += current_cm_pain_level

        #out_df_scores = save_data_on_csv([test_idx+1, n_kernels_current_GMM, threshold_current_clustering, num_relevant_config, current_error],
        #                            out_df_scores, path_results_csv)
        current_path_gmm_means_csv = path_gmm_means + "gmm_means_test_" + str(test_idx) + ".csv"
        current_path_clusters_png = path_gmm_means + "gmm_clusters_test_" + str(test_idx) + ".png"
        save_GMM_mean_info(preliminary_clustering.gmm.means, selected_lndks_idx, current_path_gmm_means_csv, current_path_clusters_png)


    mean_test_error = sum(test_errors) / n_test
    mean__test_error = round(mean_test_error, 3)
    print("Total Mean Absolute Test Error: " + str(mean_test_error))

    mean_train_error = sum(train_errors) / n_test
    mean_train_error = round(mean_train_error, 3)
    print("Total Mean Absolute Train Error: " + str(mean_train_error))

    path_errors = path_results + "graphics_errors.png"
    path_conf_test_matrix = path_results + "confusion_test_matrix.png"
    path_conf_test_matrix_pain_levels = path_results + "confusion_test_matrix_pain_levels.png"
    path_conf_train_matrix = path_results + "confusion_train_matrix.png"
    path_conf_train_matrix_pain_levels = path_results + "confusion_train_matrix_pain_levels.png"
    print("Mean absolute errors detected at each round saved in a csv file on path '" + path_results_csv+"'")
    print("Confusion matrices detected at each round saved in png files on path '" + path_confusion_matrices+"'")

    if config.type_of_database == "BioVid":
        plot_matrix(cm=confusion_test_BioVid_matrix, labels=np.arange(0, 5), normalize=True, fname=path_conf_test_matrix)
        print("Overall confusion test matrix saved in png files on path '" + path_conf_test_matrix+"'")
        labels_cm = ["no pain", "weak pain", "severe pain"]
        plot_matrix(cm=confusion_matrix_test_pain_levels, labels=labels_cm, normalize=True, fname=path_conf_test_matrix_pain_levels)
        print("Overall confusion test matrix on pain level saved in png files on path '" + path_conf_test_matrix_pain_levels + "'")

        plot_matrix(cm=confusion_train_BioVid_matrix, labels=np.arange(0, 5), normalize=True, fname=path_conf_train_matrix)
        print("Overall confusion train matrix saved in png files on path '" + path_conf_train_matrix+"'")
        labels_cm = ["no pain", "weak pain", "severe pain"]
        plot_matrix(cm=confusion_matrix_train_pain_levels, labels=labels_cm, normalize=True, fname=path_conf_train_matrix_pain_levels)
        print("Overall confusion train matrix on pain level saved in png files on path '" + path_conf_train_matrix_pain_levels + "'")
    else:
        plot_matrix(cm=confusion_test_matrix, labels=np.arange(0, 11), normalize=True, fname=path_conf_test_matrix)
        print("Overall confusion test matrix saved in png files on path '" + path_conf_test_matrix+"'")
        labels_cm = ["no pain", "weak pain", "severe pain"]
        plot_matrix(cm=confusion_matrix_test_pain_levels, labels=labels_cm, normalize=True, fname=path_conf_test_matrix_pain_levels)
        print("Overall confusion test matrix on pain level saved in png files on path '" + path_conf_test_matrix_pain_levels + "'")

        plot_matrix(cm=confusion_train_matrix, labels=np.arange(0, 11), normalize=True, fname=path_conf_train_matrix)
        print("Overall confusion train matrix saved in png files on path '" + path_conf_train_matrix+"'")
        labels_cm = ["no pain", "weak pain", "severe pain"]
        plot_matrix(cm=confusion_matrix_train_pain_levels, labels=labels_cm, normalize=True, fname=path_conf_train_matrix_pain_levels)
        print("Overall confusion train matrix on pain level saved in png files on path '" + path_conf_train_matrix_pain_levels + "'")


    """
    plt.clf()
    plt.bar(np.arange(1, n_test+1), errors, color="blue")
    plt.axhline(y=mean_error, xmin=0, xmax=n_test+1, color="red", label='Mean Absolute Error: '+str(mean_error))
    plt.ylabel("Average of the Mean Absolute Error",  fontsize=15)
    plt.xlabel("Num round", fontsize=15)
    plt.title("Mean Absolute Errors", fontsize=15)
    plt.legend()
    plt.savefig(path_errors)
    plt.close()
    print("Histogram of the mean absolute error detected saved in a png file on path '" + path_results+"'")
    """

    model_rfr_overfit = ModelRFR(seq_df_path=seq_df_path,
                                 train_video_idx=train_videos,
                                 test_video_idx=test_videos,
                                 preliminary_clustering=preliminary_clustering,
                                 weighted_samples=weighted_samples)
    test_mae, train_mae = model_rfr_overfit.evaluate_overfitting(number_of_trees=100)
    overfit_test_mae = []
    overfit_test_mae.append(test_mae)
    overfit_test_mae.append(train_mae)
    plot_all_graphs(x=[i for i in np.arange(0,100)],
                    y=[error for error in overfit_test_mae],
                    x_label="Number of Tree", y_label= "Mean Absolute Error",
                    name_labels=["test", "train"],    
                    title="Mean Absolute Errors Overfit with "+str(n_kernels_GMM)+" kernels",
                    file_path= "errors_test_train_overfit_graph.png")

    