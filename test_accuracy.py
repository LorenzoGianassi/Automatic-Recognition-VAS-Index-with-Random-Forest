import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import configuration.config
from PreliminaryClustering import PreliminaryClustering
from ModelRFR import ModelRFR
from configuration import config
from utils import get_training_and_test_idx, check_existing_paths, plot_matrix, save_data_on_csv, save_GMM_mean_info


def execute_train(selected_lndks_idx, n_kernels_GMM, threshold_VAS, description, vel_frame_window, vel_frame_threshold,
                  test_num, saving_folder):
    # Type of execution info
    fit_by_bic = config.fit_by_bic
    
    # Dataset info
    coord_df_path = "data/dataset/2d_skeletal_data_unbc_coords.csv"
    seq_df_path = "data/dataset/2d_skeletal_data_unbc_sequence.csv"
    num_lndks = 66
    weighted_samples = config.weighted_samples
    # Features info
    num_videos = 200
    cross_val_protocol = config.cross_val_protocol
    train_video_idx, test_video_idx = get_training_and_test_idx(num_videos, cross_val_protocol, seq_df_path)
    index = pd.read_csv(seq_df_path).query('VAS>=' + str(threshold_VAS)).index.tolist()
    idxs = []
    for idx in train_video_idx:
        video_idx = []
        for i in idx:
            if i in index:
                video_idx.append(i)
        idxs.append(video_idx)
    clustering_train_video_idx = idxs
    # Preliminary clustering info and paths
    threshold_neutral = config.threshold_neutral
    if fit_by_bic:
        assert isinstance(n_kernels_GMM, list) and isinstance(threshold_neutral, list) and min(n_kernels_GMM) > 0 \
               and min(threshold_neutral) > 0 and max(threshold_neutral) < 1 and len(n_kernels_GMM) == len(
            threshold_neutral)
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
    path = []
    path.append(saving_folder + "/eyes/")
    path.append(saving_folder + "/mouth/")
    path.append(saving_folder + "/eyes_mouth/")
    path.append(saving_folder + "/standard/")
    path.append(saving_folder + "/kernel_16/")
    path.append(saving_folder + "/kernel_64/")
    path.append(saving_folder + "/kernel_128/")
    path.append(saving_folder + "/VAS_6/")
    path.append(saving_folder + "/VAS_7/")
    path.append(saving_folder + "/VAS_8/")
    path.append(saving_folder + "/VAS_9/")
    path.append(saving_folder + "/position/")
    path.append(saving_folder + "/vel_pos/")
    path.append(saving_folder + "/32_frames_max/")
    path.append(saving_folder + "/64_frames_max/")
    path.append(saving_folder + "/best_config/")
    current_test_path = path[test_num]

    path_errors = current_test_path + "errors_tests/"
    path_gmm_means = current_test_path + "gmm_means/"
    path_confusion_matrices = current_test_path + "confusion_matrices/"
    path_results_csv = current_test_path + "results.csv"
    path_conf_matrix_csv = current_test_path + "confusion_matrix.csv"
    n_jobs = config.n_jobs

    if __name__ == '__main__':
        dir_paths = [current_test_path, path_errors, path_confusion_matrices, path_gmm_means]
        
        if save_histo_figures:
            dir_paths.append(path_histo_figures)
        file_paths = [coord_df_path, seq_df_path]
        check_existing_paths(dir_paths=dir_paths, file_paths=file_paths)
        out_df_scores = pd.DataFrame(
            columns=['#round', '#clusters', 'threshold', '#relevant_clusters', 'Mean Absolute Error'])
        n_test = len(train_video_idx)
        errors = []
        confusion_matrix = np.zeros(shape=(11, 11))
        confusion_matrix_pain_levels = np.zeros(shape=(3, 3))
        if fit_by_bic:
            print("Generate and test models with fitting GMM by BIC using " + str(
                n_kernels_GMM) + " kernels, " + covariance_type + " covariance and " + cross_val_protocol)
        else:
            print("Generate and test models with " + str(
                n_kernels_GMM) + " kernels GMM, " + covariance_type + " covariance, threshold = " + str(
                threshold_neutral) + " and using " + cross_val_protocol)            
        for test_idx in np.arange(0, n_test):
            print("- Round " + str(test_idx + 1) + "/" + str(n_test) + " -")
            test_videos = test_video_idx[test_idx]
            train_videos = train_video_idx[test_idx]
            clustering_train_videos = clustering_train_video_idx[test_idx]
            path_histo_current = None
            if fit_by_bic:
                print("-- Execute preliminary clustering fitting GMM by BIC... --")
            else:
                print("-- Execute preliminary clustering using " + str(n_kernels_GMM) + " kernels GMM... --")
                preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                               seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                               selected_lndks_idx=selected_lndks_idx,
                                                               train_video_idx=train_videos,
                                                               n_kernels=n_kernels_GMM,
                                                               covariance_type=covariance_type,
                                                               verbose=False,
                                                               threshold_neutral=threshold_neutral,
                                                               fit_by_bic=fit_by_bic)
            if save_histo_figures == True:
                path_histo_current = path_histo_figures + "test_" + str(test_idx) + "_"
            preliminary_clustering.execute_preliminary_clustering(histo_figures_path=path_histo_current)
            n_kernels_current_GMM = preliminary_clustering.n_kernels
            threshold_current_clustering = preliminary_clustering.threshold_neutral
            num_relevant_config = len(preliminary_clustering.index_relevant_configurations)
            if num_relevant_config == 0:
                print(
                    "-- No relevant configurations were found using " + str(
                        n_kernels_current_GMM) + " kernels and " + str(
                        threshold_current_clustering) + " for the threshold of neutral configurations "
                                                        "(try to lower the threshold by analyzing the histograms produced by clustering in the test module )--")
                current_error = current_accuracy = "None"
            else:
                print("-- Preliminary clustering ended: " + str(num_relevant_config) + " relevant clusters founded --")
                model_rfr = ModelRFR(seq_df_path=seq_df_path,
                                     train_video_idx=train_videos,
                                     test_video_idx=test_videos,
                                     preliminary_clustering=preliminary_clustering,
                                     weighted_samples=weighted_samples)
                print("-- Train and save SVR model... --")
                model_rfr.train_RFR(train_by_max_score=True, n_jobs=n_jobs)
                print("-- Calculate scores for trained SVR... --")
                current_test_path_error = path_errors + "errors_test_" + str(test_idx) + ".csv"
                current_path_cm = path_confusion_matrices + "conf_matrix_test_" + str(test_idx) + ".png"
                current_error, current_confusion_matrix = model_rfr.evaluate_performance(
                    path_scores_parameters=current_test_path_error,
                    path_scores_cm=current_path_cm)
                current_cm_pain_level = model_rfr.evaluate_performance_on_scaled_pain()
                errors.append(current_error)
                print("-- Mean Absolute Error: " + str(current_error) + " --")
                confusion_matrix += current_confusion_matrix
                confusion_matrix_pain_levels += current_cm_pain_level

            out_df_scores = save_data_on_csv(
                [test_idx + 1, n_kernels_current_GMM, threshold_current_clustering, num_relevant_config, current_error],
                out_df_scores, path_results_csv)
            current_path_gmm_means_csv = path_gmm_means + "gmm_means_test_" + str(test_idx) + ".csv"
            current_path_clusters_png = path_gmm_means + "gmm_clusters_test_" + str(test_idx) + ".png"
            save_GMM_mean_info(preliminary_clustering.gmm.means, selected_lndks_idx, current_path_gmm_means_csv,
                               current_path_clusters_png)

        mean_error = sum(errors) / n_test
        mean_error = round(mean_error, 3)

        print("Mean Absolute Error: " + str(mean_error))

        path_errors = current_test_path + "graphics_errors.png"
        path_conf_matrix = current_test_path + "confusion_matrix.png"
        path_conf_matrix_pain_levels = current_test_path + "confusion_matrix_pain_levels.png"
        print("Mean absolute errors detected at each round saved in a csv file on path '" + path_results_csv + "'")
        print("Confusion matrices detected at each round saved in png files on path '" + path_confusion_matrices + "'")

        plot_matrix(cm=confusion_matrix, labels=np.arange(0, 11), normalize=True, fname=path_conf_matrix)
        print("Overall confusion matrix saved in png files on path '" + path_conf_matrix + "'")
        labels_cm = ["no pain", "weak pain", "severe pain"]
        plot_matrix(cm=confusion_matrix_pain_levels, labels=labels_cm, normalize=True,
                    fname=path_conf_matrix_pain_levels)
        print(
            "Overall confusion matrix on pain level saved in png files on path '" + path_conf_matrix_pain_levels + "'")

        plt.bar(np.arange(1, n_test + 1), errors, color="blue")
        plt.axhline(y=mean_error, xmin=0, xmax=n_test + 1, color="red", label='Mean Absolute Error: ' + str(mean_error))
        plt.ylabel("Average of the Mean Absolute Error")
        plt.xlabel("Num round")
        plt.title("Mean Absolute Errors")
        plt.legend()
        plt.savefig(path_errors)
        plt.close()
        print("Histogram of the mean absolute error detected saved in a png file on path '" + current_test_path + "'")


def test_accuracy():
    """
    - Test with landmarks: 1) eyes, 2) mouth, 3) eyes + mouth, 4) all, standard
    - Test with different number of kernel in clusters: 16, 32, 64, 128
    - Getting clusters: 1) with all the sequences of train, 2) only with sequences with VAS greater than K = 6, 7, 8, 9
    - Landmark's description with: 1) only position, 2) only velocity, 3) position and velocity
    - Sequence's description with: 1) all the available frames, 2) a window of N frmaes centered in the frame with max dynamic
    """
    saving_folder = 'test_accuracy'
    
    test_num = 0
    print("Test with landmarks: 1) eyes, 2) mouth, 3) eyes + mouth, 4) all, standard")
    n_kernels_GMM = 32
    threshold_VAS = 0
    description = 'vel'
    vel_frame_window = 0
    vel_frame_threshold = 0

    # Eyes, Mouth, Eyes+Mouth, Standard
    selected_lndks_idx = [[30, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                          [30, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
                          [30, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                           58, 59, 60, 61, 62, 63, 64, 65],
                          [5, 11, 19, 24, 30, 37, 41, 44, 46, 50, 52, 56, 58]]
    for selected_lndks in selected_lndks_idx:
        execute_train(selected_lndks, n_kernels_GMM, threshold_VAS, description, vel_frame_window, vel_frame_threshold,
                      test_num, saving_folder)
        test_num += 1

    print("Test with different number of kernel in clusters: 16, 32, 64, 128")
    selected_lndks_idx = [5, 11, 19, 24, 30, 37, 41, 44, 46, 50, 52, 56, 58]
    n_kernels_GMM = [16, 64, 128]
    for kernel in n_kernels_GMM:
        execute_train(selected_lndks_idx, kernel, threshold_VAS, description, vel_frame_window, vel_frame_threshold,
                      test_num, saving_folder)
        test_num += 1

    print(
        "Getting clusters: 1) with all the sequences of train, 2) only with sequences with VAS greater than K = 6, 7, 8, 9")
    n_kernels_GMM = 32
    threshold_VAS = [6, 7, 8, 9]
    for threshold in threshold_VAS:
        execute_train(selected_lndks_idx, n_kernels_GMM, threshold, description, vel_frame_window, vel_frame_threshold,
                      test_num, saving_folder)
        test_num += 1

    threshold_VAS = 0
    print("Landmark's description with: 1) only position, 2) only velocity, 3) position and velocity")
    description = ['pos', 'pos+vel']
    for desc in description:
        execute_train(selected_lndks_idx, n_kernels_GMM, threshold_VAS, desc, vel_frame_window, vel_frame_threshold,
                      test_num, saving_folder)
        test_num += 1

    description = 'vel'
    print(
        "Sequence's description with: 1) all the available frames, 2) a window of N frmaes centered in the frame with max dynamic")
    vel_frame_window = [32, 64]
    for vel in vel_frame_window:
        execute_train(selected_lndks_idx, n_kernels_GMM, threshold_VAS, description, vel, vel_frame_threshold, test_num,
                      saving_folder)
        test_num += 1

    print("Test the best configuration")
    selected_lndks_idx = [5, 11, 19, 24, 30, 37, 41, 44, 46, 50, 52, 56, 58]
    execute_train(selected_lndks_idx, 32, 9, 'vel', 0, 6, test_num, saving_folder)


test_accuracy()
