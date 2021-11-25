import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from utils import plot_matrix
from subprocess import call
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.datasets import load_wine

class ModelRFR:
    """Class that deals with training a Random Forest starting from the relevant configurations
    to characterize the VAS index obtained during the preliminary clustering phase. """

    def __init__(self, seq_df_path, train_video_idx, test_video_idx, preliminary_clustering, weighted_samples=False, verbose=False):
        self.seq_df_path = seq_df_path  # Path of csv file contained sequences info
        self.train_video_idx = train_video_idx  # Indexes of the videos to use for training
        self.test_video_idx = test_video_idx  # Indexes of the videos to use for test
        self.preliminary_clustering = preliminary_clustering  # preliminary clustering performed
        self.vas_sequences = None  # VAS of the dataset sequences
        self.means_gmm = self.preliminary_clustering.gmm.means  # center of the GMM clusters
        self.desc_relevant_config_videos = None  # descriptor of the sequences contained in the dataset
        self.model = None  # RFR model
        self.verbose = verbose # define if the output must be printed in the class
        self.weighted_samples = weighted_samples  # define if the samples must be weighted for the model fitting
        self.sample_weights = None  # sample weights (populated only if weighted_samples=True)

    def __generate_descriptors_relevant_configuration(self):
        """
        Generate a descriptor for each sequences contained in the dataset
        """

        if self.verbose:
            print("---- Generate descriptors of video sequences... ----")
        fisher_vectors = self.preliminary_clustering.fisher_vectors
        n_videos = len(fisher_vectors)
        num_relevant_config = len(self.preliminary_clustering.index_relevant_configurations)
        descriptors_of_videos = []
        idx_relevant_row = self.preliminary_clustering.index_relevant_configurations + \
                                  [config + len(self.means_gmm) for config in self.preliminary_clustering.index_relevant_configurations]
        for index in range(0, n_videos):
            current_video_fv = fisher_vectors[index][0]
            video_descriptor = np.zeros(shape=(num_relevant_config * 2, current_video_fv.shape[2]))
            for index_frame in range(0, current_video_fv.shape[0]):
                frame = current_video_fv[index_frame]
                video_descriptor[:] += frame[idx_relevant_row]
            if sum(video_descriptor).any() != 0:
                video_descriptor = np.sqrt(np.abs(video_descriptor)) * np.sign(video_descriptor)
                video_descriptor = video_descriptor / np.linalg.norm(video_descriptor, axis=(0, 1))[None, None]
            descriptors_of_videos.append(video_descriptor)
        return descriptors_of_videos

    """Read vas index of all sequences from dataset. 
    Return a list contained the vas index of all sequences """
    def __read_vas_videos(self):
        if self.verbose:
            print("---- Read vas indexes for sequences in dataset... ----")
        seq_df = pd.read_csv(self.seq_df_path)
        vas_sequences = []
        for num_video in np.arange(len(self.desc_relevant_config_videos)):
            vas_sequences.append(seq_df.iloc[num_video][1])
        return vas_sequences


    def __train(self):
        """
        Train RFT using fisher vectors calculated and vas indexes readed of the sequences.
        Return the trained classifier
        """

        training_set_histo = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.train_video_idx])
        training_set_vas = np.asarray([self.vas_sequences[i] for i in self.train_video_idx])
        model_rfr = RandomForestRegressor(n_estimators=3, max_depth=1)
        print("training_set_histo", len(training_set_histo[0]))
        print("training_set_vas", training_set_vas)
        return model_rfr.fit(training_set_histo, training_set_vas)

    def __print(self,model_rfr):
        for i in np.arange(len(model_rfr.estimators_)):
            estimator=model_rfr.estimators_[i]
            fig=plt.figure(figsize=(100, 100), dpi=80)
            plot_tree(estimator, 
                    filled=True, impurity=True, 
                    rounded=True)
            fig.savefig('tree' + str(i) + '.png')
            


    def train_RFR(self, model_dump_path=None, n_jobs=1):
            """
            Performs the model training procedure based on what was done in the preliminary clustering phase
            """

            self.desc_relevant_config_videos = self.__generate_descriptors_relevant_configuration()
            self.vas_sequences = self.__read_vas_videos()
            if self.weighted_samples:
                self.__calculate_sample_weights()
            # if train_by_max_score == True:
            #     self.model = self.__train_maximizing_score(n_jobs=n_jobs)
            self.model = self.__train()
            self.__print(self.model)
            # if model_dump_path is not None:
            #     self.__dump_on_pickle(model_dump_path)

    def __calculate_sample_weights(self):
        vas_occ = {}
        vas_weights = {}
        for vas in np.arange(0, 11):
            vas_occ[vas] = [self.vas_sequences[i] for i in self.train_video_idx].count(vas)
        sum_vas_occ = sum(vas_occ.values())
        for vas in np.arange(0, 11):
            if vas_occ[vas] > 0:
                vas_weights[vas] = sum_vas_occ / (11 * vas_occ[vas])
            else:
                vas_weights[vas] = 0
        self.sample_weights = np.ones(len(self.train_video_idx))
        for idx, video_idx in enumerate(self.train_video_idx):
            self.sample_weights[idx] = vas_weights[self.vas_sequences[video_idx]]


    def evaluate_performance(self, path_scores_parameters=None, path_scores_cm=None):
        test_set_desc = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.test_video_idx])
        test_set_vas = np.asarray([self.vas_sequences[i] for i in self.test_video_idx])
        num_test_videos = test_set_desc.shape[0]
        sum_error = 0
        confusion_matrix = np.zeros(shape=(11, 11))
        real__vas = []
        predicted__vas = []
        if path_scores_parameters is not None:
            out_df_scores = pd.DataFrame(columns=['sequence_num', 'real_vas', 'vas_predicted', 'error'])
        for num_video in np.arange(num_test_videos):
            real_vas = test_set_vas[num_video]
            real__vas.append(real_vas)
            vas_predicted = self.__predict(test_set_desc[num_video].reshape(1,-1))
            predicted__vas.append(vas_predicted)
            error = abs(real_vas-vas_predicted)
            sum_error += error
            if path_scores_parameters is not None:
                data = np.hstack(
                    (np.array([self.test_video_idx[num_video], real_vas, vas_predicted, error]).reshape(1, -1)))
                out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),
                                                     ignore_index=True)
            confusion_matrix[real_vas][vas_predicted] += 1
        if path_scores_parameters is not None:
            out_df_scores.to_csv(path_scores_parameters, index=False, header=True)
        if path_scores_cm is not None:
            plot_matrix(cm=confusion_matrix, labels=np.arange(0, 11), normalize=True, fname=path_scores_cm)
        mean_error = round(sum_error / num_test_videos, 3)
        return mean_error, confusion_matrix

    def __predict(self, sequence_descriptor):
        vas_predicted = self.model.predict(sequence_descriptor)[0]  
        if vas_predicted < 0:
            vas_predicted = 0
        elif vas_predicted > 10:
            vas_predicted = 10
        return int(round(vas_predicted, 0))


    def evaluate_performance_on_scaled_pain(self, path_scores_cm=None):
            test_set_desc = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.test_video_idx])
            test_set_vas = np.asarray([self.vas_sequences[i] for i in self.test_video_idx])
            num_test_videos = test_set_desc.shape[0]
            confusion_matrix = np.zeros(shape=(3, 3))
            dict_pain_level = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2}
            labels_cm = ["no pain", "weak pain", "severe pain"]
            for num_video in np.arange(num_test_videos):
                real_vas = test_set_vas[num_video]
                real_level_idx = dict_pain_level[real_vas]
                vas_predicted = self.__predict(test_set_desc[num_video].reshape(1,-1))
                predicted_level_idx = dict_pain_level[vas_predicted]
                confusion_matrix[real_level_idx][predicted_level_idx] += 1
            if path_scores_cm is not None:
                plot_matrix(cm=confusion_matrix, labels=labels_cm, normalize=True, fname=path_scores_cm)
            return confusion_matrix