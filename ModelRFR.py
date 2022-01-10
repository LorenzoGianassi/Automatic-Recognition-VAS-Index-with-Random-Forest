from os import error
import pandas as pd
import time
import pickle
import numpy as np
from pprint import pprint
from scipy.sparse.construct import random
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as sm
from sklearn.tree import export_graphviz
from utils import plot_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from configuration import config
import pydot

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


    def __read_vas_videos(self):
        """
        Read vas index of all sequences from dataset. 
        Return a list contained the vas index of all sequences 
        """
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
        model_rfr = RandomForestRegressor(n_estimators= 1100, min_samples_split = 10, min_samples_leaf = 2, max_features = 'sqrt', max_depth = 110, criterion = 'squared_error', bootstrap = False)
        self.model = model_rfr.fit(training_set_histo, training_set_vas)
        return model_rfr.fit(training_set_histo, training_set_vas)


    def __train_randomized(self):
        """
        Train RFT using fisher vectors calculated and vas indexes readed of the sequences.
        Return the trained classifier
        """
        training_set_histo = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.train_video_idx])
        training_set_vas = np.asarray([self.vas_sequences[i] for i in self.train_video_idx])
        random_forest = RandomForestRegressor(random_state = 42)

        # Grid creation
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Criterion
        criterion = ["squared_error", "absolute_error", "poisson"]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'criterion': criterion,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        random_forest_randomized = RandomizedSearchCV(estimator=random_forest, param_distributions=random_grid,
                              n_iter = 20, scoring='neg_mean_absolute_error', 
                              cv = 2, verbose=1, random_state=42, n_jobs=-1,
                              return_train_score=True)
        

        random_forest_randomized.fit(training_set_histo, training_set_vas)
        print("best param", random_forest_randomized.best_params_)
        #print("cv", random_forest_randomized.cv_results_)
        self.random_model = random_forest_randomized


    def __print(self,model_rfr):
        for i in np.arange(len(model_rfr.estimators_)):
            estimator=model_rfr.estimators_[i]
            figure=plt.figure(figsize=(100, 100), dpi=80)
            plt.clf()
            plot_tree(estimator, 
                    filled=True, impurity=True, 
                    rounded=True)
            figure.savefig('tree' + str(i) + '.png')


    def train_RFR(self, model_dump_path=None, n_jobs=1):
        """
        Performs the model training procedure based on what was done in the preliminary clustering phase
        """
        np.seterr(divide='ignore', invalid='ignore')
        self.desc_relevant_config_videos = self.__generate_descriptors_relevant_configuration()
        self.vas_sequences = self.__read_vas_videos()
        if self.weighted_samples:
            self.__calculate_sample_weights()
        # if train_by_max_score == True:
        #     self.model = self.__train_maximizing_score(n_jobs=n_jobs)
        if config.train_type == "normal_train":
            start_time = time.time()
            self.__train()
            print("--- RRF time: %s seconds ---" % (time.time() - start_time))
        elif config.train_type == "randomized_train":
            self.__train_randomized()
            self.compare_random()
        elif config.train_type == "grid_train":
            self.__train_grid()
            self.compare_random()
        #self.__print(self.model)
        # if model_dump_path is not None:
        #     self.__dump_on_pickle(model_dump_path)
    
    

    
    def evaluate_random(self, model, test_features, test_labels):

        predictions = model.predict(test_features)
        #print("prediction", predictions)
        errors = abs(predictions - test_labels)
        #print("test label", test_labels)
        mape = 100 * np.mean((errors / test_labels))
        np.seterr(divide='ignore', invalid='ignore')
        mean_errors = np.mean(errors)
        print("Explain variance score =", round(sm.explained_variance_score(test_labels, predictions), 2)) 
        print("R2 score =", round(sm.r2_score(test_labels, predictions), 2))
        print('Average Error: {:0.4f}.'.format(mean_errors))
        print("MAE:", mean_absolute_error(test_labels,predictions))
        print("MSQE:", mean_squared_error(test_labels,predictions))
        
        return mean_errors
      

    def mean_absolute_scaled_error(self,y_true, y_pred, y_train):
        e_t = y_true - y_pred
        scale = mean_absolute_error(y_train[1:], y_train[:-1])
        return np.mean(np.abs(e_t / scale))

    def compare_random(self):
        base_model = RandomForestRegressor(random_state = 42)
        test_set_desc = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.test_video_idx])
        test_set_vas = np.asarray([self.vas_sequences[i] for i in self.test_video_idx])
        train_set_desc = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.train_video_idx])
        train_set_vas = np.asarray([self.vas_sequences[i] for  i in self.train_video_idx])
        base_model.fit(train_set_desc, train_set_vas)
        
        print("--Base model Random--")
        base_error = self.evaluate_random(base_model, test_set_desc, test_set_vas)
        print("Fold error", base_error)

        best_random = self.random_model.best_estimator_
        print("--Best model Random--")
        best_error = self.evaluate_random(best_random, test_set_desc, test_set_vas)
        print("Fold error Random", best_error)

        #print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
        
        return base_error, best_error
    

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
        print("Mean Error: ", mean_error)
        return mean_error, confusion_matrix

    def __predict(self, sequence_descriptor):
        vas_predicted = self.model.predict(sequence_descriptor)[0]  
        #vas_predicted = self.random_model.predict(sequence_descriptor)[0]  
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