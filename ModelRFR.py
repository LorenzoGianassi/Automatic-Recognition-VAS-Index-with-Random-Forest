from os import error
from tabnanny import verbose
import pandas as pd
import time
import pickle
import numpy as np
import random
from random import randrange
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from configuration import config
import pydot
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

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
        self.space = {
            "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400, 500, 600, 2000]),
            'max_depth': hp.choice('max_depth', range(1,20)),
            'min_samples_split' : hp.choice("min_samples_split", range(5,15)),
            'max_features': hp.choice('max_features', range(1,5)),
        }
        self.verbose = verbose # define if the output must be printed in the class
        self.weighted_samples = weighted_samples  # define if the samples must be weighted for the model fitting
        self.sample_weights = None  # sample weights (populated only if weighted_samples=True)
        self.random_forest_regressor = RandomForestRegressor(random_state = 42)
        


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
    

       #Define Objective Function
    def objective(self, params):
        """
        Return the Mean Absolute Error for Train Set
        """
        training_set_histo = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.train_video_idx])
        training_set_vas = np.asarray([self.vas_sequences[i] for i in self.train_video_idx])

        rf = RandomForestRegressor(**params)

        # fit Training model
        rf.fit(training_set_histo, training_set_vas)
        
        # Making predictions and find MAE
        y_pred = rf.predict(training_set_histo)
        mae = mean_absolute_error(training_set_vas,y_pred)
        
        # Return MAE
        return mae


    def _trials(self):
        """
        Return the best result over the Hyperparameter space
        """
        #Create Hyperparameter space
        trials = Trials()

        #Minimize a function over the Hyperparameter space
        best = fmin(self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=2,
            trials=trials)

        print("Best: ", best)
        print("Trials Result: ", trials.results)
    
    
    def train_RFR(self, model_dump_path=None, n_jobs=1, train_by_max_score=False, path_tree_fig=None, threshold=None):
        """
        Performs the model training procedure based on what was done in the preliminary clustering phase
        """
        np.seterr(divide='ignore', invalid='ignore')
        self.desc_relevant_config_videos = self.__generate_descriptors_relevant_configuration()
        self.vas_sequences = self.__read_vas_videos()
        if self.weighted_samples:
            self.__calculate_sample_weights()
        if train_by_max_score == True:
            self.model = self.__train_maximizing_score(n_jobs=n_jobs)
            if path_tree_fig != None:
                self.__print(path_tree_fig=path_tree_fig, threshold=threshold)
            self._trials()
        else:
            self.model = self.__train()
            if path_tree_fig != None:
                self.__print(path_tree_fig=path_tree_fig, threshold=threshold)
            if config.num_tree == True:
                self.plot_results(self.__grid_search(),path_tree_fig=path_tree_fig)
        if model_dump_path is not None:
            self.__dump_on_pickle(model_dump_path)


    def __train(self):
        """
        Train RFR using Random Forest Regressor calculated and vas indexes readed of the sequences.
        Return the trained regressor
        """

        training_set_histo = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.train_video_idx])
        training_set_vas = np.asarray([self.vas_sequences[i] for i in self.train_video_idx])
        model_rfr = RandomForestRegressor(n_estimators= 1, min_samples_split = 10, min_samples_leaf = 2, max_features = 'sqrt', max_depth = 110, criterion = 'squared_error', bootstrap = False)
        self.model = model_rfr.fit(training_set_histo, training_set_vas)
        return self.model


    def __train_maximizing_score(self, n_jobs):
        if self.verbose:
            print("---- Find parameters that minimizes mean absolute error... ----")
        training_set_desc = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.train_video_idx])
        training_set_vas = np.asarray([self.vas_sequences[i] for i in self.train_video_idx])


        # Grid creation
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 1, stop = 51, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Criterion
        criterion = ["squared_error", "absolute_error", "poisson"]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(20, 71, num = 2)]
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

        #Randomized search on hyper parameters
        random_forest_randomized = RandomizedSearchCV(estimator=self.random_forest_regressor, param_distributions=random_grid,
                                n_iter = 10, scoring='neg_mean_absolute_error', 
                                cv = None, verbose=1, random_state=None, n_jobs=-1,
                                return_train_score=True).fit(training_set_desc, training_set_vas, sample_weight=self.sample_weights)
        best_params = random_forest_randomized.best_params_
        print("--- Best Params ---\n", best_params)
        
        return RandomForestRegressor(n_estimators= best_params['n_estimators'], min_samples_split = best_params['min_samples_split'], min_samples_leaf = best_params['min_samples_leaf'],
                                        max_features = best_params['max_features'], max_depth = best_params['max_depth'], criterion = best_params['criterion'], bootstrap = best_params['bootstrap'])\
                                            .fit(training_set_desc, training_set_vas, sample_weight=self.sample_weights)
        
    def __grid_search(self):
        if self.verbose:
            print("---- Perform Grid Search from the result of random search... ----")
        
        training_set_desc = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.train_video_idx])
        training_set_vas = np.asarray([self.vas_sequences[i] for i in self.train_video_idx])

        # Create the parameter grid based on the results of random search 
        param_grid = {
            'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]
        }

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = self.random_forest_regressor, param_grid = param_grid, 
                                cv = 5, n_jobs = -1, verbose = 1, return_train_score=True)

        grid_search.fit(training_set_desc, training_set_vas)

        final_model = grid_search.best_estimator_

        # Grid with only the number of trees changed
        tree_grid = {'n_estimators': [int(x) for x in np.linspace(1, 301, 30)]}

        # Create the grid search model and fit to the training data
        tree_grid_search = GridSearchCV(final_model, param_grid=tree_grid, verbose = 1, n_jobs=-1, cv = 5,
                                        scoring = 'neg_mean_absolute_error', return_train_score=True)
        tree_grid_search.fit(training_set_desc, training_set_vas)
        tree_grid_search.cv_results_
        return tree_grid_search


    def plot_results(self, model, path_tree_fig, param = 'n_estimators', name = 'Num Trees'):
        param_name = 'param_%s' % param

        # Extract information from the cross validation model
        train_scores = model.cv_results_['mean_train_score']
        test_scores = model.cv_results_['mean_test_score']
        train_time = model.cv_results_['mean_fit_time']
        param_values = list(model.cv_results_[param_name])
        
        # Plot the scores over the parameter
        plt.subplots(1, 2, figsize=(10, 6))
        plt.subplot(121)
        plt.plot(param_values, train_scores, 'bo-', label = 'train')
        plt.plot(param_values, test_scores, 'go-', label = 'test')
        plt.ylim(ymin = -10, ymax = 0)
        plt.legend()
        plt.xlabel(name)
        plt.ylabel('Neg Mean Absolute Error')
        plt.title('Score vs %s' % name)
        
        
        plt.subplot(122)
        plt.plot(param_values, train_time, 'ro-')
        plt.ylim(ymin = 0.0, ymax = 2.0)
        plt.xlabel(name)
        plt.ylabel('Train Time (sec)')
        plt.title('Training Time vs %s' % name)
        
        
        plt.tight_layout(pad = 4)

        plt.savefig(path_tree_fig +'numTree.png')
        plt.close()

      
    
    def __print(self, path_tree_fig, threshold):
        estimator=self.model.estimators_[0]
        figure=plt.figure(1,figsize=(100, 100), dpi=80)
        plt.clf()
        plot_tree(estimator, 
                filled=True, impurity=True, 
                rounded=True)
        figure.savefig(path_tree_fig + str(threshold) + " _tree " + '.png')
        plt.close
       


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
        train_set_desc = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.train_video_idx])
        train_set_vas = np.asarray([self.vas_sequences[i] for i in self.train_video_idx])
        num_train_videos = train_set_desc.shape[0]
        sum_test_error = 0
        sum_train_error = 0
        test_confusion_matrix = np.zeros(shape=(11, 11))
        train_confusion_matrix = np.zeros(shape=(11, 11))
        test_confusion_BioVid_matrix = np.zeros(shape=(5, 5))
        train_confusion_BioVid_matrix = np.zeros(shape=(5, 5))
        real__vas = []
        predicted__vas = []
        if path_scores_parameters is not None:
            out_df_scores = pd.DataFrame(columns=['sequence_num', 'real_vas', 'vas_predicted', 'error'])
        for num_video in np.arange(num_test_videos):
            real_vas = test_set_vas[num_video]
            real__vas.append(real_vas)
            vas_predicted = self.__predict(test_set_desc[num_video].reshape(1,-1))
            predicted__vas.append(vas_predicted)
            test_error = abs(real_vas-vas_predicted)
            sum_test_error += test_error
            if path_scores_parameters is not None:
                data = np.hstack(
                    (np.array([self.test_video_idx[num_video], real_vas, vas_predicted, test_error]).reshape(1, -1)))
                out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),
                                                     ignore_index=True)
            if config.type_of_database == "BioVid":
                test_confusion_BioVid_matrix[real_vas][vas_predicted] += 1
            else:
                test_confusion_matrix[real_vas][vas_predicted] += 1
        for num_video in np.arange(num_train_videos):
            real_vas = train_set_vas[num_video]
            real__vas.append(real_vas)
            vas_predicted = self.__predict(train_set_desc[num_video].reshape(1,-1))
            predicted__vas.append(vas_predicted)
            train_error = abs(real_vas-vas_predicted)
            sum_train_error += train_error
            if path_scores_parameters is not None:
                data = np.hstack(
                    (np.array([self.train_video_idx[num_video], real_vas, vas_predicted, train_error]).reshape(1, -1)))
                out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),
                                                     ignore_index=True)
            if config.type_of_database == "BioVid":
                train_confusion_BioVid_matrix[real_vas][vas_predicted] += 1
            else:
                train_confusion_matrix[real_vas][vas_predicted] += 1
        if path_scores_parameters is not None:
            out_df_scores.to_csv(path_scores_parameters, index=False, header=True)
        if path_scores_cm is not None:    
            if config.type_of_database == "BioVid":
                plot_matrix(cm=test_confusion_BioVid_matrix, labels=np.arange(0, 5), normalize=True, fname=path_scores_cm)
                plot_matrix(cm=train_confusion_BioVid_matrix, labels=np.arange(0, 5), normalize=True, fname=path_scores_cm)
            else:
                plot_matrix(cm=test_confusion_matrix, labels=np.arange(0, 11), normalize=True, fname=path_scores_cm)
                plot_matrix(cm=train_confusion_matrix, labels=np.arange(0, 11), normalize=True, fname=path_scores_cm)
        mean_test_error = round(sum_test_error / num_test_videos, 3)
        mean_train_error = round(sum_train_error / num_train_videos, 3)
        if config.type_of_database == "BioVid":
            return mean_test_error, mean_train_error, test_confusion_BioVid_matrix, train_confusion_BioVid_matrix
        else:
            return mean_test_error, mean_train_error, test_confusion_matrix, train_confusion_matrix


    def __predict(self, sequence_descriptor):
        vas_predicted = self.model.predict(sequence_descriptor)[0]  
        if vas_predicted < 0:
            vas_predicted = 0
        elif vas_predicted > 10:
            vas_predicted = 10
        return int(round(vas_predicted, 0))


 # to test the overfitting of the model we test it with an increasing number of trees
    def evaluate_overfitting(self,number_of_trees=50):
        #default number of treees is 50
        test_mae = []
        train_mae = []
        for iter in range(number_of_trees):
            self.train_RFR()
            mean_test_error, mean_train_error, test_confusion_matrix, train_confusion_matrix = self.evaluate_performance()
            self.model.n_estimators +=1
            test_mae.append(mean_test_error)
            train_mae.append(mean_train_error)
        return test_mae, train_mae
    
        

    def evaluate_performance_on_scaled_pain(self, path_scores_cm=None):
            test_set_desc = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.test_video_idx])
            test_set_vas = np.asarray([self.vas_sequences[i] for i in self.test_video_idx])
            num_test_videos = test_set_desc.shape[0]
            confusion_matrix = np.zeros(shape=(3, 3))
            if config.type_of_database == "BioVid":
                dict_pain_level = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
            else:
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