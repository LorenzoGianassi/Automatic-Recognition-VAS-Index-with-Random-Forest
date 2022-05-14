# Automatic-Recognition-VAS-Index-with-Random-Forest
## Table of Contents  
- [About the Project](#1)  
  - [Built with](#2)
- [Datasets](#3)
- [Implementation](#4)

# About the Project <a name="1"/>
![](Images/framework_scheme.jpg) <br/>
Implementation of the Automatic Recognition with VAS Index (pain index) with the aim of demonstrating the effectiveness of the Random Forest on the problem. <br/>
This project has the task of extending and trying to improve the results obtained by our colleague [Alessandro Arezzo](https://github.com/AlessandroArezzo) in his [work](https://github.com/AlessandroArezzo/Automatic-Recognition-VAS-Index), using a different supervised learning model (Random Forest Regressor).

# Built with <a name="2"/>
- [Python](https://www.python.org/)
- [Scikit-learn](https://scikit-learn.org/stable/): It's a simple and efficient tools for predictive data analysis.

The Model is a Random Forest Regressor. A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

# Datasets <a name="3"/>
The Datasets used are two:
- UNBC-McMASTER Shoulder Pain Expression : contains video sequences of patients' faces when they were actively and passively moving their shoulders following painful impulses. You can download it [here](https://datasets.bifrost.ai/info/1439).
- BioVid Heat Pain Database (BioVid) :  is a recent dataset created to improve the reliability and objectivity of pain measurement. <br/>
 You can download it [here](https://nextcloud.univ-lille.fr/index.php/s/MjFirkrqBZmbb7w).<br/>
 ![](Images/Biovid.png) <br/>
# Implementation <a name="4"/>
The project is based on two scripts called ```PreliminaryClustering.py``` and ```ModelRFR.py```, which have the purpose of implementing respectively the phase suitable for extracting the relevant configurations and that relating to the management of Random Forest Regression. The script used to perform these tests is ```test_regression.py```, whose purpose is to be able to compare the results obtained when the value used as a threshold for neutral configurations varies and at the same time evaluate the different groupings of landmarks. This is done by scrolling through 5 groups of landmarks, representing in order the eyes, nose, mouth, the best configuration and all the landmarks. The other script implemented, called ```generate_model_predictor.py```, allows instead to evaluate the performance of a fixed model both the number of kernels of the GMM and the threshold to be used for the extraction of neutral configurations.

# Authors
- **Lorenzo Gianassi**
- **Francesco Gigli**
# Acknowledgments
Image and Video Analysis Project © Course held by Professor [Pietro Pala](https://www.unifi.it/p-doc2-2012-200006-P-3f2a3d30372e2a.html) - Computer Engineering Master Degree @[University of Florence](https://www.unifi.it/changelang-eng.html)
