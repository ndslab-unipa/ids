import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras import layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import joblib
from utils_ids import *

class MultiLayerClassifierIDS:
    """
    Multi-layer Intrusion Detection System (IDS) composed of:
    - Multiple expert classifiers (Decision Tree, Random Forest, Neural Network)
    - A gatekeeper for benign/attack filtering
    - A meta-classifier combining expert predictions
    - Ensemble voting with heuristic-based confidence weighting

    Attributes:
        data_path (str): Path where training/testing CSV files are stored.
        models_path (str): Path where trained models are saved.
        column_dict (dict): Mapping from raw labels to normalized attack labels.
        labels_list (list): List of normalized attack labels.
        input_shape (int): Number of input features.
        output_shape (int): Number of output classes.
        tree, forest (sklearn estimators): Expert tree-based classifiers.
        nn_clf (keras.Model): Expert neural network classifier.
        metaclassifier (sklearn estimator): Meta-classifier combining expert outputs.
        gatekeeper (sklearn estimator): Binary classifier for benign filtering.
    """

    def __init__(self, data_path='data', models_path='models'):
        """
        Initialize the IDS architecture and set up models and structure.

        Args:
            data_path (str): Directory with dataset CSV files.
            models_path (str): Directory where trained models will be stored.
        """
        # Mapping of raw labels to short identifiers
        self.column_dict = {
            "BENIGN": 'BENIGN',
            "DoS Hulk": 'Dos Hulk',
            "Portscan": 'Portscan',
            "DDoS": 'DDoS',
            "DoS GoldenEye": 'Dos GE',
            "FTP-Patator": 'FTP-Patator',
            "SSH-Patator": 'SSH-Patator',
            "DoS Slowloris": 'Dos SL',
            "DoS Slowhttptest": 'Dos SHT',
            "Botnet": 'Botnet',
            "Web Attack - Brute Force": 'WA - BF',
            "Web Attack - XSS": 'WA - XSS',
            "Infiltration": 'Inf',
            "Web Attack - SQL Injection": 'WA - SQLi',
            "Heartbleed": 'Heartbleed',
        }

        # Precompute label list and dataframe format
        self.labels_list = list(self.column_dict.values())
        self.labels_df = pd.DataFrame(self.labels_list, columns=["Label"])
        self.data_path = data_path
        self.models_path = models_path
        
        # Dimensionality of input/output descriptors
        self.input_shape = 76
        self.output_shape = len(self.labels_df.Label.unique())
        
        # Initialize expert classifiers
        self.tree = DecisionTreeClassifier(criterion="entropy", max_depth=50) 
        self.forest = RandomForestClassifier(criterion="entropy", max_depth=50)
        self.nn_clf = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=[self.input_shape]),
                layers.Dense(64, activation='relu'),
                layers.Dropout(rate=0.3),
                layers.Dense(self.output_shape, activation='softmax'),
            ])
        self.nn_clf.compile(
            optimizer=Adam(1e-2),
            loss='categorical_crossentropy'
        )
        self.early_stopping = EarlyStopping(
                min_delta=0.001, 
                patience=5, 
                restore_best_weights=True
        )
        # Meta-classifier and gatekeeper
        self.metaclassifier = DecisionTreeClassifier(criterion="entropy")
        self.gatekeeper = DecisionTreeClassifier(criterion="entropy", max_depth=15)
  
    def load_training_data(self):
        """
        Load all training and validation datasets required by the experts.

        Returns:
            tuple: (X_train, y_train, X_nn, y_nn, X_val, y_val)
                - X_train (pd.DataFrame): Features for tree-based models.
                - y_train (pd.DataFrame): Labels for tree-based models.
                - X_nn, y_nn (pd.DataFrame): Neural network training set.
                - X_val, y_val (pd.DataFrame): Validation set for NN & performance weighting.
        """
        # Load datasets for Decision Tree and Random Forest
        X_train = pd.read_csv(f'{self.data_path}/X_train.csv')
        y_train = pd.read_csv(f'{self.data_path}/y_train.csv')
        
        # Load datasets for neural network expert
        X_nn = pd.read_csv(f'{self.data_path}/X_nn.csv')
        X_val = pd.read_csv(f'{self.data_path}/X_val.csv')
        y_nn = pd.read_csv(f'{self.data_path}/y_nn.csv')
        y_val = pd.read_csv(f'{self.data_path}/y_val.csv')
        return X_train, y_train, X_nn, y_nn, X_val, y_val
    
    def load_testing_data(self):
        """
        Load the testing dataset for evaluation.

        Returns:
            tuple: (X_test, y_test)
                - X_test (pd.DataFrame): Features for testing.
                - y_test (pd.DataFrame): True labels for testing.
        """
        X_test = pd.read_csv(f'{self.data_path}/X_test.csv')
        y_test = pd.read_csv(f'{self.data_path}/y_test.csv')
        return X_test, y_test
    
    def train_expert_members(self):
        """
        Train individual expert classifiers: Decision Tree, Random Forest, and Neural Network.

        Steps:
            - Load training and validation datasets
            - One-hot encode labels for the neural network
            - Train each expert on the respective training data
            - Save trained expert models to disk
        """
        X_train, y_train, X_nn, y_nn, X_val, y_val = self.load_training_data()
        y_train_one_hot = pd.get_dummies(y_nn, prefix='', prefix_sep='')
        y_val_one_hot = pd.get_dummies(y_val, prefix='', prefix_sep='')   

        # training expert systems
        self.tree.fit(X_train, y_train.to_numpy().ravel())
        self.forest.fit(X_train, y_train.to_numpy().ravel())
        self.nn_clf.fit(
                X_nn, 
                y_train_one_hot,
                validation_data=(X_val, y_val_one_hot),
                batch_size=128,
                epochs=30,
                callbacks =[self.early_stopping],
                verbose=1
            )
        
        # saving expert models
        self.nn_clf.save(f'{self.models_path}/expert_nn_clf.h5')
        joblib.dump(self.tree, f'{self.models_path}/expert-tree.joblib')
        joblib.dump(self.forest, f'{self.models_path}/expert-forest.joblib')
    
    def train_gatekeeper(self):
        """
        Train the gatekeeper classifier to filter benign samples.

        Steps:
            - Load training dataset
            - Binarize labels for gatekeeper training
            - Fit the gatekeeper tree model
            - Save trained model to disk
        """
        X_train, y_train, _, _, _, _ = self.load_training_data()
        X_train["Label"] = y_train

        # Only benign samples for specific training analysis (optional)
        benign_train = X_train[X_train["Label"] == "BENIGN"]

        X_train = X_train.drop(columns="Label")
        benign_train = benign_train.drop(columns="Label")

        y_train_bin = binarize(y_train)
        self.gatekeeper.fit(X_train, y_train_bin.to_numpy().ravel())

        # save gatekeeper trained model
        joblib.dump(self.gatekeeper, f'{self.models_path}/gatekeeper-tree.joblib')

    def generate_weights(self, iterations=10):
        """
        Generate per-class F1-score weights for expert classifiers across multiple iterations.

        Args:
            iterations (int): Number of repeated trainings to calculate stable weights.

        Steps:
            - Train tree, forest, and neural network for each iteration
            - Compute F1-scores on validation set for each class
            - Save averaged weights for all experts
        """
        X_train, y_train, X_nn, y_nn, X_val, y_val = self.load_training_data()
        y_train_one_hot = pd.get_dummies(y_nn, prefix='', prefix_sep='')
        y_val_one_hot = pd.get_dummies(y_val, prefix='', prefix_sep='')

        # Temporary models for weight calculation
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=50)
        forest = RandomForestClassifier(criterion="entropy", max_depth=50)
        nn_clf = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=[self.input_shape]),
            layers.Dense(64, activation='relu'),
            layers.Dropout(rate=0.3),
            layers.Dense(self.output_shape, activation='softmax'),
        ])
        nn_clf.compile(
            optimizer=Adam(1e-2),
            loss='categorical_crossentropy'
        )
        early_stopping = EarlyStopping(
                min_delta=0.001, 
                patience=5, 
                restore_best_weights=True
        )

        # Initialize F1-score storage
        dt_weights = pd.DataFrame(index=self.labels_df.Label.unique(), columns=[i for i in range(iterations)])
        rf_weights = pd.DataFrame(index=self.labels_df.Label.unique(), columns=[i for i in range(iterations)])
        nn_weights = pd.DataFrame(index=self.labels_df.Label.unique(), columns=[i for i in range(iterations)])
        for i in range(iterations):
            # Train experts
            tree.fit(X_train, y_train.to_numpy().ravel())
            forest.fit(X_train, y_train.to_numpy().ravel())
            nn_clf.fit(
                X_nn, 
                y_train_one_hot,
                validation_data=(X_val, y_val_one_hot),
                batch_size=128,
                epochs=30,
                callbacks =[early_stopping],
                verbose=1
            )

            # Get predictions
            forest_pred = forest.predict(X_val)
            tree_pred = tree.predict(X_val)
            nn_proba = nn_clf.predict(X_val)
            nn_pred = [y_train_one_hot.columns[np.argmax(one_hot)] for one_hot in nn_proba]
            
            # Compute classification reports
            tree_report = classification_report(y_val, tree_pred, output_dict=True)
            forest_report = classification_report(y_val, forest_pred, output_dict=True)
            nn_report = classification_report(y_val, nn_pred, output_dict=True)
            
            # Extract F1-scores per class
            keys = list(tree_report.keys())
            keys.remove('accuracy')
            keys.remove('macro avg')
            keys.remove('weighted avg')
            for key in keys:
                dt_weights.loc[key][i]= tree_report[key]["f1-score"]
                rf_weights.loc[key][i]= forest_report[key]["f1-score"]
                nn_weights.loc[key][i]= nn_report[key]["f1-score"]
            

        # Save weights
        dt_weights.to_csv(f"{self.data_path}/expert_dt_weights.csv")
        rf_weights.to_csv(f"{self.data_path}/expert_rf_weights.csv")
        nn_weights.to_csv(f"{self.data_path}/expert_nn_weights.csv")


    def generate_metadataset(self):
        """
        Generate the meta-dataset for training the meta-classifier.

        Steps:
            - Load training and validation data
            - Train temporary experts (tree, forest, NN)
            - Compute weighted probability distributions and entropies
            - Concatenate all expert outputs to form meta-features
            - Save meta-dataset and meta-labels
            - Train the meta-classifier on the meta-dataset
        """
        X_train, y_train, X_nn, y_nn, X_val, y_val = self.load_training_data()
        y_train_one_hot = pd.get_dummies(y_nn, prefix='', prefix_sep='')
        y_val_one_hot = pd.get_dummies(y_val, prefix='', prefix_sep='')

        # Initialize temporary expert models
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=50)
        forest = RandomForestClassifier(criterion="entropy", max_depth=50)
        nn_clf = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=[self.input_shape]),
            layers.Dense(64, activation='relu'),
            layers.Dropout(rate=0.3),
            layers.Dense(self.output_shape, activation='softmax'),
        ])

        nn_clf.compile(
            optimizer=Adam(1e-2),
            loss='categorical_crossentropy'
        )

        early_stopping = EarlyStopping(
                min_delta=0.001, 
                patience=5, 
                restore_best_weights=True
        )

        # Load previously generated expert weights
        dt_weights = pd.read_csv(f"{self.data_path}/expert_dt_weights.csv", index_col=0)
        rf_weights = pd.read_csv(f"{self.data_path}/expert_rf_weights.csv", index_col=0)
        nn_weights = pd.read_csv(f"{self.data_path}/expert_nn_weights.csv", index_col=0)

        # media sui 10 valori per ottenere un peso per ciascuna calsse di attacco
        dt_weights = dt_weights.mean(axis=1).to_dict()
        rf_weights = rf_weights.mean(axis=1).to_dict()
        nn_weights = nn_weights.mean(axis=1).to_dict()

        X_metadataset = pd.DataFrame() 

        # Train experts on full training data
        tree.fit(X_train, y_train.to_numpy().ravel())
        forest.fit(X_train, y_train.to_numpy().ravel())
        nn_clf.fit(
            X_nn,
            y_train_one_hot,
            validation_data=(X_val, y_val_one_hot),
            batch_size=128,
            epochs=30,
            callbacks =[early_stopping],
            verbose=1
        )
        
        # Compute probabilities and log-probabilities
        tree_proba = tree.predict_proba(X_val)
        forest_proba = forest.predict_proba(X_val)
        nn_proba = nn_clf.predict(X_val)
        tree_log_proba = tree.predict_log_proba(X_val)
        forest_log_proba = forest.predict_log_proba(X_val)
        nn_log_proba = np.log(nn_proba)

        # Compute weighted probabilities and entropies
        rf_proba_df  = pd.DataFrame(forest_proba)
        rf_proba_weighted = pd.DataFrame(weighted_proba(forest_proba, forest.classes_, rf_weights))
        rf_entropy   = pd.DataFrame(entropy(forest_proba, forest_log_proba))
        dt_proba_df  = pd.DataFrame(tree_proba)
        dt_proba_weighted = pd.DataFrame(weighted_proba(tree_proba, tree.classes_, dt_weights))
        dt_entropy   = pd.DataFrame(entropy(tree_proba, tree_log_proba))
        nn_proba_df  = pd.DataFrame(nn_proba)
        nn_proba_weighted = pd.DataFrame(weighted_proba(nn_proba, y_nn.Label.unique(), nn_weights))
        nn_entropy   = pd.DataFrame(entropy(nn_proba, nn_log_proba))
            
        # Form meta-dataset by concatenating expert outputs
        X_metadataset = pd.concat([dt_proba_df, dt_proba_weighted, dt_entropy, rf_proba_df, rf_proba_weighted, rf_entropy, nn_proba_df, nn_proba_weighted, nn_entropy], axis=1)
        y_metadaset = y_val.copy()
        X_metadataset.columns = ['s' + str(x) for x in range(X_metadataset.columns.shape[0])]

        # Save meta-dataset and labels
        X_metadataset.to_csv(f"{self.data_path}/clf_metadataset.csv")
        y_metadaset.to_csv(f"{self.data_path}/clf_metalabel.csv")

        # Train the meta-classifier
        self.train_meta_classifier(X_train=X_metadataset, y_train=y_metadaset)

    def train_meta_classifier(self, X_train, y_train):
        """
        Train the meta-classifier using the generated meta-dataset.

        Args:
            X_train (pd.DataFrame): Meta-features derived from expert predictions
            y_train (pd.Series): Original labels corresponding to meta-features

        Steps:
            - Fit the meta-classifier
            - Save the trained model to disk
        """
        self.metaclassifier.fit(X_train, y_train)
        joblib.dump(self.metaclassifier, f'{self.models_path}/metaclassifier.joblib')
    
    def generate_capacity(self, iterations=10):
        """
        Compute and save weights for ensemble and meta-classifier predictions.

        Args:
            iterations (int): Number of iterations to calculate stable performance weights

        Steps:
            - Generate weighted predictions for each expert
            - Compute weighted voting and meta-classifier predictions
            - Save ensemble and meta-classifier F1-score weights
        """
        X_train, y_train, X_nn, y_nn, X_val, y_val = self.load_training_data()
        y_train_one_hot = pd.get_dummies(y_nn, prefix='', prefix_sep='')
        y_val_one_hot = pd.get_dummies(y_val, prefix='', prefix_sep='')

        # Temporary expert models
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=50)
        forest = RandomForestClassifier(criterion="entropy", max_depth=50)
        nn_clf = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=[self.input_shape]),
            layers.Dense(64, activation='relu'),
            layers.Dropout(rate=0.3),
            layers.Dense(self.output_shape, activation='softmax'),
        ])

        nn_clf.compile(
            optimizer=Adam(1e-2),
            loss='categorical_crossentropy'
        )

        early_stopping = EarlyStopping(
                min_delta=0.001, 
                patience=5, 
                restore_best_weights=True
        )
        
        # Load precomputed expert weights
        dt_weights = pd.read_csv(f"{self.data_path}/expert_dt_weights.csv", index_col=0)
        rf_weights = pd.read_csv(f"{self.data_path}/expert_rf_weights.csv", index_col=0)
        nn_weights = pd.read_csv(f"{self.data_path}/expert_nn_weights.csv", index_col=0)

        dt_weights = dt_weights.mean(axis=1).to_dict()
        rf_weights = rf_weights.mean(axis=1).to_dict()
        nn_weights = nn_weights.mean(axis=1).to_dict()

        label = pd.DataFrame(self.labels_list, columns=["Label"])
        ensemble_weights = pd.DataFrame(index=label.Label.unique(), columns=[i for i in range(iterations)])
        meta_weights = pd.DataFrame(index=label.Label.unique(), columns=[i for i in range(iterations)])

        for z in range(iterations):
            # Train experts
            tree.fit(X_train, y_train.to_numpy().ravel())
            forest.fit(X_train, y_train.to_numpy().ravel())
            nn_clf.fit(
                X_nn, 
                y_train_one_hot,
                validation_data=(X_val, y_val_one_hot),
                batch_size=128,
                epochs=30,
                callbacks =[early_stopping],
                verbose=1
            )
        
            # Compute weighted expert outputs
            forest_proba = forest.predict_proba(X_val)
            tree_proba = tree.predict_proba(X_val)
            nn_proba = nn_clf.predict(X_val)

            rf_log_proba = forest.predict_log_proba(X_val)
            dt_log_proba = tree.predict_log_proba(X_val)
            nn_log_proba = np.log(nn_proba)
            
            rf_proba_df  = pd.DataFrame(forest_proba)
            rf_proba_weighted = pd.DataFrame(weighted_proba(forest_proba, forest.classes_, rf_weights))
            rf_entropy   = pd.DataFrame(entropy(forest_proba, rf_log_proba))

            dt_proba_df  = pd.DataFrame(tree_proba)
            dt_proba_weighted = pd.DataFrame(weighted_proba(tree_proba, tree.classes_, dt_weights))
            dt_entropy   = pd.DataFrame(entropy(tree_proba, dt_log_proba))

            nn_proba_df  = pd.DataFrame(nn_proba)
            nn_proba_weighted = pd.DataFrame(weighted_proba(nn_proba, y_nn.Label.unique(), nn_weights))
            nn_entropy   = pd.DataFrame(entropy(nn_proba, nn_log_proba))

            # Generate meta-dataset for evaluation
            print("Meta classifier")
            X_val_meta = pd.concat([dt_proba_df, dt_proba_weighted, dt_entropy, rf_proba_df, rf_proba_weighted, rf_entropy, nn_proba_df, nn_proba_weighted, nn_entropy], axis=1)
            X_val_meta.columns = ['s' + str(x) for x in range(X_val_meta.columns.shape[0])]
            # Predict with meta-classifier
            meta_pred = self.metaclassifier.predict(X_val_meta)
        
            # compute ensemble F1-scores
            print("Ensemble")
            ensemble_pred = list()
            rf_proba_weighted.columns = forest.classes_
            dt_proba_weighted.columns = tree.classes_
            nn_proba_weighted.columns = y_nn.Label.unique()

            for i in range(len(tree_proba)):
                vote = {l: 0 for l in self.labels_list}
                for column in dt_proba_weighted.columns:
                        vote[column] = dt_proba_weighted.iloc[i][column] + rf_proba_weighted.iloc[i][column] + nn_proba_weighted.iloc[i][column]
                most_voted = max(vote, key=vote.get)
                ensemble_pred.append(most_voted)

            # Compute per-class F1 scores    
            ensemble_report = classification_report(y_val, ensemble_pred, output_dict=True)
            meta_report = classification_report(y_val, meta_pred, output_dict=True)
            
            keys = list(ensemble_report.keys())
            keys.remove('accuracy')
            keys.remove('macro avg')
            keys.remove('weighted avg')
            
            for key in keys:
                ensemble_weights.loc[key][z]= ensemble_report[key]["f1-score"]
                meta_weights.loc[key][z]= meta_report[key]["f1-score"]

        # Save the weights
        ensemble_weights.to_csv(f"{self.data_path}/ensemble_weights.csv")
        meta_weights.to_csv(f"{self.data_path}/meta_weights.csv")
    
    def _load_models(self):
        """
        Load all trained models from disk into the current instance.

        Models loaded:
            - Neural Network expert
            - Decision Tree expert
            - Random Forest expert
            - Gatekeeper (binary tree classifier)

        This method should be called before performing evaluation or predictions.
        """
        self.nn_clf = load_model(f'{self.models_path}/expert_nn_clf.h5')
        self.tree = joblib.load(f'{self.models_path}/expert-tree.joblib')
        self.forest = joblib.load(f'{self.models_path}/expert-forest.joblib')
        self.gatekeeper = joblib.load(f'{self.models_path}/gatekeeper-tree.joblib')
        self.metaclassifier = joblib.load(f'{self.models_path}/metaclassifier.joblib')
    
    def _load_weights(self):
        """
        Load all precomputed weights for experts, ensemble, and meta-classifier.

        Returns:
            dt_weights (dict): Decision Tree F1-score weights per class
            rf_weights (dict): Random Forest F1-score weights per class
            nn_weights (dict): Neural Network F1-score weights per class
            ensemble_weights (dict): Ensemble F1-score weights per class
            meta_weights (dict): Meta-classifier F1-score weights per class

        Steps:
            - Load CSV files with per-class weights
            - Compute mean across iterations for stable weights
        """
        dt_weights = pd.read_csv(f"{self.data_path}/expert_dt_weights.csv", index_col=0)
        rf_weights = pd.read_csv(f"{self.data_path}/expert_rf_weights.csv", index_col=0)
        nn_weights = pd.read_csv(f"{self.data_path}/expert_nn_weights.csv", index_col=0)
        ensemble_weights = pd.read_csv(f"{self.data_path}/ensemble_weights.csv", index_col=0)
        meta_weights = pd.read_csv(f"{self.data_path}/meta_weights.csv", index_col=0)
        dt_weights = dt_weights.mean(axis=1).to_dict()
        rf_weights = rf_weights.mean(axis=1).to_dict()
        nn_weights = nn_weights.mean(axis=1).to_dict()
        ensemble_weights = ensemble_weights.mean(axis=1).to_dict()
        meta_weights = meta_weights.mean(axis=1).to_dict()
        return dt_weights, rf_weights, nn_weights, ensemble_weights, meta_weights
    
    def evaluate(self, X_test):
        """
        Evaluate the ensemble model on a given test set.

        Args:
            X_test (pd.DataFrame): Test data

        Returns:
            final_predictions (list): Final predicted labels for each sample

        Steps:
            1. Apply gatekeeper to filter samples
            2. Compute expert model outputs (weighted probabilities, entropies)
            3. Compute meta-classifier predictions
            4. Perform weighted voting between ensemble and meta-classifier
        """
        dt_weights, rf_weights, nn_weights, ensemble_weights, meta_weights = self._load_weights()
        
        # Gatekeeper selection: remove samples classified as benign
        X_test_filtered = self.gatekeeper_selection(X_test=X_test)

        # Compute expert outputs on filtered test set
        X_test_meta, rf_proba_weighted, dt_proba_weighted, nn_proba_weighted, classes_experts = self.compute_expert_outputs(X_test=X_test_filtered, rf_weights=rf_weights, dt_weights=dt_weights, nn_weights=nn_weights)
        
        # Compute meta-classifier predictions
        meta_predict, meta_proba, meta_log_proba = self.compute_metapredictions(X_test_meta=X_test_meta)
        
        # Weighted voting between ensemble and meta-classifier
        rf_proba_weighted.columns = classes_experts
        dt_proba_weighted.columns = classes_experts
        nn_proba_weighted.columns = classes_experts

        y_preds = self.voting_function(dt_proba_weighted, rf_proba_weighted, nn_proba_weighted, meta_proba, meta_log_proba, ensemble_weights, meta_weights, meta_predict)
        return y_preds

    def gatekeeper_selection(self, X_test):
        """
        Filter the test set using the gatekeeper model.

        Args:
            X_test (pd.DataFrame): Test data

        Returns:
            X_test_new (pd.DataFrame): Filtered test set where gatekeeper predicts 1 (non-benign)

        Steps:
            - Predict with gatekeeper
            - Keep only samples where gk_pred == 1
        """
        gk_pred = self.gatekeeper.predict(X_test)
        X_test['gk_pred'] = gk_pred
        X_test_new = X_test[X_test['gk_pred'] == 1].drop(columns=['gk_pred'])
        return X_test_new
    
    def compute_expert_outputs(self, X_test, rf_weights, dt_weights, nn_weights):
        """
        Compute predictions, weighted probabilities, and entropies from expert models.

        Args:
            X_test (pd.DataFrame): Input data
            rf_weights (dict): Random Forest per-class weights
            dt_weights (dict): Decision Tree per-class weights
            nn_weights (dict): Neural Network per-class weights

        Returns:
            X_test_meta (pd.DataFrame): Meta-features for meta-classifier
            rf_proba_weighted (pd.DataFrame): Weighted probabilities from Random Forest
            dt_proba_weighted (pd.DataFrame): Weighted probabilities from Decision Tree
            nn_proba_weighted (pd.DataFrame): Weighted probabilities from Neural Network
            classes_experts (list): List of expert classes

        Steps:
            - Compute predicted probabilities and log-probabilities
            - Compute weighted probabilities using per-class weights
            - Compute entropy for each expert
            - Concatenate all outputs to form meta-features
        """
        classes_experts = self.forest.classes_

        forest_proba = self.forest.predict_proba(X_test)
        tree_proba = self.tree.predict_proba(X_test)
        nn_proba = self.nn_clf.predict(X_test)
        #
        rf_log_proba = self.forest.predict_log_proba(X_test)
        dt_log_proba = self.tree.predict_log_proba(X_test)
        nn_log_proba = np.log(nn_proba)
        #
        rf_proba_df  = pd.DataFrame(forest_proba)
        rf_proba_weighted = pd.DataFrame(weighted_proba(forest_proba, classes_experts, rf_weights))
        rf_entropy   = pd.DataFrame(entropy(forest_proba, rf_log_proba))
        #
        dt_proba_df  = pd.DataFrame(tree_proba)
        dt_proba_weighted = pd.DataFrame(weighted_proba(tree_proba, classes_experts, dt_weights))
        dt_entropy   = pd.DataFrame(entropy(tree_proba, dt_log_proba))
        # 
        nn_proba_df  = pd.DataFrame(nn_proba)
        nn_proba_weighted = pd.DataFrame(weighted_proba(nn_proba, classes_experts, nn_weights))
        nn_entropy   = pd.DataFrame(entropy(nn_proba, nn_log_proba))
        #
        # Meta-dataset for meta-classifier
        X_test_meta = pd.concat([dt_proba_df, dt_proba_weighted, dt_entropy, rf_proba_df, rf_proba_weighted, rf_entropy, nn_proba_df, nn_proba_weighted, nn_entropy], axis=1)
        X_test_meta.columns = ['s' + str(x) for x in range(X_test_meta.columns.shape[0])]
        return X_test_meta, rf_proba_weighted, dt_proba_weighted, nn_proba_weighted, classes_experts

    def compute_metapredictions(self, X_test_meta):
        """
        Compute predictions, probabilities, and log-probabilities from the meta-classifier.

        Args:
            X_test_meta (pd.DataFrame): Meta-features generated from expert outputs

        Returns:
            meta_predict (np.array): Predicted labels from meta-classifier
            meta_proba (np.array): Predicted probabilities
            meta_log_proba (np.array): Log-probabilities for entropy computation
        """
        meta_predict = self.metaclassifier.predict(X_test_meta)
        meta_proba = self.metaclassifier.predict_proba(X_test_meta)
        meta_log_proba = self.metaclassifier.predict_log_proba(X_test_meta)
        return meta_predict, meta_proba, meta_log_proba

    def voting_function(self, dt_proba_weighted, rf_proba_weighted, nn_proba_weighted, meta_proba, meta_log_proba, ensemble_weights, meta_weights, meta_predict, gamma=0.5):
        """
        Compute final predictions using weighted voting between ensemble and meta-classifier.

        Args:
            dt_proba_weighted (pd.DataFrame): Weighted DT probabilities
            rf_proba_weighted (pd.DataFrame): Weighted RF probabilities
            nn_proba_weighted (pd.DataFrame): Weighted NN probabilities
            meta_proba (np.array): Meta-classifier probabilities
            meta_log_proba (np.array): Meta-classifier log-probabilities
            ensemble_weights (dict): Precomputed F1-score weights for ensemble
            meta_weights (dict): Precomputed F1-score weights for meta-classifier
            meta_predict (np.array): Predicted labels from meta-classifier
            gamma (float): Weighting factor between model performance and confidence

        Returns:
            final_pred (list): Final predicted labels after voting

        Steps:
            - Compute ensemble probabilities and confidence (entropy-based)
            - Compute meta-classifier confidence
            - Compute heuristic scores for ensemble and meta
            - Select final label based on higher heuristic score
        """
        ensemble_pred = list()
        meta_pred = list()
        final_pred = list()

        for i in range(len(dt_proba_weighted)):
            vote = {l: 0 for l in self.labels_list}

            for column in dt_proba_weighted.columns:
                    vote[column] = dt_proba_weighted.iloc[i][column] + rf_proba_weighted.iloc[i][column] + nn_proba_weighted.iloc[i][column]
            
            most_voted = max(vote, key=vote.get)
            
            votes = np.array(list(vote.values()))
            
            ensemble_proba = votes / sum(votes)
            ensemble_log_proba = np.log(ensemble_proba)
            ensemble_entropy = entropy_single(ensemble_proba, ensemble_log_proba)
            
            meta_entropy = entropy_single(meta_proba[i], meta_log_proba[i])
            
            # Entropy-based confidence
            ensemble_confidence = 1 - ensemble_entropy
            meta_confidence = 1- meta_entropy
            
            # Heuristic score combining model F1 and confidence
            s_ensemble = (gamma * ensemble_weights[most_voted]) + ((1 - gamma) * ensemble_confidence)
            s_meta = (gamma * meta_weights[meta_predict[i]]) + ((1 - gamma) * meta_confidence)
            
            # Final decision based on max heuristic
            final_decision = ''
            if s_ensemble >= s_meta:
                final_decision = most_voted  
            else:
                final_decision = meta_predict[i]

            ensemble_pred.append(most_voted)
            meta_pred.append(meta_predict[i])
            final_pred.append(final_decision)
        return final_pred
    
    
    def training(self):
        """
        Complete training pipeline for the ensemble IDS system.

        Steps:
            1. Train expert classifiers (tree, forest, neural network)
            2. Train the gatekeeper
            3. Generate per-class weights for experts
            4. Train meta-classifier using expert outputs
            5. Compute initial model capacity for ensemble and meta-classifier
        """
        print("Training expert members of the ensemble")
        self.train_expert_members()
        print("Training Gatekeeper")
        self.train_gatekeeper()
        print("Extracting weights for each expert member")
        self.generate_weights()
        print("Training meta-classifier on Experts outputs")
        self.generate_metadataset()
        print("Compute initial model capacity")
        self.generate_capacity()
        print("Initial training completed")





