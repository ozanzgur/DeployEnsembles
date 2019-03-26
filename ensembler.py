
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import roc_callback

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pickle
import os.path
import gc
from imblearn.over_sampling import SMOTE

import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

#from skopt.space import Real, Integer
#from skopt.utils import use_named_args
#from skopt import gp_minimize
from hyperopt import hp, tpe, fmin
from hyperopt import Trials

from IPython.display import clear_output

# Some starter parameters with a wide range ###############
lgb_starter_parameters = dict(
    num_leaves = hp.choice('num_leaves', np.arange(30, 2000, dtype=int)),
    #max_bin = hp.choice('max_bin', np.arange(128, 155, dtype=int)),
    min_data_in_leaf = hp.choice('min_data_in_leaf', np.arange(30, 1000, dtype=int)),
    bagging_freq = hp.choice('bagging_freq', np.arange(2, 40, dtype=int)),
    #max_depth = hp.choice('max_depth', np.arange(2, 40, dtype=int)),
    
    learning_rate = hp.loguniform('learning_rate', -6, -2),
    min_sum_hessian_in_leaf = hp.loguniform('min_sum_hessian_in_leaf', -4, 3),
    reg_alpha = hp.loguniform('reg_alpha', -7, 3),
    reg_lambda = hp.loguniform('reg_lambda', -7, 3),
    bagging_fraction = hp.uniform('bagging_fraction', 0.1, 0.6),
    feature_fraction = hp.uniform('feature_fraction', 0.05, 0.25),
    
    
    is_unbalance = hp.choice('is_unbalance', [True, False]),
    boost_from_average = hp.choice('boost_from_average', [True, False]),
    tree_learner = hp.choice('tree_learner', ['serial', 'feature', 'data', 'voting'])
    #hp.choice(
)

GaussianNB_starter_parameters = dict(
    var_smoothing = hp.loguniform('var_smoothing', -10, -1)
)

SVC_starter_parameters = dict(
    C = hp.loguniform('C', -4, 3),
    tol = hp.loguniform('tol', -5, 0),
    gamma = hp.loguniform('gamma', -5, 2),
    class_weight = hp.choice('class_weight', ['balanced', None])
)
#############################################################

class Ensembler():
    def __init__(self, name, score = roc_auc_score, X_train = None, y_train = None, X_valid = None, y_valid = None, X_test = None, y_test = None):
        super(Ensembler, self).__init__()
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.num_models = 0
        self.top_ens_score = 0
        self.score = score
        
        self.out_train = None
        self.out_valid = None
        self.out_test = None
        self.weights = None
        self.out_models = []
        
    
    def ShuffleClasses(self):
        cols = self.X_train.columns
        zeros = self.X_train[self.y_train == 0].values
        zeros = zeros.T
        np.random.shuffle(zeros)
        zeros = zeros.T
        zeros = pd.DataFrame(zeros, columns = cols)
        self.X_train.loc[self.y_train == 0] = zeros
        
        ones = self.X_train[self.y_train == 1].values
        ones = ones.T
        np.random.shuffle(ones)
        ones = ones.T
        ones = pd.DataFrame(ones, columns = cols)
        self.X_train.loc[self.y_train == 1] = ones
    
    #def LoadValues(self, dict_values):
    #    for key in dict_values:
    #        if eval(key) in locals():
    #            key_eval = eval(key)
    #            self.key_eval = dict_values[key]
    #        else:
    #            print("Error: {} cannot be kept in EnsembleDataStorage.".format(key))
    
    
    def StandardizeData(self):
        from sklearn import preprocessing

        cols = self.X_train.columns
        scaler = preprocessing.StandardScaler().fit(pd.concat([self.X_train, self.X_test], axis = 0))

        self.X_train = pd.DataFrame(scaler.transform(self.X_train), columns = cols)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns = cols)
        if self.X_valid is not None:
            self.X_valid = pd.DataFrame(scaler.transform(self.X_valid), columns = cols)
        print("Standardization...")
    
    def Summary(self):
        print("# Models: {}".format(self.num_models))
        if self.X_train is not None:
            print("X_train: {}".format(self.X_train.shape))
        if self.y_train is not None:
            print("y_train: {}".format(self.y_train.shape))
        if self.X_valid is not None:
            print("X_valid: {}".format(self.X_valid.shape))
        if self.y_valid is not None:
            print("y_valid: {}".format(self.y_valid.shape))
        if self.X_test is not None:
            print("X_test: {}".format(self.X_test.shape))
        if self.y_test is not None:
            print("y_test: {}".format(self.y_test.shape))
            
        if self.out_train != None:
            print("# out_train: {}".format(len(self.out_train)))
        if self.out_valid != None:
            print("# out_valid: {}".format(len(self.out_valid)))
        if self.out_test != None:
            print("# out_test: {}".format(len(self.out_test)))
        if self.weights != None:
            print("# weights: {}".format(len(self.weights)))
    
    def SaveAll(self, save_data = True):
        try:  
            os.mkdir('./' + self.name)
        except OSError:  
            print ("Didn't create new directory.")
        else:  
            print ("Successfully created the directory %s ")
        
        properties = dict(
            num_models = self.num_models,
            top_ens_score = self.top_ens_score,
            score = self.score,
            out_models = self.out_models
        )
        
        with open("{}/properties.txt".format(self.name), "wb") as fp:
            pickle.dump(properties, fp)
            print("Saved properties: {}".format(properties))
        
        if save_data:
            if self.X_valid is not None:
                self.X_valid.to_csv('{}/X_valid.csv'.format(self.name), index = 0)
                print("Saved X_valid")
            if self.y_valid is not None:
                self.y_valid.to_csv('{}/y_valid.csv'.format(self.name), index = 0)
                print("Saved y_valid")
            if self.X_train is not None:
                self.X_train.to_csv('{}/X_train.csv'.format(self.name), index = 0)
                print("Saved X_train")
            if self.y_train is not None:
                self.y_train.to_csv('{}/y_train.csv'.format(self.name), index = 0)
                print("Saved y_train")
            if self.X_test is not None:
                self.X_test.to_csv('{}/X_test.csv'.format(self.name), index = 0)
                print("Saved X_test")
            if self.y_test is not None:
                self.y_test.to_csv('{}/y_test.csv'.format(self.name), index = 0)
                print("Saved y_test")
        else:
            print("Not saving data.")
            
        if self.out_train != None:
            with open("{}/out_train.txt".format(self.name), "wb") as fp:
                pickle.dump(self.out_train, fp)
                print("Saved out_train, len: {}".format(len(self.out_train)))
        if self.out_valid != None:
            with open("{}/out_valid.txt".format(self.name), "wb") as fp:
                pickle.dump(self.out_valid, fp)
                print("Saved out_valid, len: {}".format(len(self.out_valid)))
        if self.out_test != None:
            with open("{}/out_test.txt".format(self.name), "wb") as fp:
                pickle.dump(self.out_test, fp)
                print("Saved out_test, len: {}".format(len(self.out_test)))
        if self.weights != None:
            with open("{}/weights.txt".format(self.name), "wb") as fp:
                pickle.dump(self.weights, fp)
                print("Saved weights, len: {}".format(len(self.weights)))
                
    def LoadAll(self):
        if os.path.isfile('{}/properties.txt'.format(self.name)):
            properties = None
            with open('{}/properties.txt'.format(self.name), "rb") as fp:
                properties = pickle.load(fp)
            
            self.num_models = properties['num_models']
            self.top_ens_score = properties['top_ens_score']
            self.score = properties['score']
            self.out_models = properties['out_models']
            print("num_models: {}".format(self.num_models))
            print("top_ens_score: {}".format(self.top_ens_score))
            print("score: {}".format(self.score))
            
        self.LoadData()
        
        if os.path.isfile('{}/out_train.txt'.format(self.name)):
            with open('{}/out_train.txt'.format(self.name), "rb") as fp:
                self.out_train = pickle.load(fp)
                self.num_model = len(self.out_train)
            print("Load out_train weights, len: {}".format(len(self.out_train)))
                
        if os.path.isfile('{}/out_valid.txt'.format(self.name)):
            with open('{}/out_valid.txt'.format(self.name), "rb") as fp:
                self.out_valid = pickle.load(fp)
            print("Load out_valid weights, len: {}".format(len(self.out_valid)))
                
        if os.path.isfile('{}/out_test.txt'.format(self.name)):
            with open('{}/out_test.txt'.format(self.name), "rb") as fp:
                self.out_test = pickle.load(fp)
            print("Load out_test weights, len: {}".format(len(self.out_test)))
                
        if os.path.isfile('{}/weights.txt'.format(self.name)):
            with open('{}/weights.txt'.format(self.name), "rb") as fp:
                self.weights = pickle.load(fp)
            print("Load ensemble weights, len: {}".format(len(self.weights)))
        gc.collect()
        
        #Get accumulated predictions' score in training set (Cross-validation)
    def GetTrainScore(self):
        return self.score(self.y_train, self.GetEnsembleTrainPred())

    def GetEnsembleTrainPred(self):
        pred_ens = np.zeros(self.out_train[0].shape)
        for i, pred in enumerate(self.out_train):
            pred_ens += self.out_train[i] * self.weights[i]
        return pred_ens
    
    def GetEnsembleTestPred(self):
        pred_ens = np.zeros(self.out_test[0].shape)
        for i, pred in enumerate(self.out_test):
            pred_ens += self.out_test[i] * self.weights[i]
        return pred_ens

    def GetEnsembleSlice(self, idx):
        pred_ens = np.zeros(self.out_train[0].shape)
        for i, pred in enumerate(self.out_train):
            pred_ens += self.out_train[i] * self.weights[i]
        return pred_ens[idx]
        
    #Add a new prediction to the ensemble with its corresponding weight
    def AddOutput(self, out_train, out_test, weight = None, out_valid = None, model_name = 'Unnamed'):
        if self.out_train == None:
            self.out_train = []
            self.out_test = []
            self.weights = []
            if out_valid != None:
                self.out_valid = []
                
        #Auto weight
        if weight == None:
            ens_preds = self.GetEnsembleTrainPred()
            weights = np.logspace(start = -3, stop = -1, base = 10, num = 250)
            best_cv_ens_score = 0
            #Line search **********************************************
            for w in weights:
                new_cv_ens_score = roc_auc_score(self.y_train, (ens_preds + w * out_train))
                if new_cv_ens_score > best_cv_ens_score:
                    best_cv_ens_score = new_cv_ens_score
                    weight = w
            
            #Fine Tune ***********************************************
            up = False
            improve = True
            while improve:
                improve = False
                new_cv_ens_score = roc_auc_score(self.y_train, (ens_preds + 1.17 * weight * out_train))
                if new_cv_ens_score > best_cv_ens_score:
                    best_cv_ens_score = new_cv_ens_score
                    up = True
                    improve = True
                    weight = weight * 1.17

            if not up:
                improve = True
                while improve:
                    improve = False
                    new_cv_ens_score = roc_auc_score(self.y_train, (ens_preds + 0.9 * weight * out_train))
                    if new_cv_ens_score > best_cv_ens_score:
                        improve = True
                        best_cv_ens_score = new_cv_ens_score
                        weight = weight * 0.9
        
        self.weights.append(weight)
        added_valid = False
        if out_valid != None:
            added_valid = True
            self.out_valid.append(out_valid)
        self.out_train.append(out_train)
        self.out_test.append(out_test)
        self.num_models += 1
        
        new_score = self.GetTrainScore()
        if new_score <= self.top_ens_score:
            print("There must be an error somewhere!!!, remove back the model.")
            print("Old top score: {}, New supposedly top score: {}".format(self.top_ens_score, new_score))
            del self.weights[-1]
            del self.out_train[-1]
            del self.out_test[-1]
            self.num_models -= 1
            if added_valid:
                del self.out_valid[-1]
            return False
        else:
            print("New ensemble ouput added with weight: {}".format(weight))
            print("Old top score: {}, New top score: {}".format(self.top_ens_score, new_score))
            self.top_ens_score = new_score
            self.out_models.append(model_name)
            return True
    
    def LoadData(self):
        if os.path.isfile('{}/X_valid.csv'.format(self.name)):
            self.X_valid = pd.read_csv('{}/X_valid.csv'.format(self.name))
            print("Load X_valid ({})".format(self.X_valid.shape))
            
        if os.path.isfile('{}/y_valid.csv'.format(self.name)):
            self.y_valid = pd.Series(pd.read_csv('{}/y_valid.csv'.format(self.name), header=None).values.ravel()).astype('int')
            print("Load y_valid ({})".format(self.y_valid.shape))
            
        if os.path.isfile('{}/X_train.csv'.format(self.name)):
            self.X_train = pd.read_csv('{}/X_train.csv'.format(self.name))
            print("Load X_train ({})".format(self.X_train.shape))
                
        if os.path.isfile('{}/y_train.csv'.format(self.name)):
            self.y_train = pd.Series(pd.read_csv('{}/y_train.csv'.format(self.name), header=None).values.ravel()).astype('int')
            print("Load y_train ({})".format(self.y_train.shape))
            
        if os.path.isfile('{}/X_test.csv'.format(self.name)):
            self.X_test = pd.read_csv('{}/X_test.csv'.format(self.name))
            print("Load X_test ({})".format(self.X_test.shape))
            
        if os.path.isfile('{}/y_test.csv'.format(self.name)):
            self.y_test = pd.Series(pd.read_csv('{}/y_test.csv'.format(self.name), header=None).values.ravel()).astype('int')
            print("Load y_test ({})".format(self.y_test.shape))
    
    def GetPredictionScores(self):
        scores = []
        for train_pred in self.out_train:
            scores.append(roc_auc_score(self.y_train, train_pred))
        return scores
    
    def AddRandomSearch_NN(self, model, params, min_fold_score = 0, num_iter = 1):
        ens_score_improved = False
        folds = StratifiedKFold(n_splits=6, shuffle=True)
        scores = []
        best_param = None
        best_score = 0

        for _ in range(num_iter):
            param = dict()
            #Select random params
            for key in params:
                param[key] = np.random.choice(params[key])
            
            ens_weight =  param['ens_weight']
            print("Parameters: {}".format(param))

            #Train model and evaluate
            oof = np.zeros(len(self.X_train))
            predictions = np.zeros(len(self.X_test))
            preds_train = np.zeros(len(self.X_train))

            for fold_, (trn_idx, val_idx) in enumerate(folds.split(self.X_train.values, self.y_train)):
                print("Fold {}".format(fold_))

                train_fold_x = self.X_train.iloc[trn_idx]
                train_fold_y = self.y_train.iloc[trn_idx]
                valid_fold_x = self.X_train.iloc[val_idx]
                valid_fold_y = self.y_train.iloc[val_idx]

                ens_fold_x = self.GetEnsembleSlice(val_idx)
                print(ens_fold_x)

                K.clear_session()
                #early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto', baseline=None, restore_best_weights = True)
                if self.score != roc_auc_score:
                    print("Error: NN random search is only implemented for roc_auc_score.")
                auc_stop = roc_callback.RocAucMetricCallback(patience=param['roc_patience'], verbose=0)
                auc_stop.ens_weight = param['ens_weight']
                auc_stop.ens_fold_x = ens_fold_x
                #roc_cb = roc_callback(training_data=(train_data.iloc[trn_idx], y_train.iloc[trn_idx]),validation_data=(train_data.iloc[val_idx], y_train.iloc[val_idx]))

                clf = model(param['size'], param['dropout_rate'])
                opt = Adam(lr=param['lr'], beta_1=param['beta_1'], beta_2=param['beta_2'], epsilon=None, decay=param['decay'], amsgrad=param['amsgrad'])
                clf.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = None)
                clf.fit(x = train_fold_x,
                        y = train_fold_y,
                        verbose = 0,
                        epochs = param['epochs'],
                        batch_size = param['batch_size'],
                        validation_data = (valid_fold_x, valid_fold_y),
                        class_weight = param['class_weight'],
                        callbacks = [auc_stop])


                preds_train[val_idx] +=  clf.predict(valid_fold_x).flatten()
                score_fold = roc_auc_score(valid_fold_y,  preds_train[val_idx]) #ens_fold_x + param['ens_weight'] *
                print('Fold score(ens): {}'.format(score_fold))

                if score_fold < min_fold_score:
                    print('Skipping...')
                    break

                predictions += clf.predict(self.X_test).flatten() / folds.n_splits
                
            ens_weight = param['ens_weight']
            
            best_cv_ens_score = roc_auc_score(self.y_train, ( preds_train)) #self.GetEnsembleTrainPred() + ens_weight *
            """new_cv_ens_score = 999
            
            #Ensemble weight adjustment
            up = False
            improve = True
            while improve:
                new_cv_ens_score = roc_auc_score(self.y_train, (self.GetEnsembleTrainPred() + 1.6 * ens_weight * preds_train))
                if new_cv_ens_score > best_cv_ens_score:
                    best_cv_ens_score = new_cv_ens_score
                    up = True
                    improve = True
                    ens_weight= ens_weight * 1.4

            if not up:
                while improve:
                    new_cv_ens_score = roc_auc_score(self.y_train, (self.GetEnsembleTrainPred() + 0.7 * ens_weight * preds_train))
                    if new_cv_ens_score > best_cv_ens_score:
                        improve = True
                        best_cv_ens_score = new_cv_ens_score
                        ens_weight = ens_weight * 0.8
            """
            if best_cv_ens_score > best_score:
                best_score = best_cv_ens_score
                best_param = param
                print('Best cv score improved, params: {} ****************'.format(param))
                if best_cv_ens_score > self.top_ens_score:
                    ens_score_improved = True
                    print("*********** ENSEMBLE SCORE IMPROVED ***************///********************** ENSEMBLE SCORE IMPROVED *************")

            scores.append(best_cv_ens_score)
            print("CV ensemble score: {:<8.10f}".format(best_cv_ens_score))

        if ens_score_improved:
            self.AddOutput(weight = None, out_test = predictions, out_train = preds_train, model_name = str(model))
        else:
            print("Ensemble score did not improve, no new model.")


        return [best_score, best_param, scores, predictions, preds_train]

    
    def AddRandomSearch_NN_boost(self, model, params, min_fold_score = 0, num_iter = 1):
        ens_score_improved = False
        folds = StratifiedKFold(n_splits=6, shuffle=True, random_state = 12324)
        scores = []
        best_param = None
        best_score = 0
        y_train_orig = self.y_train.copy()
        
        #self.y_train = self.y_train.values.astype('float64').ravel()
        self.y_train -= pd.Series(self.GetEnsembleTrainPred().ravel())
        
        #Center target residuals
        #self.y_train[np.abs(self.y_train) > 0.8] = 0.0
        #self.y_train -= np.mean(self.y_train)
        
        y_train_max = np.max(self.y_train)
        y_train_min = np.min(self.y_train)
        print(y_train_max)
        print(y_train_min)
        if y_train_max > np.abs(y_train_min):
            scale = 1 / y_train_max
        else:
            scale = np.abs(1 / y_train_min)
        #scale y_train
        self.y_train *= scale
        
        
        
        for _ in range(num_iter):
            param = dict()
            #Select random params
            for key in params:
                param[key] = np.random.choice(params[key])
            
            ens_weight = param['ens_weight'] / scale
            param['ens_weight'] = ens_weight
            
            #No need to center afterwards, because we use AUC
             #param['ens_weight']1
            #param['ens_weight'] = params['ens_weight']
            print("Parameters: {}".format(param))

            #Train model and evaluate
            oof = np.zeros(len(self.X_train))
            predictions = np.zeros(len(self.X_test))
            preds_train = np.zeros(len(self.X_train))
            
            
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(self.X_train.values, y_train_orig)):
                print("Fold {}".format(fold_))

                train_fold_x = self.X_train.iloc[trn_idx]
                train_fold_y = self.y_train.iloc[trn_idx].values.ravel()
                valid_fold_x = self.X_train.iloc[val_idx]
                valid_fold_y = y_train_orig.iloc[val_idx].values.copy()

                ens_fold_y = self.GetEnsembleSlice(val_idx)
            
                K.clear_session()
                #early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto', baseline=None, restore_best_weights = True)
                if self.score != roc_auc_score:
                    print("Error: NN random search is only implemented for roc_auc_score.")
                auc_stop = roc_callback.RocAucMetricCallback(patience=param['roc_patience'], verbose=0)
                auc_stop.ens_weight = param['ens_weight']
                auc_stop.ens_fold_x = ens_fold_y
                #roc_cb = roc_callback(training_data=(train_data.iloc[trn_idx], y_train.iloc[trn_idx]),validation_data=(train_data.iloc[val_idx], y_train.iloc[val_idx]))

                clf = model(param['size'], param['dropout_rate'])
                opt = Adam(lr=param['lr'], beta_1=param['beta_1'], beta_2=param['beta_2'], epsilon=None, decay=param['decay'], amsgrad=param['amsgrad'])
                clf.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = None)
                clf.fit(x = train_fold_x,
                        y = train_fold_y,
                        verbose = 0,
                        epochs = param['epochs'],
                        batch_size = param['batch_size'],
                        validation_data = (valid_fold_x, valid_fold_y),
                        class_weight = param['class_weight'],
                        callbacks = [auc_stop])


                preds_train[val_idx] +=  clf.predict(valid_fold_x).flatten()
                score_fold = roc_auc_score(valid_fold_y, ens_fold_y + param['ens_weight'] * preds_train[val_idx])
                print('Fold score(ens): {}'.format(score_fold))

                if score_fold < min_fold_score:
                    print('Skipping...')
                    break

                predictions += clf.predict(self.X_test).flatten() / folds.n_splits
                
            ens_weight = param['ens_weight']
            
            self.y_train = y_train_orig.copy()
            
            best_cv_ens_score = roc_auc_score(self.y_train, (self.GetEnsembleSlice(list(self.X_train.index)) + param['ens_weight'] * preds_train))
            #new_cv_ens_score = 999
            
            if best_cv_ens_score > best_score:
                best_score = best_cv_ens_score
                best_param = param
                print('Best cv score improved, params: {} ****************'.format(param))
                if best_cv_ens_score > self.top_ens_score:
                    ens_score_improved = True
                    print("*********** ENSEMBLE SCORE IMPROVED ***************///********************** ENSEMBLE SCORE IMPROVED *************")

            scores.append(best_cv_ens_score)
            print("CV ensemble score: {:<8.10f}".format(best_cv_ens_score))

        if ens_score_improved & (ens_weight > 10**-5):
            self.AddOutput(weight = ens_weight, out_test = predictions, out_train = preds_train)
        else:
            print("Ensemble score did not improve, no new model.")

        
        return [best_score, best_param, scores, predictions, preds_train]
    
    def AddCVModel(self, model, param, num_runs = 1, fixed_folds = True, n_folds = 2, process_fold = None, fold_data = None):
        predictions_total = np.zeros(len(self.X_test))
        preds_train_total = np.zeros(len(self.X_train))
        
        #Separate parameters into two
        cv_param_keys = ['ens_weight', 'has_eval_set', 'early_stopping', 'verbose']


        cv_params = dict()
        for cv_param in cv_param_keys:
            if not cv_param in list(param.keys()):
                print("CV parameter [{}] is required.".format(cv_param))
            cv_params[cv_param] = param[cv_param]
            param.pop(cv_param, None)

        print("parameters: {}".format(param))
        
        ### RUNS ##############################################
        for i_run in range(num_runs):
            print('###> Run {}/{} <##################################'.format(i_run, num_runs))
            print('Best ensemble score: {}'.format(self.top_ens_score))
            folds = None
            if fixed_folds:
                folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=99999)
            else:
                folds = StratifiedKFold(n_splits=n_folds, shuffle=True)
                
            cv_scores = []
            best_score = 0
            preds_train = np.zeros(len(self.X_train))
            predictions = np.zeros(len(self.X_test))
            
            print(self.y_train.values)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(self.X_train.values, self.y_train.values)):
                clf = model(**param)
                ens_fold_x = self.GetEnsembleSlice(val_idx)
                
                train_fold_x = None
                train_fold_y = None
                
                print("Fold {}".format(fold_))
                if fold_data == None:
                    train_fold_x = self.X_train.iloc[trn_idx]
                    train_fold_y = self.y_train.iloc[trn_idx].values.ravel().copy()
                else:
                    train_fold_x = fold_data[fold_][0]
                    train_fold_y = fold_data[fold_][1]
                
                valid_fold_x = self.X_train.iloc[val_idx]
                valid_fold_y = self.y_train.iloc[val_idx].values.ravel().copy()
                
                """num_negative = len(train_fold_x[train_fold_x == 0])
                print("Oversampling...")
                train_fold_x, train_fold_y = SMOTE(sampling_strategy = {0: num_negative, 1:int(num_negative * 0.3)}).fit_resample(train_fold_x, train_fold_y)
                print(len(train_fold_y[train_fold_y == 0]))
                print(len(train_fold_y[train_fold_y == 1]))
                print("Oversampling done.")"""
                #if process_fold != None:
                #    train_fold_x, train_fold_y, valid_fold_x, self.X_test = process_fold(train_fold_x, train_fold_y, valid_fold_x, self.X_test)
                
                

                if cv_params['has_eval_set']:
                    clf.fit(train_fold_x, train_fold_y, eval_set = [(valid_fold_x, valid_fold_y)], early_stopping_rounds = cv_params['early_stopping'], verbose = cv_params['verbose'])
                else:
                    clf.fit(train_fold_x, train_fold_y)
                    print("Warning: Early stopping was not used! (Set [has_eval_set=1] if model allows.)")

                preds_train[val_idx] = clf.predict_proba(valid_fold_x)[:,1]
                score_fold_ens = roc_auc_score(valid_fold_y, ens_fold_x + cv_params['ens_weight'] * preds_train[val_idx])
                score_fold = roc_auc_score(valid_fold_y, preds_train[val_idx])

                score_fold = np.abs(score_fold - 0.5) + 0.5

                cv_scores.append(score_fold)
                print('fold score: {}'.format(score_fold))
                print('fold score (ens): {}'.format(score_fold_ens))

                # / folds.n_splits
                predictions += clf.predict_proba(self.X_test)[:,1] / folds.n_splits
            
            predictions_total += predictions / num_runs
            preds_train_total += preds_train / num_runs
            
        ens_weight = cv_params['ens_weight']
        cv_ens_score = roc_auc_score(self.y_train, (self.GetEnsembleTrainPred() + ens_weight * preds_train_total))
        cv_score = roc_auc_score(self.y_train, preds_train_total)

        cv_params['ens_weight'] = ens_weight
        #Ensemble weight adjustment --------------------------------------------------------------------------------
        #print("CV ensemble score: {:<8.10f}".format(best_cv_ens_score))
        #print("CV score: {:<8.10f}".format(cv_score))
        #print("Ens weight: {}".format(ens_weight))
        
        added = None
        if cv_ens_score > self.top_ens_score:
            ens_score_improved = True
            added = self.AddOutput(out_test = predictions_total, out_train = preds_train_total, model_name = str(model) +'-'+ str(param)) #weight = cv_param_keys, 
            print("*********** ENSEMBLE SCORE IMPROVED ***************///********************** ENSEMBLE SCORE IMPROVED *************")
        else:
            print("Ensemble score did not improve, no new model.")
        
        cv_score = np.abs(cv_score - 0.5) + 0.5
        
        return [cv_ens_score, cv_score, predictions_total, preds_train_total, added]
    
    def AddBayesianSearchModel(self, model, parameters, fixed_parameters, tpe_trials, num_iter = 1, improve_ens = False, num_runs = 1, fixed_folds = True, n_folds = 2, fold_data = None):
        space  = parameters
        cv_scores = []
        
        if len(tpe_trials.results) != 0:
            cv_scores = [- ex['loss'] for ex in tpe_trials.results]
        
        #Don't count previous trials
        num_iter += len(tpe_trials.results)
        
        preds = None
        preds_train = None
        highest_cv_score = 0
        best_params = None
        prev_params = pd.DataFrame()
        prev_params.index += 1
        fold_data = fold_data
        
        #@use_named_args(space)
        def objective(params): #**params
            nonlocal cv_scores
            nonlocal model
            nonlocal preds
            nonlocal preds_train
            nonlocal highest_cv_score
            nonlocal best_params
            nonlocal fixed_parameters
            nonlocal prev_params
            nonlocal fold_data
            
            #Combine fixed and variable parameters
            params_comb = params.copy()
            params_comb.update(fixed_parameters)
            
            [cv_score_ens, cv_score, preds_te, preds_tr, added] = self.AddCVModel(model, params_comb, num_runs = num_runs, fixed_folds = fixed_folds, n_folds = n_folds, fold_data = fold_data)
            
            #Use ensemble score for bayesian search (May not converge)
            if improve_ens:
                cv_score = cv_score_ens
            
            params['score'] = cv_score
            """prev_params.iloc[-1] = (params)
            prev_params.index += 1
            prev_params.sort_index()"""
            
            cv_scores.append(cv_score)
            if cv_score > highest_cv_score:
                best_params = params
                highest_cv_score = cv_score
                preds_train = preds_tr
                preds = preds_te
                
            clear_output()
            plt.figure(figsize = (15,10))
            plt.show(sns.regplot(list(range(len(cv_scores))),cv_scores))
            #prev_params.sort_values(by = 'score').head(10)
            print("Prev model added to ens: {}".format(added))
            print("# Models in ens: {}".format(self.num_models))
            print('best params: {}'.format(best_params))
            print('best cv score: {}'.format(highest_cv_score))
            return - cv_score

        #res_gp = gp_minimize(objective, space, n_calls=num_iter, random_state=14568)
        tpe_algo = tpe.suggest
        
        tpe_best = fmin(fn=objective, space=space, 
                algo=tpe_algo, trials=tpe_trials, 
                max_evals=num_iter)
        return [tpe_trials, tpe_best]
    
    
