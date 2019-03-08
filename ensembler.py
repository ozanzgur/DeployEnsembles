
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import roc_callback

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pickle
import os.path
import gc

import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

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
        scaler = preprocessing.StandardScaler().fit(self.X_train)

        self.train_data = pd.DataFrame(scaler.transform(self.X_train), columns = cols)
        self.test_data = pd.DataFrame(scaler.transform(self.X_test), columns = cols)
        if self.valid_data is not None:
            self.valid_data = pd.DataFrame(scaler.transform(self.valid_data), columns = cols)
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
            score = self.score
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
    def AddOutput(self, weight, out_train, out_test, out_valid = None):
        if self.out_train == None:
            self.out_train = []
            self.out_test = []
            self.weights = []
            if out_valid != None:
                self.out_valid = []
                
        added_valid = False
        if out_valid != None:
            added_valid = True
            self.out_valid.append(out_valid)
        self.out_train.append(out_train)
        self.out_test.append(out_test)
        self.weights.append(weight)
        self.num_models += 1

        new_score = self.GetTrainScore()
        if new_score <= self.top_ens_score:
            print("There must be an error somewhere!!!, remove back the model.")
            print("Old top score: {}, New supposedly top score: {}".format(self.top_ens_score, new_score))
            del self.weights[-1]
            del self.out_train[-1]
            del self.out_test[-1]
            if added_valid:
                del self.out_valid[-1]
        else:
            print("New ensemble ouput added with weight: {}".format(weight))
            print("Old top score: {}, New top score: {}".format(self.top_ens_score, new_score))
            self.top_ens_score = new_score
    
    def LoadData(self):
        if os.path.isfile('{}/X_valid.csv'.format(self.name)):
            self.X_valid = pd.read_csv('{}/X_valid.csv'.format(self.name))
            print("Load X_valid ({})".format(self.X_valid.shape))
            
        if os.path.isfile('{}/y_valid.csv'.format(self.name)):
            self.y_valid = pd.read_csv('{}/y_valid.csv'.format(self.name), header=None)
            print("Load y_valid ({})".format(self.y_valid.shape))
            
        if os.path.isfile('{}/X_train.csv'.format(self.name)):
            self.X_train = pd.read_csv('{}/X_train.csv'.format(self.name))
            print("Load X_train ({})".format(self.X_train.shape))
                
        if os.path.isfile('{}/y_train.csv'.format(self.name)):
            self.y_train = pd.read_csv('{}/y_train.csv'.format(self.name), header=None)
            print("Load y_train ({})".format(self.y_train.shape))
            
        if os.path.isfile('{}/X_test.csv'.format(self.name)):
            self.X_test = pd.read_csv('{}/X_test.csv'.format(self.name))
            print("Load X_test ({})".format(self.X_test.shape))
            
        if os.path.isfile('{}/y_test.csv'.format(self.name)):
            self.y_test = pd.read_csv('{}/y_test.csv'.format(self.name), header=None)
            print("Load y_test ({})".format(self.y_test.shape))
        
    def AddRandomSearch_NN(self, model, params, min_fold_score = 0, num_iter = 1):
        ens_score_improved = False
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=73165)
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

            for fold_, (trn_idx, val_idx) in enumerate(folds.split(self.X_train.values, self.y_train.values)):
                print("Fold {}".format(fold_))

                train_fold_x = self.X_train.iloc[trn_idx]
                train_fold_y = self.y_train.iloc[trn_idx]
                valid_fold_x = self.X_train.iloc[val_idx]
                valid_fold_y = self.y_train.iloc[val_idx]

                ens_fold_x = self.GetEnsembleSlice(val_idx)

                K.clear_session()
                #early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto', baseline=None, restore_best_weights = True)
                if self.score != roc_auc_score:
                    print("Error: NN random search is only implemented for roc_auc_score.")
                auc_stop = roc_callback.RocAucMetricCallback(patience=param['roc_patience'], verbose=0)
                auc_stop.ens_weight = param['ens_weight']
                auc_stop.ens_fold_x = ens_fold_x
                #roc_cb = roc_callback(training_data=(train_data.iloc[trn_idx], y_train.iloc[trn_idx]),validation_data=(train_data.iloc[val_idx], y_train.iloc[val_idx]))

                clf = model(param['size'])
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
                score_fold = roc_auc_score(valid_fold_y, ens_fold_x + param['ens_weight'] * preds_train[val_idx])
                print('Fold score(ens): {}'.format(score_fold))

                if score_fold < min_fold_score:
                    print('Skipping...')
                    break

                predictions += clf.predict(self.X_test).flatten() / folds.n_splits
                
            ens_weight = param['ens_weight']
            
            best_cv_ens_score = roc_auc_score(self.y_train, (self.GetEnsembleTrainPred() + ens_weight * preds_train))
            new_cv_ens_score = 999
            
            #Ensemble weight adjustment
            up = False
            while new_cv_ens_score > best_cv_ens_score:
                new_cv_ens_score = roc_auc_score(self.y_train, (self.GetEnsembleTrainPred() + 2 * ens_weight * preds_train))
                if new_cv_ens_score > best_cv_ens_score:
                    best_cv_ens_score = new_cv_ens_score
                    up = True
                    ens_weight= ens_weight * 2

            if not up:
                while new_cv_ens_score > best_cv_ens_score:
                    new_cv_ens_score = roc_auc_score(self.y_train, (self.GetEnsembleTrainPred() + 0.5 * ens_weight * preds_train))
                    if new_cv_ens_score > best_cv_ens_score:
                        best_cv_ens_score = new_cv_ens_score
                        ens_weight = ens_weight * 0.5
                
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
            self.add_output(weight = param['ens_weight'], out_test = predictions, out_train = preds_train)
        else:
            print("Ensemble score did not improve, no new model.")


        return [best_score, best_param, scores, predictions, preds_train]

