##################################################################################################

# Code was borrowed from this kaggle kernel, then modified for returning the best model after early stopping
# https://www.kaggle.com/rspadim/gini-keras-callback-earlystopping-validation/code

##################################################################################################


# coding: utf-8

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import numpy as np

class RocAucMetricCallback(Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=1024):
        #print("self vars: ",vars(self))  #uncomment and discover some things =)
        
        # FROM EARLY STOP
        super(RocAucMetricCallback, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size=predict_batch_size
        self.ens_fold_x = None
        self.ens_weight = None
    
    #def on_batch_begin(self, batch, logs={}):
    
    #def on_batch_end(self, batch, logs={}):
    
    def on_train_begin(self, logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        # :(
        global ens_fold_x
        global ens_weight
        if(self.validation_data):
            y_hat_val=self.model.predict(self.validation_data[0],batch_size=self.predict_batch_size)
            
        # FROM EARLY STOP
        if(self.validation_data):
            current = roc_auc_score(self.validation_data[1], y_hat_val.flatten()*self.ens_weight + self.ens_fold_x)
            if (self.verbose == 1):
                print("\n AUC Callback(ens):",current)
            
            
            
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
                self.model.save_weights("callback_temp_weights.h5")
                
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    
                    self.model.load_weights("callback_temp_weights.h5")

