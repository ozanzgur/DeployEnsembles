##################################################################################################

# Code was borrowed from this kaggle kernel, then modified for returning the best model after early stopping
# https://www.kaggle.com/rspadim/gini-keras-callback-earlystopping-validation/code

##################################################################################################

class RocAucMetricCallback(Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=1024):
        
        # FROM EARLY STOP
        super(RocAucMetricCallback, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size=predict_batch_size
    
    #def on_batch_begin(self, batch, logs={}):
    
    #def on_batch_end(self, batch, logs={}):
    
    def on_train_begin(self, logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
    
    #def on_train_end(self, logs={}):
    
    #def on_epoch_begin(self, epoch, logs={}):

    def on_epoch_end(self, epoch, logs={}):
        if(self.validation_data):
            y_hat_val=self.model.predict(self.validation_data[0],batch_size=self.predict_batch_size)
        
        # FROM EARLY STOP
        if(self.validation_data):
            if (self.verbose == 1):
                print("\n AUC Callback:",roc_auc_score(self.validation_data[1], y_hat_val))
            current = roc_auc_score(self.validation_data[1], y_hat_val)
            
            
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
