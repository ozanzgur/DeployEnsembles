## DeployKerasModels

The main purpose of this library is to make ensemble building process less time consuming for the user. You can
test many models and combine their results with minimum effort. In addition, this library selects models directly based on
their contribution to the ensemble.

Right now, it only creates NN ensembles with AUC loss. Creating ensembles with AUC loss is easy. The reason for that is
mean, min, max values of predictions are not considered by AUC loss. Relative weights of the predictions is the only 
thing that matters for linear combination of predictions if you are using AUC.

## How does it work?
- A class stores, saves and loads all data, keeps ensemble predictions and their weights with one line of code.
When training a new model for the ensemble, that class handles most of the repetitive task.

Keras NN models are ensembled using random search for optimization parameters. 5-Fold Stratified CV is used. However,
score is tested after each epoch and fold using the ensemble predictions in CV. Therefore, models are selected based on their
improvements to the total ensemble score. In that sense, it resembles boosting.



- I will be adding new features soon. (Sklearn methods, Bayesian search, Automatic model selection, etc.)
