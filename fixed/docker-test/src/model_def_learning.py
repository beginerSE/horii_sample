
#
# import
#
import os
import time
import math
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.models import Model, Sequential #, load_model
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
import functools
#from sklearn.model_selection import GridSearchCV
#from sklearn.base import BaseEstimator, ClassifierMixin
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import TopKCategoricalAccuracy

#
# common
#
import common

#
# model
#
#from tensorflow.keras.applications import VGG16
#from tensorflow.keras.applications import VGG19
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.applications import Xception
#from transformers import ViTFeatureExtractor, ViTImageProcessor, TFAutoModel
#from transformers import AutoModel

#from transformers import TFAutoModelForImageClassification

#
# 
#
from train_single import train_single
from train_gridsearch import train_gridsearch
from create_model_gridsearch import create_model_gridsearch 
from train_optuna import train_optuna, objective



#--------------------------------------------------------------------------------------------
#
# model_main
#
#--------------------------------------------------------------------------------------------

def model_main(X_batch, Y_batch, X_val, Y_val):
    print("#model_main")
    
    # --------------------------------------------------------------------------------------------
    # OPTUNA
    # --------------------------------------------------------------------------------------------
    if common.SW_OPTUNA:
    
        print("train_optuna")
        best_params, history, study = train_optuna(objective, X_batch, Y_batch, X_val, Y_val)

        model = None
    
    else:
    
        if common.SW_GRIDSEARCH:
            # --------------------------------------------------------------------------------------------
            # GRIDSEARCH
            # --------------------------------------------------------------------------------------------
            print("Grid Search")

            #Y_batch = np.argmax(Y_batch, axis=1)

            model, history, grid_search = train_gridsearch(
                X_batch,
                Y_batch,
                create_model_gridsearch
                )
            
        else:
            # --------------------------------------------------------------------------------------------
            # 1つのモデル, 1つのパラメータ組み合わせ
            # --------------------------------------------------------------------------------------------
            print("train_single")

            model, history = train_single(X_batch, Y_batch, X_val, Y_val)



    return model, history


