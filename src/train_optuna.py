
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import Xception

from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from transformers import TFAutoModelForImageClassification


import torch
import torch.nn as nn
import timm

import matplotlib.pyplot as plt

import functools
import optuna
from optuna.samplers import TPESampler
import csv
#from utils.plot_utils import plot_results
import os

import common




def get_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'SGD':
        return optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == 'Adam':
        return optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'Adamax':
        return optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_name == 'Nadam':
        return optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")



def create_model(input_tensor, model_params):
    """
    モデルを作成する関数。Optunaに最適化してもらうハイパーパラメータを受け取る。
    """
    print("\n### model def ###\n")
    

   
    model_name = model_params['model_name']
    print(f"model_name: {model_name}")

    dense_units = model_params['dense_units']
    dropout_rate = model_params['dropout_rate']
    activation_func = model_params['activation_func']
    optimizer = get_optimizer(model_params['optimizer_func'], model_params['learning_rate'])
    fix_layer = model_params['fix_layer']

    if model_name == 'ViT':
        # 入力テンソル
        inputs = Input(shape=(common.IMG_HEIGHT, common.IMG_WIDTH, 3), name="input_image")

        # Hugging Face ViTモデル（分類ヘッド付き）
        base_model = TFAutoModelForImageClassification.from_pretrained(
            common.VIT_MODEL_NAME, from_pt=True
        )

        base_model.trainable = True

        # 出力(logits)を取得
        outputs = base_model(inputs, training=False)
        x = outputs.logits  # shape=(batch, hidden_size)

        # 必要に応じて Dense/Dropout を追加
        if model_params['num_dense_layers'] > 0:
            for _ in range(model_params['num_dense_layers']):
                x = Dense(dense_units, activation=activation_func)(x)
                x = Dropout(dropout_rate)(x)

        # 最終出力
        x = Dense(common.NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)

    else:
        # 通常のKerasモデル
        if model_name == 'VGG16':
            base_model_class = VGG16
        elif model_name == 'VGG19':
            base_model_class = VGG19
        elif model_name == 'ResNet50':
            base_model_class = ResNet50
        elif model_name == 'Xception':
            base_model_class = Xception
        else:
            raise ValueError(f"Unsupported base model: {model_name}")

        base_model = base_model_class(include_top=False, weights='imagenet', input_tensor=input_tensor)

        x = tf.keras.layers.Flatten()(base_model.output)
        if model_params['num_dense_layers'] > 0:
            for _ in range(model_params['num_dense_layers']):
                x = Dense(dense_units, activation=activation_func)(x)
                x = Dropout(dropout_rate)(x)
        x = Dense(common.NUM_CLASS, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)



    # ベースモデルの重み固定
    if common.ENABLE_FIX_LAYER:
        fix_layer_set = -1 * fix_layer
        if fix_layer_set == 0:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for layer in base_model.layers[:fix_layer_set]:
                layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = False

    # コンパイル
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return model




def objective(trial, param1, param2, param3, param4):
    """
    Optunaの最適化関数。ここで定義されたハイパーパラメータが最適化される。
    """
    #suggest_discrete_uniform() -> suggest_int, suggest_float
    
    #model_params = {
    #    'model_name': trial.suggest_categorical('model_name', ['VGG16', 'VGG19', 'ResNet50', 'Xception']),  # モデルを選択
    #    'num_dense_layers': trial.suggest_int('num_dense_layers', 0, 2, step=1),  # Dense層の数を1から2で選択
    #    'dense_units': trial.suggest_int('dense_units', 64, 128, step=64),  # Dense層のユニット数を64から512の範囲で選択
    #    'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.5, step=0.1),  # Dropout率を0.1から0.5の範囲で選択
    #    'activation_func': trial.suggest_categorical('activation_func', ['relu', 'sigmoid']),  # 活性化関数を選択
    #    'optimizer_func': trial.suggest_categorical('optimizer_func', ['SGD', 'Adam', 'Adamax', 'Nadam']), # 最適化関数を選択
    #    #'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-1)  # 学習率を対数スケールで最適化
    #    'learning_rate': trial.suggest_categorical('learning_rate', [1e-6, 1e-5, 1e-4, 1e-3]),  # 学習率を指定された値から選択
    #    'fix_layer': trial.suggest_int('fix_layer', 0, 10, step=1)  # 重み固定のレイヤー指定(最終層から学習させる層を指定. -をかけてスライス指定.)
    #    #'epochs': 20  # エポック数（固定にしても良い）
    #}
    
    
    model_params = {
        #'model_name': trial.suggest_categorical('model_name', ['VGG16','Xception']),  # モデルを選択
        #'model_name': trial.suggest_categorical('model_name', ['VGG16']),  # モデルを選択
        'model_name': trial.suggest_categorical('model_name', ['ViT']),  # モデルを選択
        'num_dense_layers': trial.suggest_int('num_dense_layers', 0, 0),  # Dense層の数を1から2で選択
        'dense_units': trial.suggest_int('dense_units', 64, 64),  # Dense層のユニット数を64から512の範囲で選択
        'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.3),  # Dropout率を0.1から0.5の範囲で選択
        'activation_func': trial.suggest_categorical('activation_func', ['relu']),  # 活性化関数を選択
        'optimizer_func': trial.suggest_categorical('optimizer_func', ['Nadam']), # 最適化関数を選択
        #'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-1)  # 学習率を対数スケールで最適化
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-3]),  # 学習率を指定された値から選択
        'fix_layer': trial.suggest_int('fix_layer', 2, 2)  # 重み固定のレイヤー指定(最終層から学習させる層を指定. -をかけてスライス指定.)
        #'epochs': 20  # エポック数（固定にしても良い）
    }
    
    
    
    X_batch = param1
    Y_batch = param2
    X_val = param3
    Y_val = param4
    
    #print(X_batch.shape)
    #print(Y_batch.shape)
    #print(X_val.shape)
    #print(Y_val.shape)
    
    input_tensor = Input(shape=(common.IMG_WIDTH, common.IMG_HEIGHT, 3))
    model = create_model(input_tensor, model_params)

    # コールバック設定
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f"{common.RESULT_PATH}best_model.h5", save_best_only=True, monitor='val_loss', mode='min')



    # 学習
    if common.ENABLE_CALLBACKS:
        history = model.fit(
            X_batch,
            Y_batch,
            batch_size=common.BATCH_SIZE,
            epochs=common.EPOCHS,
            validation_data=(X_val, Y_val),
            callbacks=[early_stopping, model_checkpoint]
        )
    else:
        history = model.fit(
            X_batch,
            Y_batch,
            batch_size=common.BATCH_SIZE,
            epochs=common.EPOCHS,
            validation_data=(X_val, Y_val)
        )


    #print(history.history.keys())
    history = {'loss': history.history['loss'], 'accuracy': history.history['accuracy'], 'val_loss': history.history['val_loss'], 'val_accuracy': history.history['val_accuracy']}
    
    trial.set_user_attr('history', history)  # history を保存
    
    # バリデーション精度を最適化する
    return history['val_accuracy'][-1]  # 最後のエポックのバリデーション精度を返す
    
    
    
    ## 最良のモデルの履歴（最良のエポックの履歴）を保存
    #history_dict = history.history  # 履歴を取得
    #
    ## 最良のエポックの情報のみを抽出
    #best_epoch = min(range(len(history_dict['val_loss'])), key=lambda i: history_dict['val_loss'][i])
    #best_history = {key: values[best_epoch] for key, values in history_dict.items()}
    #
    ## 履歴をpandasのDataFrameに変換して保存
    #str_time = time.strftime("%Y%m%d_%H%M%S")
    #history_df = pd.DataFrame([best_history])
    #history_df.to_csv(f"{common.OUTPUT_PATH}best_history_{str_time}.csv", index=False)
    #
    #print(f"Best model history saved to {common.OUTPUT_PATH}best_history_{str_time}.csv")
    #
    ## バリデーション精度を最適化する
    #return best_history['val_accuracy']  # 最良のバリデーション精度を返す





def plot_results(histories, output_path):
    """
    複数のモデルパラメータで得られた精度と損失をグラフ化し保存する関数。
    """
    print("#plot_results")
    for model_name, history in histories.items():
        # グラフ作成
        plt.figure(figsize=(12, 6))
        
        # 精度のグラフ
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='train accuracy')
        plt.plot(history['val_accuracy'], label='val accuracy')
        plt.title(f'{model_name} - Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        # 横軸の目盛りを整数に設定
        plt.xticks(range(0, len(history['accuracy']), 1))  # Epoch数に合わせて整数の目盛りを設定
        plt.legend()

        # 損失のグラフ
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='train loss')
        plt.plot(history['val_loss'], label='val loss')
        plt.title(f'{model_name} - Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # 横軸の目盛りを整数に設定
        plt.xticks(range(0, len(history['accuracy']), 1))  # Epoch数に合わせて整数の目盛りを設定
        plt.legend()

        # 画像として保存
        plt.tight_layout()
        graph_path = os.path.join(output_path, f'{model_name}_learning_curve.png')
        plt.savefig(graph_path)
        print(f"Graph saved at {graph_path}")
        plt.close()





def train_optuna(objective, X_batch, Y_batch, X_val, Y_val):
    """
    Optunaのハイパーパラメータ最適化を実行する関数。

    Parameters:
        objective : function
            評価関数 (trialを受け取って目的関数値を返す)
        X_batch, Y_batch, X_val, Y_val :
            データセット
    Returns:
        best_params : dict
            最良パラメータ
        history : dict
            最良試行の訓練履歴
        study : optuna.Study
            完全なOptunaスタディオブジェクト
    """





    # partialで引数を固定
    objective_with_param = functools.partial(
        objective,
        param1=X_batch,
        param2=Y_batch,
        param3=X_val,
        param4=Y_val
    )

    print("#Optuna 1: スタディを作成")
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    print("#Optuna 2: 最適化を開始")

    # n_trials回の試行で最適化
    study.optimize(objective_with_param, n_trials=common.OPTUNA_N_TRIALS)
    # 並列実行する場合:
    # study.optimize(objective_with_param, n_trials=common.OPTUNA_N_TRIALS, n_jobs=4)

    print(f"Best hyperparameters: {study.best_params}")

    os.makedirs(common.OUTPUT_PATH, exist_ok=True)

    # 最良パラメータを保存
    with open(os.path.join(common.OUTPUT_PATH, "log__study.best_params.txt"), "w") as file:
        file.write(f"Best hyperparameters: {study.best_params}\n")

    # 各試行のパラメータと値を保存
    with open(os.path.join(common.OUTPUT_PATH, "log__study.params_value.txt"), "w") as file:
        for t in study.trials:
            file.write(f"{t.params}: {t.value}\n")

    # 最良トライアルの履歴を取得して可視化
    if 'history' in study.best_trial.user_attrs:
        history = study.best_trial.user_attrs['history']
        plot_results({'best_model': history}, common.OUTPUT_PATH)
    else:
        history = None

    # 各試行をCSVに出力
    csv_path = os.path.join(common.OUTPUT_PATH, "log__study_results.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['trial_number', 'value'] + [f'param_{key}' for key in study.best_params.keys()]
        writer.writerow(header)
        for t in study.trials:
            row = [t.number, t.value] + [t.params.get(key, 'N/A') for key in study.best_params.keys()]
            writer.writerow(row)

    print(f"Optuna結果を保存しました: {csv_path}")

    return study.best_params, history, study





