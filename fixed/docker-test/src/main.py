#!/usr/bin/python3.11
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
print("環境変数",os.environ["TF_USE_LEGACY_KERAS"])


#
# import
#
import os
# parser
import argparse
# time
import time
# datetime
from datetime import datetime

import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model


#
# common
#
import common

#
# config
#
from config import initialize_config

#
# data_preprocessing
#
from data_preprocessing import data_set

#
# model_def_learning
#
#from model_def_learning import model_def_learning
from model_def_learning import model_main

#
# prediction
#
from prediction import prediction

#
# confusion_matrix_exe
#
from confusion_matrix_exe import confusion_matrix_exe

#
# graph_disp
#
from graph_disp import graph_disp

#
# grad_cam
#
from grad_cam import grad_cam

#
# log_write
#
from log_write import log_write




#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
def main():


    # 現在の日時を取得
    now = datetime.now()
    # ファイル名用
    str_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    #
    # 時間計測スタート
    #
    start_time = time.time()




    #
    # parser, debug print
    #
    parser = argparse.ArgumentParser(description='Run the program by specifying the configuration file')
    parser.add_argument('config_file', type=str, help="設定ファイルのパス")
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug prints')

    # 引数を解析
    args = parser.parse_args()

    common.debug_print("Debug mode is enabled", args.debug)



    #
    # 設定値取得
    #
    print("\n### initialize_config ###\n")
    # 引数で指定された設定ファイルとデバッグオプションを渡す
    initialize_config(args.config_file, args.debug)

    # 読み込んだ設定を確認
    #print("main")
    #print(f"common.FOLDER_STRUCTURE: {common.FOLDER_STRUCTURE}")


    #
    # データ、ラベル準備
    #
    print("\n### data prepare ###\n")
    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_batch, Y_batch = data_set(common.FOLDER_STRUCTURE,
                                                                                common.DATASET_MODE,
                                                                                common.TRAIN_PATH,
                                                                                common.VAL_PATH,
                                                                                common.TEST_PATH,
                                                                                common.NUM_OF_TRAIN,
                                                                                common.NUM_OF_VAL,
                                                                                common.NUM_OF_TEST,
                                                                                common.SW_DATA_EXTENSION)


    print("X_train.shape:  ", X_train.shape)
    print("Y_train.shape:  ", Y_train.shape)
    print("X_val.shape:    ", X_val.shape)
    print("Y_val.shape:    ", Y_val.shape)
    print("X_test.shape:   ", X_test.shape)
    print("Y_test.shape:   ", Y_test.shape)
    print("X_batch.shape:  ", X_batch.shape)
    print("Y_batch.shape:  ", Y_batch.shape)





    #
    # モデル定義、コンパイル
    # 学習
    #
    print("\n### model def, compile, learning ###\n")
    #model, history = model_def_learning(X_batch,
    #                                    Y_batch,
    #                                    X_val,
    #                                    Y_val,
    #                                    str_time,
    #                                    common.OUTPUT_PATH,
    #                                    common.RESULT_PATH)



    """
    # for GRIDSEARCH
    # for SW_OPTUNA
    if common.SW_GRIDSEARCH or common.SW_OPTUNA:
        print("Transposing input to channels_last format for Keras")
        X_batch = np.transpose(X_batch, (0, 2, 3, 1))
        X_val = np.transpose(X_val, (0, 2, 3, 1))
        X_test = np.transpose(X_test, (0, 2, 3, 1))
    """
    """
    if common.SW_GRIDSEARCH:
        print("Transposing input to channels_last format for Keras")
        X_batch = np.transpose(X_batch, (0, 2, 3, 1))
        X_val = np.transpose(X_val, (0, 2, 3, 1))
        X_test = np.transpose(X_test, (0, 2, 3, 1))
    """
    
    model, history = model_main(X_batch, Y_batch, X_val, Y_val)


    if common.SW_OPTUNA:
        # ベストモデルをロード
        # OPTUNAのときだけ
        model = load_model(f"{common.RESULT_PATH}best_model.h5")





    #
    # outputディレクトリ作成
    #
    print("\n### output directory ###\n")
    #"OUTPUT_PATH": "./output/",
    # 最後のディレクトリ名を取り出す
    directory_name = os.path.basename(os.path.normpath(common.OUTPUT_PATH))
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)


    #
    # historyの保存
    #
    print("\n### history save ###\n")
    if common.SW_OPTUNA or common.SW_GRIDSEARCH:
        #OPTUNAのときhistoryが想定した形でないため
        #グリッドサーチのときhistoryが想定した形でないため
        print("history is not saved. (by OPTUNA or GRIDSEARCH)")
    else:
        # history.history は損失や精度などを保持した辞書
        history_dict = history.history

        # pandasのDataFrameに変換
        history_df = pd.DataFrame(history_dict)

        # CSVファイルとして保存
        history_df.to_csv(f"{common.OUTPUT_PATH}history__{str_time}.csv", index=False)



    #
    # 精度の評価
    #
    scores = None
    #if common.SW_OPTUNA:
    #    #OPTUNAのとき想定した形でないため
    #    print("evaluate is not done. (by OPTUNA)")
    #else:
    #    if(common.SW_EVALUATION):
    #        print("\n### evaluation ###\n")
    #        scores = model.evaluate(X_test, Y_test, verbose=1)
    #        print('Test loss    :', scores[0])
    #        print('Test accuracy:', scores[1])

    if(common.SW_EVALUATION):
        print("\n### evaluation ###\n")
        scores = model.evaluate(X_test, Y_test, verbose=1)
        print('Test loss    :', scores[0])
        print('Test accuracy:', scores[1])




    #
    # loss,acc グラフ表示
    #
    
    if common.SW_OPTUNA or common.SW_GRIDSEARCH:
        #OPTUNAのときhistoryが想定した形でないため
        #グリッドサーチのときhistoryが想定した形でないため
        print("graph_disp is not done. (by OPTUNA or GRIDSEARCH)")
    else:
        if(common.SW_LEARNING and common.SW_GRAPH_DISP):
            print("\n### graph_disp ###\n")
            graph_disp(history,
                       str_time,
                       common.SW_PLT_SHOW_graph_disp,
                       common.OUTPUT_PATH)




    #
    # 推論・予測（NGデータ取得）
    #
    #if common.SW_OPTUNA:
    #    #OPTUNAのとき想定した形でないため
    #    print("prediction is not done. (by OPTUNA)")
    #else:
    #    if(common.SW_PREDICTION):
    #        print("\n### prediction ###\n")
    #        predicted_classes, true_classes = prediction(model, 
    #                                                     str_time, 
    #                                                     common.FOLDER_STRUCTURE, 
    #                                                     common.OUTPUT_PATH)

    if(common.SW_PREDICTION):
        print("\n### prediction ###\n")
        predicted_classes, true_classes = prediction(model, 
                                                     str_time, 
                                                     common.FOLDER_STRUCTURE, 
                                                     common.OUTPUT_PATH)



    #
    # 混同行列
    #
    
    #if common.SW_OPTUNA:
    #    #OPTUNAのとき想定した形でないため
    #    print("predict, confusion_matrix_exe are not done. (by OPTUNA)")
    #else:
    #    # モデルの予測
    #    y_pred = model.predict(X_test)
    #    # 予測確率から最も確率が高いクラスを選択
    #    y_pred_class = np.argmax(y_pred, axis=1)
    #    # Y_testがone-hotエンコーディングの場合、正解ラベルもインデックスに変換
    #    y_true_class = np.argmax(Y_test, axis=1)
    #    
    #    if(common.SW_CONFUSION_MATRIX and common.SW_PREDICTION):
    #        print("\n### confusion_matrix ###\n")
    #        confusion_matrix_exe(y_true_class, # true_classes, #自作のを使うか(対象データがtestでない) or model.predictを使うか
    #                             y_pred_class, # predicted_classes, #自作のを使うか(対象データがtestでない) or model.predictを使うか
    #                             str_time,
    #                             common.OUTPUT_PATH,
    #                             common.SW_PLT_SHOW_confusion_matrix,
    #                             True,
    #                             'd')


    # モデルの予測
    y_pred = model.predict(X_test)

    ## 予測確率から最も確率が高いクラスを選択
    #y_pred_class = np.argmax(y_pred, axis=1)
    ## Y_testがone-hotエンコーディングの場合、正解ラベルもインデックスに変換
    #y_true_class = np.argmax(Y_test, axis=1)
    
    # 予測確率から最も確率が高いクラスを選択（多クラス／二値分類対応）
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_class = np.argmax(y_pred, axis=1)
    else:
        y_pred_class = (y_pred > 0.5).astype(int)

    # Y_testがone-hotエンコーディングの場合、正解ラベルも整数に変換
    if Y_test.ndim > 1 and Y_test.shape[1] > 1:
        y_true_class = np.argmax(Y_test, axis=1)
    else:
        y_true_class = Y_test



    if(common.SW_CONFUSION_MATRIX and common.SW_PREDICTION):
        print("\n### confusion_matrix ###\n")
        confusion_matrix_exe(y_true_class, # true_classes, #自作のを使うか(対象データがtestでない) or model.predictを使うか
                             y_pred_class, # predicted_classes, #自作のを使うか(対象データがtestでない) or model.predictを使うか
                             str_time,
                             common.OUTPUT_PATH,
                             common.SW_PLT_SHOW_confusion_matrix,
                             True,
                             'd')



    #
    # resultsディレクトリ作成
    #
    print("\n### results directory ###\n")
    #"RESULT_PATH": "./results/",
    directory_name = os.path.basename(os.path.normpath(common.RESULT_PATH))
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)


    result_dir = directory_name
    #result_dir = 'results'


    #
    # モデル(重み)を保存
    #
    
    #if common.SW_OPTUNA:
    #    #OPTUNAのとき想定した形でないため
    #    print("model.save is not done. (by OPTUNA)")
    #else:
    #    if(common.SW_LEARNING):#学習実施時のみ
    #        print("\n### save weights ###\n")
    #        model.save(os.path.join(result_dir, 'model.h5'))
    #        print(f"Save weights at {result_dir}/model.h5")
    #        model.save(os.path.join(result_dir, 'model.keras'))
    #        print(f"Save weights at {result_dir}/model.keras")

    if(common.SW_LEARNING):#学習実施時のみ
        print("\n### save weights ###\n")
        model.save(os.path.join(result_dir, 'model.h5'))
        print(f"Save weights at {result_dir}/model.h5")
        model.save(os.path.join(result_dir, 'model.keras'))
        print(f"Save weights at {result_dir}/model.keras")




    # 現在の日時を取得
    end = datetime.now()

    #
    # 時間計測終了
    #
    end_time = time.time()
    elapsed_time = end_time - start_time

    #
    # log write
    #
    print("\n### log write ###\n")
    log_write(str_time,
              now,
              end,
              elapsed_time,
              scores,
              model,
              common.OUTPUT_PATH)



    #
    # Grad-CAM
    #
    
    if common.SW_OPTUNA or common.SW_GRIDSEARCH or common.BASE_MODEL == "ViT":
        #OPTUNAのとき想定した形でないため
        #グリッドサーチのとき想定した形でないため
        print("grad_cam is not done. (by OPTUNA or GRIDSEARCH or common.BASE_MODEL == ViT)")
    else:
        if(common.SW_GRAD_CAM):
            print("\n### Grad-CAM ###\n")
            grad_cam(model, str_time)




    print("\n### end ###\n")
    ######## end ########









# スクリプトが直接実行された場合にmain関数を呼び出す
if __name__ == "__main__":
    main()


