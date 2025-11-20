

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import common


class KerasClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, epochs=10, batch_size=32, learning_rate=0.0001, dropout_rate=0.5, **kwargs):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs  # その他のパラメータ
        self.history = None  # historyの初期化

    #def fit(self, X, y, epochs=10, batch_size=32, learning_rate=0.0001, dropout_rate=0.5, **kwargs):
    #    # build_fn に渡すパラメータを修正
    #    #self.model = self.build_fn(**self.kwargs)
    #    
    #    # kwargsから必要なパラメータを取り出してモデルを作成
    #    #dropout_rate = self.kwargs.get('dropout_rate', 0.5)  # kwargsからdropout_rateを取得、デフォルトは0.5
    #    #learning_rate = self.kwargs.get('learning_rate', 0.0001)  # kwargsからlearning_rateを取得、デフォルトは0.0001
    #    # その他のパラメータも同様に取得
    #
    #    # モデルを作成
    #    self.model = self.build_fn(learning_rate=self.learning_rate, dropout_rate=self.dropout_rate, **self.kwargs)
    #    
    #    # モデル訓練
    #    history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
    #    # verbose=0：表示なし（デフォルト）
    #    # verbose=1：簡単な進行状況を表示
    #    # verbose=2：詳細な進行状況を表示
    #    
    #    self.history = history  # historyオブジェクトを保存
    #    
    #    return self

    def fit(self, X, y, epochs=10, batch_size=32, learning_rate=0.0001, dropout_rate=0.5, **kwargs):
        # Y が整数の場合は one-hot に変換
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y = to_categorical(y, num_classes=common.NUM_CLASS)

        # 入力データのチャンネル順を確認
        if X.shape[1] == 3:  # (N, C, H, W)
            X = np.transpose(X, (0, 2, 3, 1))  # -> (N, H, W, C)

        # build_fn に input_shape を渡す
        input_shape = self.kwargs.get('input_shape', X.shape[1:])
        self.model = self.build_fn(
            learning_rate=self.learning_rate,
            dropout_rate=self.dropout_rate,
            input_shape=input_shape,
            **self.kwargs
        )

        # モデル作成
        #self.model = self.build_fn(learning_rate=self.learning_rate, dropout_rate=self.dropout_rate, **self.kwargs)

        # 学習
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        self.history = history
        return self


    #def predict(self, X):
    #    # 予測（確率をクラスラベルに変換）
    #    predictions = self.model.predict(X)
    #    return predictions
    #    
    #    #if predictions.shape[1] > 1:  # 複数クラス分類の場合
    #    #    return predictions.argmax(axis=1)  # 最も確率が高いクラスのインデックスを返す
    #    #else:  # 2クラス分類の場合
    #    #    return (predictions > 0.5).astype("int32")  # 0.5を閾値としてクラス0か1を予測
        

    def predict(self, X):
        predictions = self.model.predict(X)
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)
        return predictions


    #def score(self, X, y):
    #    # 精度を評価
    #    return self.model.evaluate(X, y, verbose=0)[1]
    #    
    #    # モデルが出力する予測ラベルを整数に変換
    #    #y_pred = self.predict(X)
    #    #return accuracy_score(y, y_pred)  # sklearnのaccuracy_scoreを使用して精度を計算

    def score(self, X, y):
        # y が one-hot の場合は整数ラベルに変換
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_int = np.argmax(y, axis=1)
        else:
            y_int = y

        # 予測も整数ラベルに変換
        y_pred = self.predict(X)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_int = np.argmax(y_pred, axis=1)
        else:
            y_pred_int = y_pred

        return accuracy_score(y_int, y_pred_int)



    def evaluate(self, X, y, verbose):
        """
        モデルの評価を行うメソッド。損失とメトリックを返す。
        引数:
            X: 入力データ
            y: 正解データ（ターゲット）
        戻り値:
            損失とメトリックのタプル (loss, metrics)
        """
        # モデルの評価
        loss, accuracy = self.model.evaluate(X, y, verbose=verbose)
        
        # 損失と精度を返す
        return loss, accuracy

    def save(self, filepath):
        """
        モデルを指定されたファイルパスに保存するメソッド。
        
        引数:
            filepath: 保存先のファイルパス（例: "model.h5"）
        """
        self.model.save(filepath)

    def get_layer(self, layer_name_or_index):
        """
        指定されたレイヤー名またはインデックスでモデルのレイヤーを取得するメソッド。
        
        Parameters:
        - layer_name_or_index: レイヤーの名前またはインデックス
        
        Returns:
        - 指定されたレイヤー
        """
        if isinstance(layer_name_or_index, str):
            # レイヤー名で指定された場合
            return self.model.get_layer(layer_name_or_index)
        elif isinstance(layer_name_or_index, int):
            # インデックスで指定された場合
            return self.model.layers[layer_name_or_index]
        else:
            raise ValueError("layer_name_or_index should be either a string (layer name) or an integer (layer index).")

    def inputs(self):
        """
        モデルの入力層を取得するメソッド。
        
        Returns:
        - モデルの入力層（KerasのInput層）
        """
        return self.model.input

    def output(self):
        """
        モデルの出力層を取得するメソッド。
        
        Returns:
        - モデルの出力層（Kerasの最終層）
        """
        return self.model.output




def train_gridsearch(X_batch, Y_batch, create_model_gridsearch):
    """
    グリッドサーチを実行する関数
    """

    print("Performing Grid Search")

    # --------------------------------------------------------------------------------------------
    # ハイパーパラメータのグリッド定義
    # --------------------------------------------------------------------------------------------
    param_grid = {
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'dropout_rate': [0.3, 0.4, 0.5],
        'batch_size': [8],
        'epochs': [3]
    }


    # --------------------------------------------------------------------------------------------
    # モデルのラッピング
    # --------------------------------------------------------------------------------------------
    print("Grid Search: KerasClassifierCustom")
    #model = KerasClassifierCustom(build_fn=create_model_gridsearch)
    model = KerasClassifierCustom(build_fn=create_model_gridsearch, input_shape=(common.IMG_WIDTH, common.IMG_HEIGHT, 3))


    # --------------------------------------------------------------------------------------------
    # グリッドサーチの設定
    # --------------------------------------------------------------------------------------------
    print("Grid Search: GridSearchCV")

    # 単純に訓練データ全体を使用するCVスプリット
    cv_full = [(np.arange(len(X_batch)), np.arange(len(X_batch)))]

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=1,
        #cv=3,  # クロスバリデーション分割数
        cv=cv_full,  # CVなしと同等
        #scoring='accuracy',
        scoring=None,  # <- estimator.score() を使う
    )

    print("Grid Search: grid_search.fit")

    grid_search.fit(X_batch, Y_batch)

    # --------------------------------------------------------------------------------------------
    # 結果を表示
    # --------------------------------------------------------------------------------------------
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    # 結果を pandas DataFrame に変換
    results = grid_search.cv_results_
    df = pd.DataFrame(results)
    df_select = df[[
        'param_learning_rate', 'param_dropout_rate', 'param_batch_size', 'param_epochs',
        'mean_test_score', 'std_test_score', 'rank_test_score'
    ]]

    # CSV保存
    df.to_csv(f"{common.OUTPUT_PATH}grid_search_results.csv", index=False)
    df_select.to_csv(f"{common.OUTPUT_PATH}grid_search_results_select.csv", index=False)

    print("Grid search results saved to grid_search_results.csv")
    print("Grid search results saved to grid_search_results_select.csv")

    # --------------------------------------------------------------------------------------------
    # 最適なモデルで再学習
    # --------------------------------------------------------------------------------------------
    print("Grid Search: best_model")
    best_model = grid_search.best_estimator_

    print("Grid Search: best_model.fit")
    history = best_model.fit(
        X_batch,
        Y_batch,
        epochs=grid_search.best_params_['epochs'],
        batch_size=grid_search.best_params_['batch_size'],
        dropout_rate=grid_search.best_params_['dropout_rate'],
        learning_rate=grid_search.best_params_['learning_rate']
    )

    print("Grid Search: Done")

    return best_model, history, grid_search




