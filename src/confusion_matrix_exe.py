
import matplotlib.pyplot as plt

# confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

import common
from typing import List


def calculate_accuracy(cm: List[List[int]]) -> List[float]:
    """
    行ごとの正解率を計算します。
    :param cm: 混同行列
    :return: 各クラスの正解率のリスト
    """
    return cm.diagonal() / cm.sum(axis=1)


#def confusion_matrix_exe(true_classes, predicted_classes, str_time):
def confusion_matrix_exe(
    true_classes: List[int], 
    predicted_classes: List[int],
    str_time: str,
    output_path: str,
    show_plot: bool,
    annot: bool,
    fmt: str
 ) -> None:

    """
    混同行列を計算し、ヒートマップとして表示・保存する関数。
    :param true_classes: 正解ラベル
    :param predicted_classes: 予測ラベル
    :param str_time: 出力ファイル名に追加するタイムスタンプ
    :param output_path: 出力ファイルの保存先パス
    :param show_plot: プロットを表示するかどうか
    :param annot: ヒートマップに注釈を表示するか
    :param fmt: 注釈のフォーマット
    """
    
    # 混同行列を計算
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # 混同行列の正解率を計算
    #accuracy = calculate_accuracy(cm)
    
    # 混同行列を表示
    plt.figure(figsize=(10, 10))
    if cm.size == 0:
        print("混同行列が空です。")
    else:
        sns.heatmap(cm, annot=annot, fmt=fmt, cmap='Blues')
    
    # 正解率を表示
    #accuracy = cm.diagonal() / cm.sum(axis=1)  # 行ごとの正解率
    #sns.heatmap(cm, annot=annot, fmt=fmt, cmap='Blues',cbar=False, xticklabels=accuracy.round(2), yticklabels=accuracy.round(2))

    # ラベルを設定
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 出力パスに末尾スラッシュがない場合は追加
    if not output_path.endswith('/'):
        output_path += '/'

    # 保存先ファイルパスを設定
    output_file = f"{output_path}confusion_matrix_{str_time}.jpg"
    
    # 混同行列画像を保存
    try:
        plt.savefig(output_file)
        print(f"Confusion matrix saved at {output_file}")
    except Exception as e:
        print(f"Error saving confusion matrix to {output_file}: {e}")
        raise
    
    # プロットを表示するかどうかをフラグで制御
    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying confusion matrix plot: {e}")
            raise
    
    plt.close()



