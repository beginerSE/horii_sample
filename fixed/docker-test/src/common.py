

import os

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import Xception

#
#
# 初回、重みDLするため
# SSD容量確認
#
#
# from tensorflow.keras.applications import \
#    VGG16, \
#    VGG19, \
#    ResNet50, \
#    ResNet50V2, \
#    ResNet101, \
#    ResNet152, \
#    DenseNet121, \
#    DenseNet201, \
#    MobileNet, \
#    MobileNetV2, \
#    EfficientNetB0, \
#    EfficientNetB7, \
#    NASNetMobile, \
#    NASNetLarge, \
#    InceptionResNetV2, \
#    InceptionV3, \
#    Xception




#
# configx.jsonから値を読み込んで設定している。
# ここの値は初期値。基本configx.jsonの値で上書きされている。
#




# ----------------------------------------------------------------------------------
# クラスのリスト
# ----------------------------------------------------------------------------------
#CLASS_LIST = ['NORMAL',
#              'PNEUMONIA']
#CLASS_LIST = ['NORMAL',
#              'PNEUMONIA',
#              'Lung_Opacity',
#              'Viral Pneumonia']
CLASS_LIST = []


# ----------------------------------------------------------------------------------
# クラス数
# ----------------------------------------------------------------------------------
#NUM_CLASS = len(CLASS_LIST)
NUM_CLASS = 0


# ----------------------------------------------------------------------------------
# 画像サイズ
# ----------------------------------------------------------------------------------
#IMG_WIDTH = 224
#IMG_HEIGHT = 224
IMG_WIDTH = 0
IMG_HEIGHT = 0


# ----------------------------------------------------------------------------------
# 入力画像のフォルダ構成
# ----------------------------------------------------------------------------------
# medical_image  （common.FOLDER_STRUCTURE = 1）
#  train,val,testフォルダに最初から分かれている
# medical_image_2（common.FOLDER_STRUCTURE = 2）
#  train,val,testフォルダに分かれていない
#FOLDER_STRUCTURE = 1
FOLDER_STRUCTURE = 2


# ----------------------------------------------------------------------------------
# 入力画像のあるフォルダパス
# ----------------------------------------------------------------------------------
# medical_image_2（common.FOLDER_STRUCTURE = 2）
# のときは
# jsonで下記に設定
# TRAIN_PATH = VAL_PATH = TEST_PATH
#
#TRAIN_PATH = './medical_image/archive/chest_xray/train/'
#VAL_PATH = './medical_image/archive/chest_xray/val/'
#TEST_PATH = './medical_image/archive/chest_xray/test/'

TRAIN_PATH = '../medical_2/medical_image/archive/chest_xray/train/'
VAL_PATH = '../medical_2/medical_image/archive/chest_xray/val/'
TEST_PATH = '../medical_2/medical_image/archive/chest_xray/test/'



# ----------------------------------------------------------------------------------
# データ数の指定かデータ比率の指定かを切り替えるスイッチ
# ----------------------------------------------------------------------------------
# 0:比率指定
# 1:データ数指定(NUM_OF_TRAIN等での指定)
DATASET_MODE = 0


# ----------------------------------------------------------------------------------
# 使用するデータ数(1種類あたり)
# ----------------------------------------------------------------------------------
#
NUM_OF_TRAIN = 10  # FOLDER_STRUCTURE = 2 のときは左記の値にならない (NUM_OF_TRAIN * (1 - 0.2)) * NUM_CLASS -> 修正した
NUM_OF_VAL = 2
NUM_OF_TEST = 2  # FOLDER_STRUCTURE = 2 のときは左記の値にならない (NUM_OF_TRAIN *      0.2 ) * NUM_CLASS -> 修正した

# prediction()で使用するデータ数(NUM_OF_TEST 以下に設定)
# FOLDER_STRUCTURE = 2 のときは
# train,val,testフォルダに分かれていないので
# 改めてprediction()向けに入力データを設定していることになる
NUM_OF_TEST_FOR_PREDICTION = 1


# ----------------------------------------------------------------------------------
# 各ブロックのON/OFF
# ----------------------------------------------------------------------------------

#
# 学習するか、学習済みの重みを使うか
# True : 学習の実行
# False: 学習しない（学習済みの重みを使用）
#
SW_LEARNING = True
# SW_LEARNING = False

#
# EVALUATION実行するかどうか（TEST画像全てに対して）
#
SW_EVALUATION = True
# SW_EVALUATION = False

#
# GRAPH_DISP実行するかどうか
#
SW_GRAPH_DISP = True
# SW_GRAPH_DISP = False

#
# PREDICTION実行するかどうか（NG情報の取得）
#
SW_PREDICTION = True
# SW_PREDICTION = False

#
# CONFUSION_MATRIX実行するかどうか（混同行列）
#
SW_CONFUSION_MATRIX = True
# SW_CONFUSION_MATRIX = False

#
# GRAD_CAM実行するかどうか
#
SW_GRAD_CAM = True
# SW_GRAD_CAM = False


#
# データ拡張するかどうか
#
# SW_DATA_EXTENSION = True
SW_DATA_EXTENSION = False


#
# OPTUNA
#
# SW_OPTUNA = True
SW_OPTUNA = False


# ----------------------------------------------------------------------------------
# 学習関連パラメータ
# ----------------------------------------------------------------------------------
BASE_MODEL = "VGG16"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
VIT_MODEL_NAME_TIMM = "vit_base_patch16_224_in21k"
BATCH_SIZE = 100
EPOCHS = 1
ENABLE_FIX_LAYER = False
FIX_LAYER = 19  # ex.VGG16  19:全体モデルの最終段以外学習しない
ENABLE_CALLBACKS = False

DROPOUT_RATE = 0.5 # 最終段
NUM_OF_UNIT = 128 # 最終段
LEARNING_RATE = 1e-4
ALPHA = 0.1 # CosineDecay


#
# モデル
#
available_models = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'Xception': Xception
}
# available_models = {
#    'VGG16': VGG16,
#    'VGG19': VGG19,
#    'ResNet50': ResNet50,
#    'ResNet50V2': ResNet50V2,
#    'ResNet101': ResNet101,
#    'ResNet152': ResNet152,
#    'DenseNet121': DenseNet121,
#    'DenseNet201': DenseNet201,
#    'MobileNet': MobileNet,
#    'MobileNetV2': MobileNetV2,
#    'EfficientNetB0': EfficientNetB0,
#    'EfficientNetB7': EfficientNetB7,
#    'NASNetMobile': NASNetMobile,
#    'NASNetLarge': NASNetLarge,
#    'InceptionResNetV2': InceptionResNetV2,
#    'InceptionV3': InceptionV3,
#    'Xception': Xception
# }




#
# 対象とする最終層
#

last_conv_layer_name = {
    'VGG16': "block5_conv3",
    'VGG19': "block5_conv4",
    'ResNet50': "conv5_block3_3_conv",
    'Xception': "block14_sepconv2"
}

#last_conv_layer_name = {
#    'VGG16': "block5_conv3",          # VGG16の最終畳み込み層
#    'VGG19': "block5_conv4",          # VGG19の最終畳み込み層
#    'ResNet50': "conv5_block3_3_conv", # ResNet50の最終畳み込み層
#    'InceptionV3': "mixed10",         # InceptionV3の最終層
#    'Xception': "block14_sepconv2",   # Xceptionの最終畳み込み層
#    'DenseNet121': "conv5_block16_2_relu", # DenseNet121の最終畳み込み層
#    'MobileNet': "conv_dw_13",        # MobileNetの最終畳み込み層
#    'EfficientNetB0': "top_activation", # EfficientNetB0の最終層
#    'NASNetLarge': "normal_concat_8", # NASNetLargeの最終層
#    'ResNet101': "conv5_block3_3_conv", # ResNet101の最終畳み込み層
#    'ResNet152': "conv5_block3_3_conv", # ResNet152の最終畳み込み層
#}




# ----------------------------------------------------------------------------------
# 入力画像の処理で
# cv2使うかkeras.preprocessing.image使うか
# True : CV2
# False: keras.preprocessing.image
# ----------------------------------------------------------------------------------
SW_CV2 = True
# SW_CV2 = False


# ----------------------------------------------------------------------------------
# マスク画像をもとにマージするかどうか
# ----------------------------------------------------------------------------------
ENABLE_MERGE = True
# ENABLE_MERGE = False


# ----------------------------------------------------------------------------------
# plt.show()関連を有効にするかどうか
# True : 有効
# False: 無効
# ----------------------------------------------------------------------------------

# SW_PLT_SHOW_graph_disp = True
SW_PLT_SHOW_graph_disp = False

# SW_PLT_SHOW_prediction = True
SW_PLT_SHOW_prediction = False

# SW_PLT_SHOW_confusion_matrix = True
SW_PLT_SHOW_confusion_matrix = False

# SW_PLT_SHOW_grad_cam = True
SW_PLT_SHOW_grad_cam = False



# ----------------------------------------------------------------------------------
# GRAD_CAM
# ----------------------------------------------------------------------------------

# 対象とする入力画像
#IMAGE_PATH_INPUT_OF_GRAD_CAM = "./medical_image_2/archive/COVID-19_Radiography_Dataset/COVID/images/COVID-5.png"
IMAGE_PATH_INPUT_OF_GRAD_CAM = "../medical_2/medical_image_2/archive/COVID-19_Radiography_Dataset/COVID/images/COVID-5.png"

# 対象とするマスク画像
# = "../medical_2/medical_image_2/archive/COVID-19_Radiography_Dataset/COVID/masks/COVID-5.png"
IMAGE_PATH_INPUT_OF_GRAD_CAM_MASKS = "../medical_2/medical_image_2/archive/COVID-19_Radiography_Dataset/COVID/masks/COVID-5.png"

# 対象とする最終層
# LAST_CONV_LAYER_NAME = "block5_conv3" #VGG16
# LAST_CONV_LAYER_NAME = "conv5_block3_3_conv" #ResNet50
# LAST_CONV_LAYER_NAME = "block14_sepconv2" #Xception
# ヒートマップ合成時パラメータ
INTENSITY = 0.6


# ----------------------------------------------------------------------------------
# OUTPUT_PATH
# ----------------------------------------------------------------------------------
# 出力ファイル先のフォルダ
OUTPUT_PATH = "./output/"


# ----------------------------------------------------------------------------------
# モデルの保存先
# ----------------------------------------------------------------------------------
# 出力ファイル先のフォルダ
RESULT_PATH = "./results/"


# ----------------------------------------------------------------------------------
# データ拡張
# ----------------------------------------------------------------------------------
# 1parameterあたりの拡張画像枚数
NUM_OF_EXTENSION_PER_PARAM = 10

# データ拡張後の保存先フォルダ
DATA_EXTENSION_PATH = "./ImageDataGenerator_dataset/"


# ----------------------------------------------------------------------------------
# OPTUNAの試行回数
# ----------------------------------------------------------------------------------
OPTUNA_N_TRIALS  = 10


# ----------------------------------------------------------------------------------
# parser, debug print
# ----------------------------------------------------------------------------------

def debug_print(message, debug):
    if debug:
        print(f"[DEBUG] {message}")


# ----------------------------------------------------------------------------------
# check_file_exists
# ----------------------------------------------------------------------------------
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")
    return True


def check_dir_exists(dir_path):
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"指定されたディレクトリが見つかりません: {dir_path}")
    return True



