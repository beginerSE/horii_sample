
#import json
import json5
import sys

import common



def load_config(config_file):
    """設定ファイルを読み込む関数（例外処理付き）"""
    try:
        with open(config_file, 'r') as file:
            config = json5.load(file)
        return config
    except FileNotFoundError:
        print(f"Error: 設定ファイル '{config_file}' が見つかりません。")
        return None
    except json5.JSONDecodeError:
        print(f"Error: 設定ファイル '{config_file}' が正しく読み込めません。JSON5フォーマットを確認してください。")
        return None






def initialize_config(config_file, debug=False):


    #config = load_config('./config/config.json')
    #config = load_config('./config/config2.json')

    # 設定ファイルを読み込む
    config = load_config(config_file)


    if config:
        common.CLASS_LIST = config['CLASS_LIST']
        common.NUM_CLASS = config['NUM_CLASS']
        common.IMG_WIDTH = config['IMG_WIDTH']
        common.IMG_HEIGHT = config['IMG_HEIGHT']
        common.FOLDER_STRUCTURE = config['FOLDER_STRUCTURE']
        common.TRAIN_PATH = config['TRAIN_PATH']
        common.VAL_PATH = config['VAL_PATH']
        common.TEST_PATH = config['TEST_PATH']
        common.DATASET_MODE = config['DATASET_MODE']
        common.NUM_OF_TRAIN = config['NUM_OF_TRAIN']
        common.NUM_OF_VAL = config['NUM_OF_VAL']
        common.NUM_OF_TEST = config['NUM_OF_TEST']
        common.NUM_OF_TEST_FOR_PREDICTION = config['NUM_OF_TEST_FOR_PREDICTION']
        common.SW_LEARNING = config['SW_LEARNING']
        common.SW_EVALUATION = config['SW_EVALUATION']
        common.SW_GRAPH_DISP = config['SW_GRAPH_DISP']
        common.SW_PREDICTION = config['SW_PREDICTION']
        common.SW_CONFUSION_MATRIX = config['SW_CONFUSION_MATRIX']
        common.SW_GRAD_CAM = config['SW_GRAD_CAM']
        common.SW_DATA_EXTENSION = config['SW_DATA_EXTENSION']
        common.SW_OPTUNA = config['SW_OPTUNA']
        common.SW_GRIDSEARCH = config['SW_GRIDSEARCH']
        common.BASE_MODEL = config['BASE_MODEL']
        common.VIT_MODEL_NAME = config['VIT_MODEL_NAME']
        common.VIT_MODEL_NAME_TIMM = config['VIT_MODEL_NAME_TIMM']
        common.BATCH_SIZE = config['BATCH_SIZE']
        common.EPOCHS = config['EPOCHS']
        common.ENABLE_FIX_LAYER = config['ENABLE_FIX_LAYER']
        common.FIX_LAYER = config['FIX_LAYER']
        common.ENABLE_CALLBACKS = config['ENABLE_CALLBACKS']




        common.DROPOUT_RATE = config['DROPOUT_RATE']
        common.NUM_OF_UNIT = config['NUM_OF_UNIT']
        common.LEARNING_RATE = config['LEARNING_RATE']
        common.ALPHA = config['ALPHA']
        """
        common. = config['']
        common. = config['']
        common. = config['']
        common. = config['']
        common. = config['']
        common. = config['']
        common. = config['']
        common. = config['']
        common. = config['']
        common. = config['']
        common. = config['']
        common. = config['']
        """



        common.SW_CV2 = config['SW_CV2']
        common.ENABLE_MERGE = config['ENABLE_MERGE']
        common.SW_PLT_SHOW_graph_disp = config['SW_PLT_SHOW_graph_disp']
        common.SW_PLT_SHOW_prediction = config['SW_PLT_SHOW_prediction']
        common.SW_PLT_SHOW_confusion_matrix = config['SW_PLT_SHOW_confusion_matrix']
        common.SW_PLT_SHOW_grad_cam = config['SW_PLT_SHOW_grad_cam']
        common.IMAGE_PATH_INPUT_OF_GRAD_CAM = config['IMAGE_PATH_INPUT_OF_GRAD_CAM']
        common.IMAGE_PATH_INPUT_OF_GRAD_CAM_MASKS = config['IMAGE_PATH_INPUT_OF_GRAD_CAM_MASKS']
        #common.LAST_CONV_LAYER_NAME = config['LAST_CONV_LAYER_NAME']
        common.INTENSITY = config['INTENSITY']
        common.OUTPUT_PATH = config['OUTPUT_PATH']
        common.RESULT_PATH = config['RESULT_PATH']
        common.NUM_OF_EXTENSION_PER_PARAM = config['NUM_OF_EXTENSION_PER_PARAM']
        common.DATA_EXTENSION_PATH = config['DATA_EXTENSION_PATH']
        common.OPTUNA_N_TRIALS = config['OPTUNA_N_TRIALS']
        
        # デバッグモードの場合は詳細なログを出力
        if debug:
            print("デバッグモードが有効です。設定内容を表示します。")
            print(config)
        
    else:
        print("設定の読み込みに失敗しました。プログラムを終了します。")
        sys.exit(1)






