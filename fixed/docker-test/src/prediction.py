
import os
import numpy as np
import matplotlib.pyplot as plt
import random

#
# common
#
import common

#
# data_preprocessing
#
from data_preprocessing import input_img_process



def preprocess_image_for_prediction(img, base_model_type):
    """
    予測用画像の前処理（バッチ化・チャンネル変換）

    Parameters
    ----------
    img : np.ndarray
        入力画像。単枚でもバッチでもOK。
        単枚: (H,W,C) または (C,H,W)
        バッチ: (N,H,W,C) または (N,C,H,W)
    base_model_type : str
        "ViT" なら channels_first (N,3,H,W)
        それ以外は Keras系として channels_last (N,H,W,3)

    Returns
    -------
    np.ndarray
        モデルに渡すための整形済み画像
    """

    # 単枚画像 (C,H,W) -> (H,W,C)
    if img.ndim == 3 and img.shape[0] == 3:
        print("1111")
        img = np.transpose(img, (1, 2, 0))

    # 単枚画像 (H,W,C) -> (1,H,W,C)
    if img.ndim == 3:
        print("2222")
        img = np.expand_dims(img, axis=0)

    # ViTモデルの場合 (channels_first)
    if base_model_type.upper() == "VIT":
        # (N,H,W,C) -> (N,3,H,W)
        if img.shape[-1] == 3:
            print("3333")
            img = np.transpose(img, (0, 3, 1, 2))
        # (N,C,H,W) はそのままでOK

    # Keras系モデルの場合 (channels_last)
    else:
        # (N,C,H,W) -> (N,H,W,C)
        if img.shape[1] == 3:
            print("4444")
            img = np.transpose(img, (0, 2, 3, 1))

    return img




#
# 判定を返す関数
#

def pred_core(img, model):

    """
    #if common.BASE_MODEL == "ViT":
    #    if img.ndim == 3:
    #        img = np.expand_dims(img, axis=0)  # -> (1, 224, 224, 3)
    #    img = np.transpose(img, (0, 3, 1, 2))   # -> (1, 3, 224, 224)

    #画像をモデルに入力して予測を行う。
    #ViTの場合は (1, 3, 224, 224)
    #それ以外(Keras系CNNなど)の場合は (1, 224, 224, 3)

    # 3次元（224,224,3 など）の場合、バッチ次元を追加
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    # ViTモデル（PyTorch形式）向け
    if common.BASE_MODEL == "ViT":
        # ViTの場合、チャンネルが最後にあれば (N,224,224,3) → (N,3,224,224) に変換
        if img.shape[-1] == 3:
            img = np.transpose(img, (0, 3, 1, 2))  # channels_last → channels_first

    # ViT以外（Keras系モデル）は TensorFlow形式に統一
    else:
        # もしチャンネルが最初にある場合 (N,3,224,224) → (N,224,224,3)
        if img.shape[1] == 3:
            img = np.transpose(img, (0, 2, 3, 1))  # channels_first → channels_last
    """



    img = preprocess_image_for_prediction(img, common.BASE_MODEL)

    # モデル予測
    probability = model.predict(img)
    #print('[pred_class] probability: ' + str(probability))

    # 確率の最も高いindex
    pred_index = np.argmax(probability)
    #print('[pred_class] pred_index: ' + str(pred_index))

    # 最も高い確率
    pred_prob = np.max(probability)
    #print('[pred_class] pred_prob: ' + str(pred_prob))

    # 予想クラスを文字列で返す
    pred_class = common.CLASS_LIST[pred_index]
    #print('[pred_class] pred: ' + str(pred_class))

    return pred_index, pred_prob, pred_class





#
#推論・予測
#  不正解の情報を取得
#  各種類からcommon.NUM_OF_TEST_FOR_PREDICTION枚ずつ
#  common.TEST_PATH以下の画像が対象
#

def prediction(
    model,
    str_time,
    folder_structure,
    output_path
):

    ng_file_list = []
    ng_prob_list = []
    ng_index_list = []
    ng_class_list = []

    predicted_classes = []
    true_classes = []

    ng_count = 0


    #print("prediction")
    for i in range(common.NUM_CLASS):
        
        #print("i: " + str(i) + "\n")
        #print(f"CLASS[{common.CLASS_LIST[i]}]\n")

        # 入力画像ファイルリスト
        # file_list = os.listdir(root_path + common.CLASS_LIST[i] + '/' )
        
        #dir_path = common.TEST_PATH + common.CLASS_LIST[i] + '/'
        
        if folder_structure==1 :# medical_image #"FOLDER_STRUCTURE": 1
            dir_path       = common.TEST_PATH + common.CLASS_LIST[i] + '/'
            #dir_path_mask  = "" #"None is OK. (because of FOLDER_STRUCTURE=1)"
            #dir_path_merge = "" #"None is OK. (because of FOLDER_STRUCTURE=1)"
            #dir_path_tmp   = "" #"None is OK. (because of FOLDER_STRUCTURE=1)"
            
            common.check_dir_exists(dir_path)
        else:# medical_image_2 "FOLDER_STRUCTURE": 2
            dir_path       = common.TEST_PATH + common.CLASS_LIST[i] + '/' + 'images/'
            dir_path_mask  = common.TEST_PATH + common.CLASS_LIST[i] + '/' + 'masks/'
            dir_path_merge = common.TEST_PATH + common.CLASS_LIST[i] + '/' + 'merges/'
            dir_path_tmp   = common.TEST_PATH + common.CLASS_LIST[i] + '/' + 'tmp/'
        
            common.check_dir_exists(dir_path)
            common.check_dir_exists(dir_path_mask)
            common.check_dir_exists(dir_path_merge)
            common.check_dir_exists(dir_path_tmp)
        
        
        
        #対象ディレクトリ以下のファイル名のリスト化
        file_list = os.listdir(dir_path)
        
        #
        # シード値を設定（任意の整数を指定）
        #
        np.random.seed(5710)
        #
        # ファイルリストの並び順をランダムにシャッフル
        #
        random.shuffle(file_list)

        
        
        for j in range(common.NUM_OF_TEST_FOR_PREDICTION):
            #print("j: " + str(j) + "\n")

            # 入力画像パス
            # img_path = root_path + common.CLASS_LIST[i] + '/' + file_list[j]
            img_path   = dir_path       + file_list[j]
            mask_path  = dir_path_mask  + file_list[j] #本画像とマスク画像のファイル名同じ
            merge_path = dir_path_merge + file_list[j]
            tmp_path   = dir_path_tmp   + file_list[j]
            
            #print( img_path + '\n')

            #
            # 入力画像読み込み、処理
            #
            # 1画像ずつ,学習済みモデルに入力して,予測している
            # NUMPY化,次元追加のフラグを1にしている
            # 1画像だけなのでNUMPY化にプラスして次元追加が必要
            
            #sw_vgg16_preprocess=0
            #img      = input_img_process(img_path, common.IMG_WIDTH, common.IMG_HEIGHT, common.SW_CV2, 1, 1, 0)
            #sw_vgg16_preprocess=1
            #img      = input_img_process(img_path, common.IMG_WIDTH, common.IMG_HEIGHT, common.SW_CV2, 1, 1, 1)
            
            img      = input_img_process(img_path, common.IMG_WIDTH, common.IMG_HEIGHT, common.SW_CV2, 0, 0, 0)
            
            
            img_disp = input_img_process(img_path, common.IMG_WIDTH, common.IMG_HEIGHT, common.SW_CV2, 0, 0, 0)  # 表示用
            
            
            #sw_vgg16_preprocess=0
            #mask     = input_img_process(mask_path, common.IMG_WIDTH, common.IMG_HEIGHT, common.SW_CV2, 1, 1, 0)
            #sw_vgg16_preprocess=1
            #mask     = input_img_process(mask_path, common.IMG_WIDTH, common.IMG_HEIGHT, common.SW_CV2, 1, 1, 1)
            
            mask     = input_img_process(mask_path, common.IMG_WIDTH, common.IMG_HEIGHT, common.SW_CV2, 0, 0, 0)
            
            
            
            
            
            
            
            if common.ENABLE_MERGE:
                #マスク画像をもとにマージ画像を作成
                #x[mask == 0] = [0, 0, 0] #肺以外のピクセルを黒に設定
                #x[mask != 255] = [0, 0, 0] #肺以外のピクセルを黒に設定
                img[np.all(mask == [0, 0, 0], axis=-1)] = [0, 0, 0]  #肺以外のピクセルを黒に設定

            # 入力画像表示
            #plt.imshow( img_disp )
            #if(common.SW_PLT_SHOW_prediction):
            #    plt.show()
            #plt.close()
            
            
            #
            # ピクセル値を0から1にスケーリング
            #
            img = img / 255.0
            

            # 訓練時の手順
            # data_set
            # dara_prepare
            # でやっていることと合わせる
            # 関数化しないほうがよかったかも
            img = np.array(img)  # to NumPy
            img = np.expand_dims(img, axis=0) # 次元追加



            # 予測
            #if common.SW_GRIDSEARCH:
            #    img = np.transpose(img, (0, 2, 3, 1)) 

            pred_index, pred_prob, pred_class = pred_core(img, model)
            #print('pred: ' + pred_class)

            # for confusion matrix
            predicted_classes.append(pred_index)
            true_classes.append(i)

            # OK, NG表示
            if common.CLASS_LIST[i] == pred_class:
                #print("OK \n")
                pass
            else:
                ng_count = ng_count + 1
                #print("NG count: " + str(ng_count) + "\n")
                #print("NG file name: " + str(file_list[j]) + " \n")
                # 間違った画像の情報取得
                ng_file_list.append(file_list[j])
                ng_prob_list.append(pred_prob)
                ng_index_list.append(pred_index)
                ng_class_list.append(pred_class)
                # 間違った画像の表示
                plt.imshow( img_disp )
                if(common.SW_PLT_SHOW_prediction):
                    plt.show()
                plt.close()


    #print()
    #print("\n### NG info ###\n")
    #print("NG num: ")
    #print(str(len(ng_file_list)))
    #print()


    # プリント
    #for i in range(len(ng_file_list)):
    #    print(str(i))
    #    print(str(ng_file_list[i]))
    #    print(str(ng_prob_list[i]))
    #    print(str(ng_index_list[i]))
    #    print(str(ng_class_list[i]))
    #    print()

    # ファイル出力
    txt_name = f"{output_path}ng_info_test_for_prediction__{str_time}.txt"
    
    #f = open(txt_name, 'w')
    #f = open('ng_info_test_for_prediction__' + str_time + '.txt', 'w')

    with open(txt_name, 'w') as f:
        f.write("common.NUM_CLASS: "                  + str(common.NUM_CLASS) + "\n")
        f.write("common.NUM_OF_TEST_FOR_PREDICTION: " + str(common.NUM_OF_TEST_FOR_PREDICTION) + "\n")
        f.write("NUM_OF_NG:                         " + str(len(ng_file_list)) + "\n\n")
        f.write("1クラスあたり common.NUM_OF_TEST_FOR_PREDICTION \n")
        f.write("試行したデータ数は全部で common.NUM_CLASS * common.NUM_OF_TEST_FOR_PREDICTION \n")
        num_of_all_test_data = common.NUM_CLASS * common.NUM_OF_TEST_FOR_PREDICTION
        f.write(str(num_of_all_test_data) + "\n")
        f.write("NG_RATE:                           " + str(len(ng_file_list) / num_of_all_test_data) + "\n\n")
        f.write("FOLDER_STRUCTURE: 2 の場合はテスト画像の中から選択して実施しているわけではない。全体が対象。\n\n")
        
        for i in range(len(ng_file_list)):
            f.write(str(i) + "\n")
            f.write("画像ファイル名:       " + str(ng_file_list[i]) + "\n")
            f.write("推論した確率:         " + str(ng_prob_list[i]) + "\n")
            f.write("推論したインデックス: " + str(ng_index_list[i]) + "\n")
            f.write("推論したクラス:       " + str(ng_class_list[i]) + "\n")
            f.write("---------- \n")
        
        print(f"ng_info saved at {txt_name}")



    #print("predicted_classes")
    #print(type(predicted_classes), type(predicted_classes))
    #print("true_classes")
    #print(len(true_classes), len(true_classes))


    return predicted_classes, true_classes




