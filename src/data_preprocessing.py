
import os

import cv2
import numpy as np


from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator


#
# common
#
import common


#
# transformers を使用する ViT 専用前処理関数
#
from transformers import ViTImageProcessor

vit_model_name = common.VIT_MODEL_NAME
vit_processor = ViTImageProcessor.from_pretrained(vit_model_name)

def vit_preprocess(img_np):
    """
    ViT用の前処理関数。img_np は NumPy 形式 (H, W, 3)、0〜255 範囲想定。
    戻り値は Tensor (3, 224, 224)
    """
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    processed = vit_processor(images=img_np, return_tensors="tf")
    return processed["pixel_values"][0]  # shape: (3, 224, 224)



#
# 入力画像の処理関数(統合版)
#
def input_img_process(
    img_path,
    img_width,
    img_height,
    sw_cv2,
    sw_np_array,
    sw_dim_add,
    sw_vgg16_preprocess
):

    """
    This function processes an image using OpenCV, including resizing, conversion to RGB, and applying preprocessing steps.
    
    Parameters:
    - img_path: The path to the image file.
    - img_width: The target width of the image.
    - img_height: The target height of the image.
    - sw_cv2: If True, use OpenCV2 func.
    - sw_np_array: If True, convert the image to a numpy array.
    - sw_dim_add: If True, add an extra dimension for batch size.
    - sw_vgg16_preprocess: If True, apply VGG16-specific preprocessing.
    
    Returns:
    - Processed image ready for model input.
    """


    if(sw_cv2):
        img = cv2.imread(img_path)  # image read
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img = cv2.resize(img, (img_width, img_height))  # resize
    else:
        img = load_img(img_path, target_size=(img_width, img_height))  # image read, resize

    if sw_np_array == 1:
        img = np.array(img)  # to NumPy
        #img = img_to_array(img)  # to NumPy
        #np.array までする必要なく img_to_arrayで十分なようだ
        
        #sw_cv2 = falseのとき
        #load_img した時点でnumpy化しているのでここを通す必要ない

    if sw_dim_add == 1:
        # VGG16入力のために次元を1つ追加(バッチ次元の追加) 
        # 複数画像の場合はNumpy化するだけでよい
        img = np.expand_dims(img, axis=0)
        #img = img.reshape(1,img_width,img_height,3)
        
    if sw_vgg16_preprocess == 1:
        img = preprocess_input(img)  # VGG16用の前処理(推奨される処理)

    return img








#
# データ、ラベル準備
#

def data_prepare(
    path, 
    num_of_data,
    class_list,
    img_width,
    img_height,
    folder_structure,
    dataset_mode,
    sw_cv2
):

    data = []
    label = []

    count = 0

    for i in class_list:
        # file_list = os.listdir(root_path + str(i) + '/')
        # file_list = os.listdir(root_path + 'train/' + str(i) + '/')
        # file_list = os.listdir(train_path + str(i) + '/')
        
        
        
        #dir_path = path + str(i) + '/'
        if folder_structure==1 :# medical_image #"FOLDER_STRUCTURE": 1
            dir_path       = path + str(i) + '/'
            #dir_path_mask  = "" #"None is OK. (because of FOLDER_STRUCTURE=1)"
            #dir_path_merge = "" #"None is OK. (because of FOLDER_STRUCTURE=1)"
            #dir_path_tmp   = "" #"None is OK. (because of FOLDER_STRUCTURE=1)"
            
            common.check_dir_exists(dir_path)
            
        else:# medical_image_2 "FOLDER_STRUCTURE": 2
            dir_path       = path + str(i) + '/' + 'images/'
            dir_path_mask  = path + str(i) + '/' + 'masks/'
            dir_path_merge = path + str(i) + '/' + 'merges/'
            dir_path_tmp   = path + str(i) + '/' + 'tmp/'
            
            common.check_dir_exists(dir_path)
            common.check_dir_exists(dir_path_mask)
            common.check_dir_exists(dir_path_merge)
            common.check_dir_exists(dir_path_tmp)
            
        
        
        file_list = os.listdir(dir_path)
        #print("CLASS_LIST : " + i)
        print(f"Class [{i}]: process start.")


        if dataset_mode == 0:
            data_size = len(file_list)
        else:
            data_size = num_of_data



        count = 0
        # for j in range(len(file_list)):
        # for j in range(0, num_of_data):
        for j in range(0, data_size):

            # print(j + '\n')
            # print(root_path + str(i) + '/' + file_list[j] + '\n')
            count += 1

            # img_path = root_path + str(i) + '/' + file_list[j]
            # img_path = root_path + 'train/' + str(i) + '/' + file_list[j]
            # img_path = train_path + str(i) + '/' + file_list[j]
            if folder_structure==1 :# medical_image #"FOLDER_STRUCTURE": 1
                img_path   = dir_path       + file_list[j]
                
                common.check_file_exists(img_path)
            else:# medical_image_2 "FOLDER_STRUCTURE": 2
                img_path   = dir_path       + file_list[j]
                mask_path  = dir_path_mask  + file_list[j] #本画像とマスク画像のファイル名同じ
                merge_path = dir_path_merge + file_list[j]
                tmp_path   = dir_path_tmp   + file_list[j]
                
                common.check_file_exists(img_path)
                common.check_file_exists(mask_path)
                #common.check_file_exists(merge_path)
                #common.check_file_exists(tmp_path)


            #if not os.path.exists(img_path):
            #    print(f"Error: File {img_path} not found!")
            #    continue


            # この関数(data_prepare())をぬけたあとにnumpy化する
            # また複数画像前提なので
            # NUMPY化,次元追加のフラグを0にしている
            
            if folder_structure==1 :# medical_image #"FOLDER_STRUCTURE": 1
                #sw_vgg16_preprocess=0
                x    = input_img_process(img_path,  img_width, img_height, sw_cv2, 0, 0, 0)
                #sw_vgg16_preprocess=1
                #x    = input_img_process(img_path,  img_width, img_height, sw_cv2, 0, 0, 1)
            else:# medical_image_2 "FOLDER_STRUCTURE": 2
                #sw_vgg16_preprocess=0
                x    = input_img_process(img_path,  img_width, img_height, sw_cv2, 0, 0, 0)
                #sw_vgg16_preprocess=1
                #x    = input_img_process(img_path,  img_width, img_height, sw_cv2, 0, 0, 1)
                
                mask = input_img_process(mask_path, img_width, img_height, sw_cv2, 0, 0, 0)# マージするため VGG16用処理はしない
                
                #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) #グレースケールに
                
                if common.ENABLE_MERGE:
                    #マスク画像をもとにマージ画像を作成
                    #x[mask == 0] = [0, 0, 0] #肺以外のピクセルを黒に設定
                    #x[mask != 255] = [0, 0, 0] #肺以外のピクセルを黒に設定
                    x[np.all(mask == [0, 0, 0], axis=-1)] = [0, 0, 0]  #肺以外のピクセルを黒に設定
                    #マージ画像を保存
                    cv2.imwrite(merge_path, x)
                    #maskのグレースケール画像を保存
                    #cv2.imwrite(tmp_path, mask)
            
            
            #
            # ピクセル値を0から1にスケーリング
            #
            #x = x / 255.0
            if common.BASE_MODEL == "ViT":
                x = vit_preprocess(x)
            else:
                x = x / 255.0
            
            data.append(x)
            label.append(class_list.index(i))

        # print(str(i) + "\n")
        # print(str(class_list.index(i)) + "\n")
        # print(str(class_list[class_list.index(i)]) + "\n")
        #print("count: " + str(count) + "\n")
        #print(f"Processed {count} images from class {i}")
        print(f"Class [{i}]: process end. {count} images.")

    return data, label




#
# データ拡張
#

def data_extension(
    X_train,
    Y_train,
    num_of_extension_per_param,#1parameterあたりの拡張画像枚数
    data_extension_path        #拡張画像の保存ディレクトリ
):

    param_list = [
        {'rotation_range': 45},
        {'width_shift_range': 0.2},
        {'height_shift_range': 0.2},
        {'brightness_range': [0.3, 1.5]},
        {'shear_range': 45},
        {'zoom_range': [0.7, 1.7]},
        {'channel_shift_range': 50.0},
        {'horizontal_flip': True},
        {'vertical_flip': True},
        {'vertical_flip': True}  # Dummy #実際は標準化
    ]

    prefix_list = [
        'rotation_',
        'width_shift_',
        'height_shift_',
        'brightness_',
        'shear_',
        'zoom_',
        'channel_shift_',
        'horizontal_flip_',
        'vertical_flip_',
        'samplewise_std_normalization_'
    ]

    for i in range(len(prefix_list)):

        idg_param = param_list[i]
        prefix = prefix_list[i]

        # ディレクトリ準備
        #idg_dir = EXT_PATH + '/' + prefix
        idg_dir = data_extension_path + prefix
        if not os.path.exists(idg_dir):
            os.mkdir(idg_dir)

        if i == len(prefix_list) - 1:
            # 標準化
            datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
        else:
            datagen = ImageDataGenerator(**idg_param)

        g = datagen.flow(X_train,
                         Y_train,
                         batch_size=num_of_extension_per_param,
                         shuffle=False,  # True
                         seed=None,  # None
                         save_to_dir=idg_dir,
                         save_prefix=prefix,
                         save_format='jpeg',
                         subset=None)
        X_idg, Y_idg = g.next()

        if i == 0:  # 最初だけ元の訓練データ2000x0.8=1600に連結
            X_batch = np.concatenate([X_train, X_idg])
            Y_batch = np.concatenate([Y_train, Y_idg])
        else:
            X_batch = np.concatenate([X_batch, X_idg])
            Y_batch = np.concatenate([Y_batch, Y_idg])

        #print(str(i) + "\n")

    return X_batch, Y_batch





def data_set(
    folder_structure,
    dataset_mode,
    train_path,
    val_path,
    test_path,
    num_of_train,
    num_of_val,
    num_of_test,
    sw_data_extension
):


    if folder_structure==1 :# medical_image #"FOLDER_STRUCTURE": 1
        print("FOLDER_STRUCTURE: 1")
    
        print("Train Data:")
        data_train, label_train = data_prepare(train_path,
                                               num_of_train,
                                               common.CLASS_LIST,
                                               common.IMG_WIDTH,
                                               common.IMG_HEIGHT,
                                               folder_structure,
                                               dataset_mode,
                                               common.SW_CV2
                                               )
        print("Val Data:")
        data_val,   label_val   = data_prepare(val_path,
                                               num_of_val,
                                               common.CLASS_LIST,
                                               common.IMG_WIDTH,
                                               common.IMG_HEIGHT,
                                               folder_structure,
                                               dataset_mode,
                                               common.SW_CV2
                                               )
        print("Test Data:")
        data_test,  label_test  = data_prepare(test_path,
                                               num_of_test,
                                               common.CLASS_LIST,
                                               common.IMG_WIDTH,
                                               common.IMG_HEIGHT,
                                               folder_structure,
                                               dataset_mode,
                                               common.SW_CV2
                                               )

    else:# medical_image_2 "FOLDER_STRUCTURE": 2
        print("FOLDER_STRUCTURE: 2")

        X, Y = data_prepare(train_path, 
                            num_of_train,
                            common.CLASS_LIST,
                            common.IMG_WIDTH,
                            common.IMG_HEIGHT,
                            folder_structure,
                            dataset_mode,
                            common.SW_CV2)
        
        #
        #
        #
        print("\n### data split ###\n")
        #data_train, data_test, label_train, label_test = train_test_split(X,
        #                                                                  Y,
        #                                                                  test_size=0.2,
        #                                                                  random_state=1,
        #                                                                  stratify=Y)
        
        
        
        ## 訓練データとテストデータに分ける
        #data_train, data_test, label_train, label_test = train_test_split(X, 
        #                                                                  Y, 
        #                                                                  test_size=common.NUM_OF_TEST, 
        #                                                                  random_state=4017, 
        #                                                                  stratify=Y)
        #
        ## さらに訓練データを検証データとして分割
        #data_train, data_val, label_train, label_val = train_test_split(data_train, 
        #                                                                label_train, 
        #                                                                test_size=common.NUM_OF_VAL, 
        #                                                                random_state=4017, 
        #                                                                stratify=label_train)
        
        
        if dataset_mode == 0: # 比率指定
            train_size_1 = 0.8
            train_size_2 = 0.5
        else: # 各データ数を指定
            train_size_1 = common.NUM_CLASS * num_of_train - (num_of_test + num_of_val)
            train_size_2 = num_of_val
        
        # 訓練データとDUMMYデータに分ける
        data_train, data_dummy, label_train, label_dummy = train_test_split(X,
                                                                            Y,
                                                                            train_size=train_size_1,
                                                                            shuffle=True,
                                                                            random_state=123,
                                                                            stratify=Y)
        
        # さらに検証データとテストデータに分割
        data_val, data_test, label_val, label_test = train_test_split(data_dummy,
                                                                      label_dummy,
                                                                      train_size=train_size_2,
                                                                      shuffle=True,
                                                                      random_state=123,
                                                                      stratify=label_dummy)
        
        
        
        
        print("TRAIN")
        print(f"data_train : {len(data_train)}")
        print(f"label_train: {len(label_train)}")
        print("TEST")
        print(f"data_test  : {len(data_test)}")
        print(f"label_test : {len(label_test)}")
        
        print("VAL")
        print(f"data_val   : {len(data_val)}")
        print(f"label_val  : {len(label_val)}")
        
        #import pdb; pdb.set_trace()



    #
    # X: np.array
    # Y: one-hot
    #
    X_train = np.array(data_train)
    Y_train = to_categorical(label_train)
    X_val   = np.array(data_val)
    Y_val   = to_categorical(label_val)
    X_test  = np.array(data_test)
    Y_test  = to_categorical(label_test)

    #print("X_train.shape:" + str(X_train.shape))
    #print("Y_train.shape:" + str(Y_train.shape))
    #print("X_val.shape:  " + str(X_val.shape))
    #print("Y_val.shape:  " + str(Y_val.shape))
    #print("X_test.shape: " + str(X_test.shape))
    #print("Y_test.shape: " + str(Y_test.shape))




    #データの並び替え
    #np.random.seed(42)
    #rand_index = np.random.permutation(np.arange(len(X)))
    #X = X[rand_index]
    #Y = Y[rand_index]
    #
    #訓練データと検証データに分ける
    #X_train = X[:int(len(X)*0.8)]
    #Y_train = Y[:int(len(Y)*0.8)]
    #X_test = X[int(len(X)*0.8):]
    #Y_test = Y[int(len(Y)*0.8):]


    #
    # データ拡張
    #
    if sw_data_extension:
        print("\n### data_extension ###\n")
        X_batch, Y_batch = data_extension(X_train, 
                                          Y_train,
                                          common.NUM_OF_EXTENSION_PER_PARAM,
                                          common.DATA_EXTENSION_PATH)
    else:
        X_batch = X_train
        Y_batch = Y_train


    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_batch, Y_batch




