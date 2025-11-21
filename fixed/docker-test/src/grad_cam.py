

#-----------------------------------------------------------------
#必要なライブラリのインポート
#-----------------------------------------------------------------
import math
import re
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K
import cv2
import matplotlib.pyplot as plt



print("\n")
print("Tensorflow version: " + tf.__version__)

# Eager Execution を無効にする
# バージョンが古いため
#tf.compat.v1.disable_eager_execution()


#
# common
#
import common


#
# vit
#
from transformers import ViTFeatureExtractor  # または AutoImageProcessor
vit_processor = ViTFeatureExtractor.from_pretrained(common.VIT_MODEL_NAME)



#-----------------------------------------------------------------
#モデルの準備
#-----------------------------------------------------------------
#####model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)  # VGG16
# model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet", include_top=True)  # MobileNetV2
# model = tf.keras.applications.resnet50.ResNet50(weights="imagenet", include_top=True) #ResNet50
# model = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(weights="imagenet", include_top=True) # efficientnet_v2
#####model.summary()




#-----------------------------------------------------------------
#GPUの読み込み
#-----------------------------------------------------------------
"""
if tf.test.is_gpu_available():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("GPUの読み込みが完了しました")

else:
    print("GPUが存在していません")
    device_lib.list_local_devices()
"""




#-----------------------------------------------------------------
#入力画像の準備
#-----------------------------------------------------------------
#IMAGE_PATH = "./data/cat.jpg"
#IMAGE_PATH = "./cat.jpg"
#IMAGE_PATH = "./medical_image/archive/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
#last_conv_layer_name = "block5_conv3"


def input_image_for_grad_cam():
    """
    指定されたパスから画像を読み込み、Grad-CAM 用に前処理を行う。

    戻り値:
        前処理が施された画像
    """

    #if not os.path.exists(common.IMAGE_PATH_INPUT_OF_GRAD_CAM):
    #    raise FileNotFoundError(f"指定された画像ファイルが見つかりません: {common.IMAGE_PATH_INPUT_OF_GRAD_CAM}")
    common.check_file_exists(common.IMAGE_PATH_INPUT_OF_GRAD_CAM)
    
    resize_image      = tf.keras.preprocessing.image.load_img(common.IMAGE_PATH_INPUT_OF_GRAD_CAM,       target_size=(common.IMG_HEIGHT, common.IMG_WIDTH))
    resize_image_mask = tf.keras.preprocessing.image.load_img(common.IMAGE_PATH_INPUT_OF_GRAD_CAM_MASKS, target_size=(common.IMG_HEIGHT, common.IMG_WIDTH))

    resize_image      = np.array(resize_image)
    resize_image_mask = np.array(resize_image_mask)

    if common.ENABLE_MERGE:
        resize_image[np.all(resize_image_mask == [0, 0, 0], axis=-1)] = [0, 0, 0]  #肺以外のピクセルを黒に設定


    plt.figure(figsize=(4, 4))
    plt.imshow(resize_image)
    if(common.SW_PLT_SHOW_grad_cam):
        plt.show()
    plt.close()


    # 画像をモデルに合わせた形式に変換
    #reshape_image = tf.keras.preprocessing.image.img_to_array(resize_image).reshape(1, common.IMG_HEIGHT, common.IMG_WIDTH, 3)
    #target_image = tf.keras.applications.vgg16.preprocess_input(reshape_image) # vgg16
    ## target_image = tf.keras.applications.mobilenet_v2.preprocess_input(reshape_image)  # mobilenet_v2
    ## target_image = tf.keras.applications.resnet50.preprocess_input(reshape_image)  # resnet50
    ## target_image = tf.keras.applications.efficientnet_v2.preprocess_input(reshape_image) # efficientnet_v2

    reshape_image = tf.keras.preprocessing.image.img_to_array(resize_image).reshape(1, common.IMG_HEIGHT, common.IMG_WIDTH, 3)
    if(common.BASE_MODEL == "VGG16"):
        target_image = tf.keras.applications.vgg16.preprocess_input(reshape_image) # vgg16
    elif(common.BASE_MODEL == "VGG19"):
        target_image = tf.keras.applications.vgg19.preprocess_input(reshape_image)  # vgg19
    elif(common.BASE_MODEL == "ResNet50"):
        target_image = tf.keras.applications.resnet50.preprocess_input(reshape_image)  # resnet50
    elif(common.BASE_MODEL == "Xception"):
        target_image = tf.keras.applications.xception.preprocess_input(reshape_image)  # xception
    elif(common.BASE_MODEL == "ViT"):
        # uint8に変換 (ViTは通常float32ではなくuint8を想定)
        image_uint8 = tf.cast(reshape_image, tf.uint8)
        # numpyに変換（feature_extractorがNumPy想定）
        image_np = image_uint8.numpy()
        # feature_extractorで前処理
        inputs = vit_processor(images=image_np, return_tensors="tf")
        target_image = inputs["pixel_values"]  # shape: (1, 3, 224, 224)
    else:
        raise ValueError(f"未対応の BASE_MODEL が指定されています: {common.BASE_MODEL}")




    return target_image




#-----------------------------------------------------------------
#AIで推論
#-----------------------------------------------------------------
#predict_result = model.predict(target_image)
###############print(tf.keras.applications.vgg16.decode_predictions(predict_result, top=3))




#-----------------------------------------------------------------
#AIが着目している部分を抽出
#-----------------------------------------------------------------
def normalize_and_reshape(
    heatmap,
    target_shape
    ):
    """ヒートマップを正規化し、指定された形状にリサイズ"""
    heatmap = tf.maximum(heatmap, 0)  # 負の値をゼロに
    heatmap /= tf.reduce_max(heatmap)  # 最大値で割って正規化
    return tf.reshape(heatmap, target_shape)  # 形状を変更




def make_heatmap(
    last_conv_layer_name,
    model,
    str_time,
    output_path,
    target_image
    ):

    #with tf.GradientTape() as tape:
    #    last_conv_layer = model.get_layer(last_conv_layer_name)  # 最後の畳込み層を取り出す
    #    #print(f"last_conv_layer: {last_conv_layer}")
    #    
    #    iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    #    model_out, last_conv_layer = iterate(target_image)
    #    #print(f"model_out: {model_out}")
    #    #print(f"last_conv_layer: {last_conv_layer}")
    #
    #    #class_out = model_out[:, np.argmax(model_out[0])] ###########ERROR
    #
    #    # テンソルをNumPy配列に変換
    #    #model_out_value = K.eval(model_out)
    #    #class_out = model_out_value[:, np.argmax(model_out_value[0])]
    #    
    #    # TensorFlowのargmaxを使用
    #    class_out = model_out[:, tf.argmax(model_out[0])] 
    #    #print(f"tf.argmax(model_out[0]): {tf.argmax(model_out[0])}")
    #    #print(f"class_out: {class_out}")
    #
    #    #################差分ここだけ
    #    #grads = tape.gradient(class_out, last_conv_layer) 
    #    #####grads = tf.gradients(class_out, last_conv_layer)
    #    grads = tf.gradients(class_out, last_conv_layer)[0]
    #    
    #    #print(f"grads: {grads}")
    #    
    #    #
    #    #
    #    # 下記の加算、クリッピングは要検討
    #    # なしでうまくいくなら、コメントアウト
    #    #
    #    #
    #    # 勾配計算中にNaNを防ぐために、少しの小さな定数を加えてみてください
    #    grads = tf.maximum(grads, 1e-8)  # 勾配に小さい値を加える
    #    # 勾配をクリッピングする方法
    #    grads = tf.clip_by_value(grads, -1.0, 1.0)  # 勾配を-1から1の範囲に制限
    #    
    #    
    #    ## grads がリストであれば、それをテンソルに変換
    #    #grads = tf.convert_to_tensor(grads)
    #    #print(f"grads: {grads}")
    #    
    #    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    #    #print(f"pooled_grads: {pooled_grads}")
    #    
    #    
    #    
    
    with tf.GradientTape() as tape:
        # モデルにおける最後の畳み込み層を取得
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # 新しいモデルを定義（入力から最後の畳み込み層の出力までを取り出す）
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        
        # モデルの出力と最後の畳み込み層の出力を取得
        model_out, last_conv_layer_out = iterate(target_image)
        
        # 最も確信度が高いクラスのインデックスを取得
        class_idx = tf.argmax(model_out[0])  # 出力の最も確信度が高いクラスのインデックスを取得
        class_out = model_out[:, class_idx]  # そのクラスの出力を取得
        
    # 勾配を計算
    grads = tape.gradient(class_out, last_conv_layer_out)
    
    # 勾配に小さい値を加えてゼロ除算を防止
    grads = tf.maximum(grads, 1e-8)
    
    # 勾配を-1から1の範囲に制限
    grads = tf.clip_by_value(grads, -1.0, 1.0)
    
    # 勾配の平均値を計算（平均プール）
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # チャンネル次元に沿って平均
    
    
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_out ), axis=-1)
    #print(f"heatmap: {heatmap}")

    heatmap_shape = (grads.shape[1], grads.shape[2])
    #print(f"heatmap_shape: {heatmap_shape}")
    #heatmap_shape: (Dimension(1), Dimension(14))





    ## ヒートマップの値を正規化
    ##heatmap_Emphasis = np.maximum(heatmap, 0)###########ERROR
    #
    ## テンソルをNumPy配列に変換
    ##heatmap_value = K.eval(heatmap)
    ##heatmap_Emphasis = np.maximum(heatmap_value, 0)
    #
    ## TensorFlowのtf.maximumを使用
    #heatmap_Emphasis = tf.maximum(heatmap, 0)
    ##print(f"heatmap_Emphasis: {heatmap_Emphasis}")
    #
    ##heatmap_Emphasis /= np.max(heatmap_Emphasis)###########ERROR
    #
    ## テンソルをNumPy配列に変換
    ##heatmap_Emphasis_value = K.eval(heatmap_Emphasis)
    ##heatmap_Emphasis_value /= np.max(heatmap_Emphasis_value)
    #
    ## TensorFlowのtf.reduce_maxを使用して最大値を取得
    #heatmap_Emphasis /= tf.reduce_max(heatmap_Emphasis)
    ##print(f"heatmap_Emphasis: {heatmap_Emphasis}")
    #
    ##heatmap_Emphasis = heatmap_Emphasis.reshape(heatmap_shape)###########ERROR
    ## reshapeの方法を変更
    #heatmap_Emphasis = tf.reshape(heatmap_Emphasis, heatmap_shape)
    ##print(f"heatmap_Emphasis: {heatmap_Emphasis}")

    heatmap_Emphasis = normalize_and_reshape(heatmap, heatmap_shape)





    # テンソルをNumPy配列に変換
    #heatmap_Emphasis_value = K.eval(heatmap_Emphasis)
    heatmap_Emphasis_value = heatmap_Emphasis.numpy()  # tf.Tensor を numpy 配列に変換
    #print(f"heatmap_Emphasis_value: {heatmap_Emphasis_value}")

    # 1次元を2次元行列に変換する
    #heatmap_Emphasis_value = heatmap_Emphasis_value.reshape(1, -1)
    
    
    # NumPy配列としてmatshowでプロット
    #plt.matshow(heatmap_Emphasis)###########ERROR
    #plt.matshow(heatmap_Emphasis_value)
    plt.imshow(heatmap_Emphasis_value)
    
    
    #img_name = 'heatmap_Emphasis_value__' + str_time + '.jpg'
    img_name = f"{output_path}heatmap_Emphasis_value__{str_time}.jpg"
    
    #plt.savefig(img_name)
    
    # heatmap_Emphasis_value画像を保存
    try:
        plt.savefig(img_name)
        print(f"heatmap_Emphasis_value saved at {img_name}")
    except Exception as e:
        print(f"Error saving heatmap_Emphasis_value to {img_name}: {e}")




    if(common.SW_PLT_SHOW_grad_cam):
        plt.show()
    
    plt.close()

    return heatmap_Emphasis




#-----------------------------------------------------------------
#元画像にヒートマップを合成
#-----------------------------------------------------------------
def compose_heatmap(
    image_path,
    heatmap,
    str_time,
    output_path
):

    img = cv2.imread(image_path)

    # テンソルをNumPy配列に変換
    #heatmap = K.eval(heatmap)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    img = heatmap * common.INTENSITY + img

    #img_name = 'heatmap__' + str_time + '.jpg'
    img_name = f"{output_path}heatmap__{str_time}.jpg"

    #cv2.imwrite(img_name, img)

    # Grad-cam画像を保存
    try:
        cv2.imwrite(img_name, img)
        print(f"Grad-cam saved at {img_name}")
    except Exception as e:
        print(f"Error saving grad-cam to {img_name}: {e}")

    
    img_imshow = plt.imread(img_name)
    plt.imshow(img_imshow)
    
    if(common.SW_PLT_SHOW_grad_cam):
        plt.show()
    
    plt.close()









def grad_cam(
    model, 
    str_time
):

    target_image = input_image_for_grad_cam()
    
    last_conv_layer_name_ = common.last_conv_layer_name[common.BASE_MODEL]
    
    #print("TEST")
    #print("TEST")
    #print("TEST")
    #
    #print(f"last_conv_layer_name_ {last_conv_layer_name_}")
    #print("TEST")
    #print(f"common.BASE_MODEL {common.BASE_MODEL}")
    #print("TEST")
    #print(f"common.last_conv_layer_name {common.last_conv_layer_name}")
    #print("TEST")
    
    heatmap = make_heatmap(last_conv_layer_name_,
                           model, 
                           str_time,
                           common.OUTPUT_PATH,
                           target_image)

    compose_heatmap(common.IMAGE_PATH_INPUT_OF_GRAD_CAM,
                    heatmap,
                    str_time,
                    common.OUTPUT_PATH)





