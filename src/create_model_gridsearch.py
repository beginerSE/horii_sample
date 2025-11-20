
import tensorflow as tf
from tensorflow.keras import optimizers, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, Xception
from transformers import TFAutoModel
from transformers import TFAutoModelForImageClassification
#from tensorflow.keras.applications import vit

import common

#def create_model_gridsearch(learning_rate=1e-4, dropout_rate=0.5):
#    inputs = Input(shape=(common.FEATURE_SIZE,))

def create_model_gridsearch(learning_rate=1e-4, dropout_rate=0.5, input_shape=(common.IMG_WIDTH, common.IMG_HEIGHT, 3)):
    
    #inputs = Input(shape=input_shape)
    #x = Flatten()(inputs)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(dropout_rate)(x)
    #outputs = Dense(common.NUM_CLASS, activation='softmax')(x)
    #model = Model(inputs, outputs)


    if common.BASE_MODEL == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output

        base_model.trainable = False  # 転移学習で凍結

        x = Flatten()(x)
        x = Dense(common.NUM_OF_UNIT, activation='relu')(x)
        x = Dropout(common.DROPOUT_RATE)(x)
        outputs = Dense(common.NUM_CLASS, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

    elif common.BASE_MODEL == "ViT":

        input_shape=(common.IMG_WIDTH, common.IMG_HEIGHT, 3)
        
        inputs = Input(shape=input_shape)

        # 特徴量のみ使う場合(最後にdense追加する場合)
        base_model = TFAutoModel.from_pretrained(common.VIT_MODEL_NAME, from_pt=True)

        base_model.trainable = False  # 転移学習で凍結

        # pixel_values に入力
        x = base_model(pixel_values=inputs).last_hidden_state  # shape: (batch, 196, hidden_dim)

        # CLS トークンだけ使う場合
        x = x[:, 0, :]  # shape: (batch, hidden_dim)

        # 追加の Dense 層
        x = Dense(common.NUM_OF_UNIT, activation='relu')(x)
        x = Dropout(common.DROPOUT_RATE)(x)
        outputs = Dense(common.NUM_CLASS, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)

    else:
        raise ValueError(f"Unsupported BASE_MODEL: {common.BASE_MODEL}")


    #base_model.trainable = False  # 転移学習で凍結

    #x = Flatten()(x)
    #x = Dense(common.NUM_OF_UNIT, activation='relu')(x)
    #x = Dropout(common.DROPOUT_RATE)(x)
    #outputs = Dense(common.NUM_CLASS, activation='softmax')(x)
    #model = Model(inputs=base_model.input, outputs=outputs)



    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


