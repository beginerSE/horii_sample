
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Dropout,
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    AveragePooling2D, MaxPooling2D, BatchNormalization
)
from tensorflow.keras import initializers, regularizers, optimizers
from transformers import TFAutoModelForImageClassification
from my_callback import MyCallback  # 独自コールバック
import common


def train_single(X_batch, Y_batch, X_val, Y_val):
    """
    モデルを構築・コンパイル・学習して返す関数
    """

    # ===== モデルの種類 =====
    #model_name = "ViT"
    model_name = common.BASE_MODEL
    vit_model_name = common.VIT_MODEL_NAME

    print(f"model : {model_name}")

    if model_name == "ViT":
        print("Using Vision Transformer (ViT) model...")

        input_shape = (3, common.IMG_WIDTH, common.IMG_HEIGHT)
        inputs = Input(shape=input_shape, name="input_image")

        vit_model = TFAutoModelForImageClassification.from_pretrained(
            vit_model_name, from_pt=True
        )

        vit_model.trainable = True
        outputs = vit_model(pixel_values=inputs, training=False)
        logits = outputs.logits
        x = logits

    else:
        base_model = getattr(tf.keras.applications, model_name)(
            weights='imagenet',
            include_top=False,
            input_shape=(common.IMG_WIDTH, common.IMG_HEIGHT, 3),
            pooling='max'
        )

        x = base_model.output

    # ===== Dropout =====
    dropout_rate = common.DROPOUT_RATE # 0.5
    # x = Dropout(dropout_rate)(x)

    # ===== Flatten =====
    if model_name != "ViT":
        x = Flatten()(x)

    # ===== 全結合層 =====
    num_of_unit = common.NUM_OF_UNIT # 128
    activation_h = 'relu'
    initializer = initializers.HeNormal()
    x = Dense(num_of_unit, activation=activation_h, kernel_initializer=initializer)(x)

    # ===== 出力層 =====
    activation_o = 'softmax'
    predictions = Dense(common.NUM_CLASS, activation=activation_o)(x)

    # ===== モデル定義 =====
    if model_name == "ViT":
        model = Model(inputs=inputs, outputs=predictions)
    else:
        model = Model(inputs=base_model.input, outputs=predictions)

    print("model compile")

    # ===== オプティマイザと学習率スケジュール =====
    LEARNING_RATE = common.LEARNING_RATE # 1e-4
    steps_per_epoch = int(np.ceil(len(Y_batch) / common.BATCH_SIZE))
    total_steps = steps_per_epoch * common.EPOCHS
    ALPHA = common.ALPHA # 0.1
    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=total_steps,
        alpha=ALPHA
    )
    optimizer = optimizers.Adamax(learning_rate=lr_schedule)

    # ===== 損失関数とメトリクス =====
    loss = 'categorical_crossentropy'
    metrics_list = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)

    # ===== コールバック =====
    batch_size = common.BATCH_SIZE
    epochs = common.EPOCHS
    patience = 3
    stop_patience = 10
    threshold = 0.9
    factor = 0.5
    ask_epoch = 5
    batches = int(np.ceil(len(Y_batch) / batch_size))

    callbacks = [
        MyCallback(
            model=model,
            patience=patience,
            stop_patience=stop_patience,
            threshold=threshold,
            factor=factor,
            batches=batches,
            epochs=epochs,
            ask_epoch=ask_epoch
        )
    ]

    # ===== モデル訓練 =====
    print("model fit")

    if common.ENABLE_CALLBACKS:
        history = model.fit(
            X_batch,
            Y_batch,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks
        )
    else:
        history = model.fit(
            X_batch,
            Y_batch,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, Y_val),
        )

    return model, history






