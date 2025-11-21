


            # モデル名を変数として定義
            #model_name = "ResNet50"
            #model_name = "ResNet101"
            #model_name = "ResNet152"
            #model_name = "InceptionV3"
            #model_name = "Xception"
            #model_name = "VGG16"
            #model_name = "VGG19"
            #model_name = "MobileNet"
            #model_name = "MobileNetV2"
            #model_name = "DenseNet121"
            #model_name = "DenseNet169"
            #model_name = "DenseNet201"
            #model_name = "NASNetLarge"
            #model_name = "NASNetMobile"
            #model_name = "EfficientNetB0"
            #model_name = "EfficientNetB1"
            #model_name = "EfficientNetB2"
            #model_name = "EfficientNetB3"
            #model_name = "EfficientNetB4"
            #model_name = "EfficientNetB5"
            #model_name = "EfficientNetB6"
            model_name = "ViT"
            # ViT モデル名（Hugging Faceの事前学習済モデル）
            vit_model_name = common.VIT_MODEL_NAME


            print(f"model : {model_name}")


            if model_name == "ViT":
                print("Using Vision Transformer (ViT) model...")
             
                input_shape = (3, common.IMG_WIDTH, common.IMG_HEIGHT)
                inputs = Input(shape=input_shape, name="input_image")
                
                #vit_model = TFAutoModel.from_pretrained(vit_model_name)

                #
                # TensorFlow形式への変換
                #
                vit_model = TFAutoModelForImageClassification.from_pretrained(vit_model_name, from_pt=True)




                vit_model.trainable = True  # 学習させたい場合



                #outputs = vit_model(pixel_values=inputs, training=False).last_hidden_state
                #x = outputs[:, 0, :]  # CLSトークン



                # 出力を取得（logitsを使う）
                outputs = vit_model(pixel_values=inputs, training=False)
                # logitsを取り出す
                logits = outputs.logits  # 予測スコア（logits）
                # CLSトークン（logitsの最初の部分を使用する場合）
                #x = logits[:, 0, :]  # 必要に応じてこの部分を変更
                x = logits





            else:
                # 事前学習済みモデルのロード
                base_model = getattr(tf.keras.applications, model_name)(
                    weights='imagenet', include_top=False, input_shape=(common.IMG_WIDTH, common.IMG_HEIGHT, 3), pooling='max'
                )
                
                # 層を凍結（事前学習済みの重みを更新しないようにする）
                #for layer in base_model.layers:
                #    layer.trainable = False
                
                # 層の凍結のパラメータはconfig2.jsonから指定できるようにはしているが...
                # 層を凍結せず、上位の層だけを解凍
                #for layer in base_model.layers[:-10]:  # 最後のx層を学習させる
                #    layer.trainable = False
                
                # 全層学習
                
                # 新しい分類器を追加
                x = base_model.output
            
            
            



            # プーリング
            #x = GlobalAveragePooling2D()(x)
            #x = GlobalMaxPooling2D()(x)
            #x = AveragePooling2D()(x)
            #x = MaxPooling2D()(x)
            
            
            
            # Dropout
            #dropout_rate = 0.1
            #dropout_rate = 0.2
            #dropout_rate = 0.3
            #dropout_rate = 0.4
            dropout_rate = 0.5
            #dropout_rate = 0.6
            #x = Dropout(dropout_rate)(x)
            
            
            
            # 1次元にフラット化
            if model_name == "ViT":
                pass
            else:
                x = Flatten()(x)
            
            
            
            # 新しい全結合層を追加
            #num_of_unit = 4
            #num_of_unit = 8
            #num_of_unit = 16
            #num_of_unit = 32
            #num_of_unit = 64
            num_of_unit = 128
            #num_of_unit = 256
            #num_of_unit = 512
            #num_of_unit = 1024
            
            activation_h = 'relu'
            #activation_h = 'LeakyReLU'
            #activation_h = 'elu'
            #activation_h = 'selu'
            #activation_h = 'ELU'
            #activation_h = 'SELU'
            
            # 初期化戦略
            #initializer = initializers.GlorotNormal()
            #initializer = initializers.RandomNormal(mean=0.0, stddev=0.05)
            initializer = initializers.HeNormal()
            #initializer = initializers.HeUniform()
            #initializer = initializers.GlorotUniform()
            
            #x = Dense(num_of_unit, activation=activation_h)(x)
            #x = Dense(num_of_unit, activation=activation_h, kernel_initializer=initializer)(x)
            # L1正則化あり
            #x = Dense(num_of_unit, activation=activation_h, kernel_initializer=initializer, kernel_regularizer=regularizers.l1(0.01))(x)
            # L2正則化あり
            #x = Dense(num_of_unit, activation=activation_h, kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.01))(x)
            # L1,L2正則化あり
            #x = Dense(num_of_unit, activation=activation_h, kernel_initializer=initializer, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
            
            
            
            # バッチ正規化を追加
            #x = BatchNormalization()(x)
            
            
            # 出力層
            activation_o = 'softmax'# 多クラス分類
            #activation_o = 'sigmoid'# 二値分類
            predictions = Dense(common.NUM_CLASS, activation=activation_o)(x)
            
            
            
            # 完全なモデル
            if model_name == "ViT":
                model = Model(inputs=inputs, outputs=predictions)
            else:
                model = Model(inputs=base_model.input, outputs=predictions)
            
            # モデルのコンパイル
            print("model compile")
            
            LEARNING_RATE = 1e-4

            #最適化関数
            #optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.RMSprop(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.Adagrad(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.Adadelta(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.Ftrl(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.LBFGS(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.ProximalAdagrad(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.ProximalGradientDescent(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.LARS(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.Adamax(learning_rate=LEARNING_RATE)
            #optimizer = optimizers.Nadam(learning_rate=LEARNING_RATE)


            #学習率最適化1
            steps_per_epoch = int(np.ceil(len(Y_batch) / common.BATCH_SIZE))
            total_steps = steps_per_epoch * common.EPOCHS 
            ALPHA = 0.1
            #lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=DECAY_STEPS, decay_rate=0.96, staircase=True)
            lr_schedule = optimizers.schedules.CosineDecay(initial_learning_rate=LEARNING_RATE, decay_steps=total_steps, alpha=ALPHA)
            #optimizer = optimizers.Adam(learning_rate=lr_schedule)
            optimizer = optimizers.Adamax(learning_rate=lr_schedule)

            #学習率最適化2
            #pip install clr 必要
            #clr = CyclicalLearningRate(initial_learning_rate=LEARNING_RATE, maximal_learning_rate=LEARNING_RATE, step_size=2000, scale_fn='linear', scale_mode='cycle')
            #optimizer = optimizers.Adam(learning_rate=clr)
            
            
            
            #損失関数
            loss = 'categorical_crossentropy'
            #loss = 'binary_crossentropy'
            #loss = 'mean_squared_error'
            #loss = 'mean_absolute_error'
            #loss = 'sparse_categorical_crossentropy'
            #loss = 'huber_loss'
            #loss = 'kullback_leibler_divergence'
            #loss = 'cosine_similarity'
            #loss = 'poisson'
            #loss = 'log_cosh'
            
            #評価指標
            metrics_list = ['accuracy']
            #metrics_list = [Precision()]
            #metrics_list = [Recall()]
            #metrics_list = [AUC()]
            #metrics_list = [MeanAbsoluteError()]
            #metrics_list = [MeanSquaredError()]
            #metrics_list = [RootMeanSquaredError()]
            #metrics_list = [TopKCategoricalAccuracy(k=5)]

            # モデルをコンパイル
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics_list)


            #
            #Set Callback Parameters
            #
            batch_size = common.BATCH_SIZE   # set batch size for training
            epochs = common.EPOCHS   # number of all epochs in training
            patience = 3   #number of epochs to wait to adjust lr if monitored value does not improve
            stop_patience = 10   # number of epochs to wait before stopping training if monitored value does not improve
            threshold = 0.9   # if train accuracy is < threshold adjust monitor accuracy, else monitor validation loss
            factor = 0.5   # factor to reduce lr by
            ask_epoch = 5   # number of epochs to run before asking if you want to halt training
            batches = int(np.ceil(len(Y_batch) / batch_size))    # number of training batch to run per epoch

            callbacks = [MyCallback(model=model,
                         patience=patience,
                         stop_patience=stop_patience,
                         threshold=threshold,
                         factor=factor,
                         batches=batches,
                         epochs=epochs,
                         ask_epoch=ask_epoch)]




            # モデルの訓練
            print("model fit")

            if common.ENABLE_CALLBACKS:
                history = model.fit(
                    X_batch,
                    Y_batch,
                    batch_size=common.BATCH_SIZE,
                    epochs=common.EPOCHS, 
                    validation_data=(X_val, Y_val),
                    callbacks=callbacks
                )
            else:
                history = model.fit(
                    X_batch,
                    Y_batch,
                    batch_size=common.BATCH_SIZE,
                    epochs=common.EPOCHS, 
                    validation_data=(X_val, Y_val),
                )


