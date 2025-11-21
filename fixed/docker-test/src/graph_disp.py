

import matplotlib.pyplot as plt

import common




def graph_disp(
    history, 
    str_time,
    sw_plt_show,
    output_path
):


    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    
    train_acc = history.history['accuracy']
    val_acc   = history.history['val_accuracy']
    
    #epochs = len(train_loss)
    epochs = range(1, len(train_loss) + 1)
    
    
    # グラフ作成
    fig, ax1 = plt.subplots()

    # 左側のy軸に対してプロット
    ax1.plot(epochs, train_loss, 'g-', label='train_loss', marker='o')
    ax1.plot(epochs, val_loss,   'r-', label='val_loss',   marker='x')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    # x軸の目盛りを整数に設定
    ax1.set_xticks(range(1, len(train_loss) + 1))
    ax1.set_ylim(bottom=0)
    
    
    # 右側のy軸を作成
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_acc, 'b-',     label='train_acc', marker='s')
    ax2.plot(epochs, val_acc,   'purple', label='val_acc',   marker='d')
    ax2.set_ylabel('Accuracy', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(bottom=0, top=1)
    
    
    # グラフのタイトルとレジェンド
    ax1.set_title('Loss and Accuracy over Epochs')
    fig.tight_layout()  # レイアウト調整
    # グラフ領域を広げるために、余白をさらに調整
    fig.subplots_adjust(right=0.65)  # 右側の余白を広げる
    
    
    # レジェンドを表示
    #ax1.legend(loc='upper left')
    #ax2.legend(loc='upper right')
    # 凡例をグラフ外に移動
    ax1.legend(loc='upper left', bbox_to_anchor=(1.2, 1))
    ax2.legend(loc='upper left', bbox_to_anchor=(1.2, 0.8))
    
    
    img_name = f"{output_path}history_acc_loss__{str_time}.jpg"
    #plt.savefig(img_name)
    
    # acc_loss画像を保存
    try:
        plt.savefig(img_name)
        print(f"history_acc_loss saved at {img_name}")
    except Exception as e:
        print(f"Error saving history_acc_loss to {img_name}: {e}")

    if(sw_plt_show):
        plt.show()
    plt.close()





