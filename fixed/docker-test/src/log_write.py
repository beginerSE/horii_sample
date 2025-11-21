

import os
#
# common
#
import common


def log_write(
    str_time,
    now,
    end,
    elapsed_time,
    scores,
    model,
    output_path
):




    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = f"{output_path}log__{str_time}.txt"


    with open(output_file, 'w', encoding='utf-8') as f:
        #
        # log write
        #
        f.write("\n\n")
        f.write(f"start   : {now} \n")
        f.write(f"end     : {end} \n")
        f.write(f"処理時間(秒換算): {elapsed_time}  \n")
        f.write(f"処理時間(分換算): {elapsed_time/60} \n")
        f.write(f"処理時間(時換算): {elapsed_time/(60*60)} \n")
        f.write(f"処理時間(日換算): {elapsed_time/(60*60*24)} \n")
        f.write("\n")

        f.write(f"CLASS_LIST: {common.CLASS_LIST}  \n")
        f.write(f"NUM_CLASS  : {common.NUM_CLASS}  \n")
        f.write(f"IMG_WIDTH  : {common.IMG_WIDTH}  \n")
        f.write(f"IMG_HEIGHT : {common.IMG_HEIGHT}  \n")
        f.write("\n")
        f.write(f"TRAIN_PATH : {common.TRAIN_PATH}  \n")
        f.write(f"VAL_PATH   : {common.VAL_PATH}  \n")
        f.write(f"TEST_PATH  : {common.TEST_PATH}  \n")
        f.write("\n")
        f.write(f"NUM_OF_TRAIN: {common.NUM_OF_TRAIN}  \n")
        f.write(f"NUM_OF_VAL  : {common.NUM_OF_VAL}  \n")
        f.write(f"NUM_OF_TEST : {common.NUM_OF_TEST}  \n")
        f.write(f"NUM_OF_TEST_FOR_PREDICTION: {common.NUM_OF_TEST_FOR_PREDICTION}  \n")
        f.write("\n")
        f.write(f"BATCH_SIZE: {common.BATCH_SIZE}  \n")
        f.write(f"EPOCHS    : {common.EPOCHS}  \n")
        f.write("\n")
        f.write("Test Data Evaluation\n")
        if scores is None:
            f.write("scores is None. \n")
        else:
            f.write(f"Test loss    : {scores[0]}  \n")
            f.write(f"Test accuracy: {scores[1]}  \n")
        f.write("\n")
        f.write("model.summary\n")
        try:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
        except Exception as e:
            f.write(f"Error in model summary: {str(e)}\n")
        f.write("\n")
        
        print(f"Log saved at {output_file}")





