from pathlib import Path

config = dict()

config["model_type"] = 'U_Net'  # 'U_Net' 'R2U_Net' 'AttU_Net' 'R2AttU_Net'
config["batch_size"] = 24
config["learning_rate"] = 1e-5
config["epochs"] = 30
config["val_percent"] = 0.2
config["out_nclass"] = 3

config["seed"] = 2022
config["save_checkpoint"] = True
config["in_chn"] = 3

config["recurrent_t"] = 3  # Recurrent step of R2U_Net or R2AttU_Net
config["beta1"] = 0.5   # momentum1 in Adam
config["beta2"] = 0.999 # momentum2 in Adam

config["map_path"] = Path('./data_modified_0604/train/sheets/')
config["label_path"] = Path('./data_modified_0604/train/labels')
config["checkpoint-path"] = Path('./ckp0604/')


 # ****************** FOR Test *****************
config["test-model-type"] = 'U_Net'  # 'U_Net' 'R2U_Net' 'AttU_Net' 'R2AttU_Net'
config["test-checkpoint-path"] = r"D:\SanbornMap\UNet_brick_frame\ckp0604\U_Net_batchsize_24_epoch30.pth"
config["test-img-dir"] = Path('./data_modified_0604/test/sheets')
config["test-gt-dir"] = Path('./data_modified_0604/test/labels')