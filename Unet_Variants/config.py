from pathlib import Path

config = dict()

config["model_type"] = 'R2AttU_Net'  # 'U_Net' 'R2U_Net' 'AttU_Net' 'R2AttU_Net'
config["batch_size"] = 4
config["learning_rate"] = 1e-7
config["epochs"] = 10
config["val_percent"] = 0.2
config["out_nclass"] = 2

config["seed"] = 2022
config["save_checkpoint"] = True
config["in_chn"] = 3

config["recurrent_t"] = 3  # Recurrent step of R2U_Net or R2AttU_Net
config["beta1"] = 0.5   # momentum1 in Adam
config["beta2"] = 0.999 # momentum2 in Adam

config["map_path"] = Path('./data/train/tif/')
config["label_path"] = Path('./data/train/label/')
config["checkpoint-path"] = Path('./checkpoints0427/')


 # ****************** FOR Test *****************
config["test-model-type"] = 'R2AttU_Net'  # 'U_Net' 'R2U_Net' 'AttU_Net' 'R2AttU_Net'
config["test-checkpoint-path"] = r"D:\SanbornMap\UNet_family\checkpoints0427\R2AttU_Net_batchsize_4_epoch10.pth"
config["test-img-dir"] = Path('./data/test/tif')
config["test-gt-dir"] = Path('./data/test/label')