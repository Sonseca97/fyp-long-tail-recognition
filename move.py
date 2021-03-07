import os 
import shutil

root_path = './logs/ImageNet_LT/stage1'
for ckpt in os.listdir(root_path):
    ckpt_path = os.path.join(root_path, ckpt)
    if os.path.isdir(ckpt_path):
        model_path = ckpt_path + '.pth'
        if os.path.exists(model_path):
            model_name = model_path.split('/')[-1]
            shutil.copy(model_path, os.path.join(ckpt_path, model_name))
        model_path = ckpt_path + '.pkl'
        if os.path.exists(model_path):
            model_name = model_path.split('/')[-1]
            shutil.copy(model_path, os.path.join(ckpt_path, model_name))
        model_path = ckpt_path + '_final.pkl'
        if os.path.exists(model_path):
            model_name = model_path.split('/')[-1]
            shutil.copy(model_path, os.path.join(ckpt_path, model_name))
    