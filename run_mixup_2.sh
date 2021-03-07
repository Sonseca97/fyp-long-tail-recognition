# python main.py --trainable_logits_weight --config ./config/ImageNet_LT/weight_finetune_resnet50.py --path ImageNet_LT_90_coslrres50 --merge_logits
# python main.py --trainable_logits_weight --config ./config/ImageNet_LT/weight_finetune_resnext50.py --path ImageNet_LT_90_coslrresnext50 --merge_logits
# python main.py --trainable_logits_weight --config ./config/ImageNet_LT/weight_finetune_resnet10.py --path ImageNet_LT_90_coslr --merge_logits
python main.py --alpha 0.1 --description ablation_alpha
python main.py --alpha 0.01 --description ablation_alpha
python main.py --alpha 0.001 --description ablation_alpha