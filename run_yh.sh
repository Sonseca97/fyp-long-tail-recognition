# Instance-Balanced sampling
python main.py --config ./config/ImageNet_LT/stage_1_resnet50.py --gpu 0,1
# Fintune logits weight (no scaling_logits)
python main.py --config ./config/ImageNet_LT/stage_1_resnet50.py --trainable_logits_weight --path e90 --merge_logits
# Finetune logits weight (scaling_logits)
python main.py --config ./config/ImageNet_LT/stage_1_resnet50.py --trainable_logits_weight --path e90 --merge_logits --scaling_logits