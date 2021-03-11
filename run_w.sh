# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --w1 0.1
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --w1 0.2
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --w1 0.3
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --w1 0.4
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --w1 0.5
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --w1 0.6
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --w1 0.7
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --w1 0.8
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --w1 0.9

# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --scaling_logits --w1 0.1
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --scaling_logits --w1 0.2
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --scaling_logits --w1 0.3
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --scaling_logits --w1 0.4
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --scaling_logits --w1 0.5
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --scaling_logits --w1 0.6
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --scaling_logits --w1 0.7
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --scaling_logits --w1 0.8
# python main.py --test --path ImageNet_LT_90_coslrres50 --config ./config/ImageNet_LT/stage_1_resnet50.py --merge_logits --scaling_logits --w1 0.9
# python main.py --second_dotproduct --m_from 1 --description ablation_second_head_t10 --scaling_logits --second_head_alpha 0.1 --temperature 10
python main.py --second_dotproduct --m_from 1 --description ablation_second_head_t10 --scaling_logits --second_head_alpha 0.5 --temperature 10
python main.py --second_dotproduct --m_from 1 --description ablation_second_head_t10 --scaling_logits --second_head_alpha 0.2 --temperature 10