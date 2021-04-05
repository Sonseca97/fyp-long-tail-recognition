# ----------------03.14 morning expr
# python main.py --second_dotproduct --m_from 1 --description ablation_second_head_t2 --scaling_logits --second_head_alpha 0.2 --temperature 2
# python main.py --second_dotproduct --m_from 1 --description ablation_second_head_t10 --scaling_logits --second_head_alpha 0.2 --temperature 10
# python main.py --second_dotproduct --m_from 1 --description ablation_second_head_t5 --scaling_logits --second_head_alpha 0.2 --temperature 5
# ----------------03.15 night expr different temperature for tail median
# python main.py --second_dotproduct --m_from 1 --description ablation_second_head_distillmediantailt10t1 --scaling_logits --second_head_alpha 0.2 --temperature 10
# ----------------03.16 night resnext50 disitll tail
# python main.py --second_dotproduct --m_from 1 --description resnet50_distilltail --scaling_logits \
#         --second_head_alpha 0.2 --temperature 10  --config ./config/ImageNet_LT/stage_1_resnet50.py --gpu 1
# ---------------------0.20 night expr dynamic momentum update
# python main.py --description dynamic_moving_average --gpu 1
# --------------------3.24 knn true distill only kl loss
 python main.py --second_dotproduct --m_from 1 --description distillonly --scaling_logits --second_head_alpha 0 --temperature 1
test