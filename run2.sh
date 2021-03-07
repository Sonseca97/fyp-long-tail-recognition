# for i in {1..4} 
# do python main.py --ce_dist --not_cal_center --tensorboard --dloss cos_similarity --use_direct_f --w1 $i --description $i
# done

# for i in {3..4}
# do 
#     python main.py --test --w1 1 --w2 1 --path ImageNet_LT_90_baseline_just_check_knn --merge_logits --scaling_logits
# done
# python main.py --w1 0.5 --w2 0.5 --description w1_0.5_w2_0.5_cos --merge_logits
# python main.py --w1 0.5 --w2 0.5 --description w1_0.5_w2_0.5_l2 --merge_logits --dist_type l2
python main.py --test --w1 1 --w2 4 --path ImageNet_LT_90_baseline_just_check_knn --merge_logits --scaling_logits
python main.py --test --w1 1 --w2 5 --path ImageNet_LT_90_baseline_just_check_knn --merge_logits --scaling_logits
python main.py --test --w1 1 --w2 6 --path ImageNet_LT_90_baseline_just_check_knn --merge_logits --scaling_logits
python main.py --test --w1 1 --w2 7 --path ImageNet_LT_90_baseline_just_check_knn --merge_logits --scaling_logits
python main.py --test --w1 1 --w2 8 --path ImageNet_LT_90_baseline_just_check_knn --merge_logits --scaling_logits
python main.py --test --w1 1 --w2 3 --path ImageNet_LT_90_baseline_just_check_knn --merge_logits --scaling_logits
python main.py --test --w1 1 --w2 1 --path ImageNet_LT_90_baseline_just_check_knn --merge_logits --scaling_logits
python main.py --test --w1 1 --w2 3 --path ImageNet_LT_90_baseline_just_check_knn --merge_logits --scaling_logits

