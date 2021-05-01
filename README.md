# fyp-long-tail-recognition



## Update notifications
* __03/04/2021:__ Configured training settings for Places_LT
* __01/05/2021:__ Update finetuning with knowledge transfer
## Requirements 
* [PyTorch](https://pytorch.org/) (tested with version == 1.7.1)

## Data Preparation


- Download the [ImageNet_2014](http://image-net.org/index) and [Places_365](http://places2.csail.mit.edu/download.html) (256x256 version).
Change the `data_root` in `main.py` to your own directory.

- CIFAR dataset will be automatically downloaded when running CIFAR experiment.


## Getting Started (Training & Testing)


### ImageNet-LT
- Baseline (with k=1)
```
python main.py --m_from 1
```
- cRT (Decoupling)
```
python main.py --config ./config/ImageNet_LT/stage_1_crt.py --path <folder path> --crt
```
- LLW:
```
python main.py --trainable_logits_weight --path <baseline checkpoint directory name> --scaling_logits --config ./config/ImageNet_LT/weight_finetune_resnet10.py --merge_logits
```
- Knowledge Transfer Loss:
```
python main.py --second_dotproduct --m_from 1 --scaling_logits --second_head_alpha 0.2 --temperature 1
```