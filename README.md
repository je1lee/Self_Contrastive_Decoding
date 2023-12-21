# Self_Contrastive_Decdoing


### Example
---
```
CUDA_VISBLE_DEVICES=0 python gen_example.py --prompt Barrack Obama was born in Honolulu, Hawaii. He was born in --model_path facebook/opt-13b --max_new_tokens 128
```

### evaluation
---
Self Contrastive Decoding
```
CUDA_VISIBLE_DEVICES=0 python run.py --number_of_samples 1000 --dataset wikinews --model_path facebook/opt-13b/ --contrastive True --outside False
```
Contrastive Decoding
```
CUDA_VISIBLE_DEVICES=2 python run.py --number_of_samples 1000 --dataset wikinews --model_path facebook/opt-13b/ --contrastive True --outside True
```
Argmax Decoding
```
CUDA_VISIBLE_DEVICES=0 python run.py --number_of_samples 1000 --dataset wikinews --model_path facebook/opt-13b
```

