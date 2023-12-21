# Self_Contrastive_Decdoing


### Example
---
```
```

### evaluation
---
Self Contrastive Decoding
```
CUDA_VISIBLE_DEVICES=0 python run.py --number_of_samples 1000 --dataset wikinews --model_path facebook/opt-13b/ --contrastive True --outside False
```
Contrastive Decoding
```
CUDA_VISIBLE_DEVICES=0 python run.py --number_of_samples 1000 --dataset wikinews --model_path facebook/opt-13b
```

