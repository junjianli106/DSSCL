
## DSSCL: Domain-Informed Self-Supervised Contrastive Learning for Pathological Image Analysis

This is a PyTorch implementation of DSSCL (Domain-specific Self-Supervised Contrastive Learning) for histopathological image analysis.

### Preparation

Download the NCT-CRC-HE-100K dataset: [NCT dataset](https://zenodo.org/record/1214456).

### Unsupervised Pretraining

To perform unsupervised pretraining on NCT-CRC-HE-100K using DSSCL in a multi-GPU setup (2x A100 40G), run:

```
python H_H_prime_generate.py
```

This step generates augmented images using stain-separation based data augmentation, preparing the dataset for contrastive learning.

Next, run the following to pretrain the model using the generated data:

```
python main_w_he.py --weight_orig 0.4 --num-cluster '10'
```


### Results on NCT-CRC-HE-100K

Fine-tuning results on NCT-CRC-HE-100K using 0.1%, 1%, 10%, and 100% labeled data:

| Metric | 0.1% Labeled | 1% Labeled | 10% Labeled |
|--------|--------------|------------|-------------|
| **ACC** | 88.0 ± 1.3  | 93.0 ± 0.3 | 93.6 ± 0.1  | 
| **F1**  | 82.6 ± 1.8  | 89.4 ± 0.3 | 91.0 ± 0.2  | 



### License

This project is licensed under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
