# Dual-Frequency Sequential Recommendation (DuFTRec)

Transformer-based sequential recommender system have achieved strong results across a wide range of next-item prediction tasks. However, most advances have been calibrated to sparse interaction logs. As modern platforms accumulate dense histories with higher coverage and co-occurrencerates, self-attention exhibits low-pass characteristics that degrade separability among items displayed within the same impression set. In this paper, we present DuFTRec a sequential recommendation model specifically designed to handle dense user-item matrices. The proposed model introduces density aware spectralattention. Sequences are decomposed into low-and high-frequency components, each learned with band-specific self attention, then fused by a density-conditioned gate to avoid scale mixing. This design preserves fine-grained signals in dense interaction matrices and strengthens separability within impression sets. Our model consistently improves top-𝐾 ranking quality(e.g., NDCG@K,HR@K) over strong Transformer baselines in dense regimes while maintaining competitive complexity.

---

## Datasets

All datasets are stored in the `src/data` folder.

- **Yelp**: Downloaded from [this repository](https://github.com/Woeee/FMLP-Rec).  
- **ML-1M, LastFM**: Processed using the procedure from [this code](https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/data/data_process.py).  

Additionally:  
- `src/data/*_same_target.npy` files are used for training **DuoRec** and **FEARec**, which both incorporate contrastive learning.  
### Dataset Statistics and Density Indicators

| Dataset | #Users | #Items | #Inter. | Avg. len | Sparsity |
|---------|--------|--------|---------|----------|----------|
| ML-1M   | 6,041  | 3,417  | 999,611 | 165.5    | 95.16%   |
| LastFM  | 1,090  | 3,646  | 52,551  | 48.2     | 98.68%   |
| Yelp    | 30,431 | 20,033 | 316,354 | 10.4     | 99.95%   |
---

## Quick Start

### 1. Environment Setup

Build the Docker image:
```bash
docker build -t duftrec:latest .
```

### 2. Training DuFTRec

Run the following command:
```bash
python main.py  --data_name [DATASET]
                --lr [LEARNING_RATE]
                --alpha [ALPHA]
                --c [C]
                --num_attention_heads [N_HEADS]
                --train_name [LOG_NAME]
```

**Example (ML-1M):**
```bash
python main.py  --data_name ML-1M
                --lr 0.001
                --alpha 0.7
                --c 5
                --num_attention_heads 1
                --train_name BSARec_ML-1M
```

---

### 3. Testing with Pretrained Models

Place the pretrained model (`.pt` file) in `src/output`.  
Use the model name **without the `.pt` extension**.  

```bash
python main.py  --data_name [DATASET]
                --alpha [ALPHA]
                --c [C]
                --num_attention_heads [N_HEADS]
                --load_model [PRETRAINED_MODEL_NAME]
                --do_eval
```

**Example (LastFM):**
```bash
python main.py  --data_name ML-1M
                --alpha 0.7
                --c 5
                --num_attention_heads 1
                --load_model BSARec_ML-1M_best
                --do_eval
```

---

### 4. Training Baselines

To train baselines, set the `model_type` argument:

- **Options**: `Caser`, `GRU4Rec`, `SASRec`, `BERT4Rec`, `FMLPRec`, `DuoRec`, `FEARec`  

Check hyperparameters in `src/utils.py` (`parse_args()` function).

```bash
python main.py  --model_type SASRec
                --data_name ML-1M
                --num_attention_heads 1
                --train_name SASRec_ML-1M
```

---

## Acknowledgement

This repository is based on [BSARec](https://github.com/yehjin-shin/BSARec).  
