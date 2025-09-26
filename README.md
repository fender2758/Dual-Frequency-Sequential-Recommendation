# Dual-Frequency Sequential Recommendation (DuFTRec)

Transformer-based sequential recommender systems have achieved strong results across a wide range of next-item prediction tasks. However, most advances have been calibrated to **sparse interaction logs**.  

As modern platforms accumulate **dense histories** with higher coverage and co-occurrence rates, self-attention exhibits **low-pass characteristics** that degrade separability among items displayed within the same impression set.  

To address this, we present **DuFTRec**, a sequential recommendation model specifically designed for **dense user–item matrices**.  

---

## Datasets

All datasets are stored in the `src/data` folder.

- **Beauty, Sports, Toys, Yelp**: Downloaded from [FMLP-Rec](https://github.com/Woeee/FMLP-Rec).  
- **ML-1M, LastFM**: Processed using the procedure from [S3Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/data/data_process.py).  

Additionally:  
- `src/data/*_same_target.npy` files are used for training **DuoRec** and **FEARec**, which both incorporate contrastive learning.  

---

## 🚀 Quick Start

### 1. Environment Setup

Build the Docker image:
```bash
docker build -t duftrec:latest .
```

### 2. Training DuFTRec

Run the following command:
```bash
python main.py  --data_name [DATASET]                 --lr [LEARNING_RATE]                 --alpha [ALPHA]                 --c [C]                 --num_attention_heads [N_HEADS]                 --train_name [LOG_NAME]
```

**Example (LastFM):**
```bash
python main.py  --data_name LastFM                 --lr 0.001                 --alpha 0.7                 --c 5                 --num_attention_heads 1                 --train_name BSARec_LastFM
```

---

### 3. Testing with Pretrained Models

Place the pretrained model (`.pt` file) in `src/output`.  
Use the model name **without the `.pt` extension**.  

```bash
python main.py  --data_name [DATASET]                 --alpha [ALPHA]                 --c [C]                 --num_attention_heads [N_HEADS]                 --load_model [PRETRAINED_MODEL_NAME]                 --do_eval
```

**Example (LastFM):**
```bash
python main.py  --data_name LastFM                 --alpha 0.7                 --c 5                 --num_attention_heads 1                 --load_model BSARec_Beauty_best                 --do_eval
```

---

### 4. Training Baselines

To train baselines, set the `model_type` argument:

- **Options**: `Caser`, `GRU4Rec`, `SASRec`, `BERT4Rec`, `FMLPRec`, `DuoRec`, `FEARec`  

Check hyperparameters in `src/utils.py` (`parse_args()` function).

```bash
python main.py  --model_type SASRec                 --data_name Beauty                 --num_attention_heads 1                 --train_name SASRec_Beauty
```

---

## 🙏 Acknowledgement

This repository is based on [BSARec](https://github.com/yehjin-shin/BSARec).  
