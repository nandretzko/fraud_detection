# 🔍 Credit Card Fraud Detection

Détection de fraude par carte de crédit via un pipeline ML industrialisé, basé sur un kaggle que j'avais effectué.

## Structure du projet

```
fraud-detection/
├── src/
│   ├── train.py          # Pipeline d'entraînement complet
│   └── predict.py        # Script d'inférence (scoring de nouvelles transactions)
├── data/                 # ← Placer fraudTrain.csv et fraudTest.csv ici
├── models/               # Artefacts générés après entraînement (.pkl)
├── outputs/              # Prédictions générées
├── Dockerfile            # Image Docker pour industrialisation
├── Makefile              # Commandes courantes
├── requirements.txt      # Dépendances Python
└── README.md
```

## Données attendues

Placer dans `data/` les fichiers CSV suivants (issus de Kaggle – [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)) :

| Fichier | Description |
|---|---|
| `fraudTrain.csv` | Transactions d'entraînement (~1.3M lignes) |
| `fraudTest.csv` | Transactions de test (~555K lignes) |

Colonnes principales : `trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `gender`, `lat`, `long`, `city_pop`, `dob`, `merch_lat`, `merch_long`, **`is_fraud`** (target).

## Features construites

Le pipeline reproduit fidèlement le notebook :

- **`age`** : âge du client calculé à partir de `dob` et de la date de transaction
- **`distance`** : distance euclidienne (degrés) entre le client et le commerçant
- **`hour`** / **`day_of_week`** : variables temporelles
- **`gender`** : encodage binaire (F=0, M=1)
- **`category_*`** : one-hot encoding de la catégorie de transaction

## Usage rapide – local

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
# ou via Make :
make install
```

### 2. Entraîner le modèle

```bash
make train
# ou directement :
python src/train.py \
  --train data/fraudTrain.csv \
  --test  data/fraudTest.csv \
  --model-dir models \
  --threshold 0.01
```

Artefacts générés dans `models/` :
- `logistic_model.pkl` – modèle Logistic Regression entraîné
- `scaler.pkl` – StandardScaler ajusté sur le train
- `feature_cols.pkl` – liste ordonnée des features

### 3. Scorer de nouvelles transactions

```bash
make predict INPUT=data/fraudTest.csv
# ou :
python src/predict.py \
  --input data/new_transactions.csv \
  --model-dir models \
  --output outputs/scored.csv \
  --threshold 0.01
```

## Usage Docker

### Build

```bash
make docker-build
```

### Entraînement dans Docker

```bash
make docker-train
```

(Monte automatiquement `data/`, `models/` et `outputs/` en volumes locaux.)

### Inférence dans Docker

```bash
make docker-predict INPUT=data/fraudTest.csv
```

## Paramètres configurables

| Paramètre | Défaut | Description |
|---|---|---|
| `--train` | `data/fraudTrain.csv` | Chemin du CSV d'entraînement |
| `--test` | `data/fraudTest.csv` | Chemin du CSV de test |
| `--model-dir` | `models` | Répertoire de sauvegarde des artefacts |
| `--threshold` | `0.01` | Seuil de classification (sensible aux fraudes) |
| `--C` | `1.0` | Régularisation Logistic Regression |
| `--max-iter` | `1000` | Nombre max d'itérations du solver |

> **Note sur le seuil** : le dataset est très déséquilibré (~0.6% de fraudes). Un seuil bas (0.01) maximise le recall aux dépens de la précision, ce qui est souvent préférable en détection de fraude.

## Modèle

**Logistic Regression** avec `class_weight='balanced'` pour gérer le déséquilibre de classes. Le modèle produit des probabilités de fraude pour chaque transaction.

Métriques reportées :
- ROC-AUC
- Average Precision (PR-AUC)
- Classification report complet (precision, recall, F1)
- Sweep de seuils : 0.01, 0.05, 0.1, 0.5
