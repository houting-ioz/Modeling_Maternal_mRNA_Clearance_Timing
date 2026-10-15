# Modeling Maternal mRNA Clearance Timing

Code and data for modeling maternal mRNA clearance timing using sequence-derived and molecular features. 

## Repository structure

```text
.
├── data/                       # Input table data 
├── dl_classify/                # MLP/CNN/Transformer/BiLSTM training code
├── xgboost_classify/           # Binary and three-class XGBoost code
│   └── draw/                   # Figure-generation scripts
├── outputs/                    
├── requirements.txt            
└── README.md
```

## Data

The checked-in `data/` directory includes:

| File | Role indicated by filename / code |
| --- | --- |
| `binary_tab.csv` | feature and label table for binary classification |
| `3class_tab.csv` | feature and label table for three-class classification |
| `seqs_pc_3utr.csv`, `seqs_pc_cds.csv` | 3′ UTR and CDS sequence joined by `gene_name` |
| `gene_3utr_mirna_summary.csv` | Mean 3′ UTR binding strength of the 20 most highly expressed oocyte-stage miRNAs |
| `gene_mirna_20d_features.csv` | Expression levels of the top 20 miRNAs during the oocyte stage that bind to the 3′ UTR |
| `cluster_gene_te_gene_level.tsv` | Transposable element counts per gene, by class and in total. |
| `molecular_features.xlsx` | Raw data table |

The scripts merge the relevant CSV/TSV tables on `gene_name` and expect labels in a `label` column. 

## Installation

Install the dependencies listed in [`requirements.txt`](requirements.txt). 

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

##### Run these scripts with `xgboost_classify/` ：

```bash
cd xgboost_classify
python run_binary.py
python run_3class.py
```

##### Run these scripts with `dl_classify/` ：

Before training this model, generate the mRNA sequence embeddings with GLMs. Save the resulting embeddings as `.h5` files in a directory, then pass that directory to the script using `--pt_path`. The script used to generate the embeddings will be uploaded at a later date.

```bash
cd dl_classify

python mlp_binary1D.py

python dlModel_binary2D.py --epochs 10 --batchsize 32 --encoder_type transformer --pool_type mean --in_dim 3072 --layers 2 --output_dir output_dir_g --pt_path /your_embedding_path
```

## Citation and license
