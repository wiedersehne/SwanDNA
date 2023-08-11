# DNASWAN
## Varaint Effect Prediction. 
### To pretrain a model you need to follow the steps:
1. Download GRCH38 from http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz. (3.1G)
2. Run **generate_pretrain_human.py**. Sequence length [1k, 5k, 10k, 20k] and numbers(100k) are required.
3. Run **pretraining.py** with the generated data. Configurations of different lengths shall be changed accordingly in config.yaml.
### To fine-tune a pretrained model, you need to:
1. Run **generate_ve_data.py** to save data. Sequence lengths is required. A total of 97,922 sequence will be extracted.
2. Run ve_classification.py to load the pretrained model under the folder "Pretrained_models" and train DNASwan.

## Public Benchmark: GenomicBechmarks.
### To pretrain a model you need to follow the steps:
1. Download GRCH38 from http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz. (3.1G)
2. Run **generate_pretrain_human.py**. Sequence length (100k) and numbers (200k) are required.
3. Run **pretraining.py** with the generated data. Configurations of different lengths shall be changed accordingly in config.yaml.
The hyperparameters of pretraining is in supplementaty document.
### To fine-tune a pretrained model and conduct classification of GenomicBenchmarks, you need to:
1. run genomic_benchmark.py to download the datasets. The details of the datasets are shown in the table below.

| Dataset                 | Length Range | Median | Train Num | Test Num | Classes |
|-------------------------|--------------|--------|----------|----------|---------|
| Mouse Enhancers         | 331-4776     | 2381   | 1210     | 242      | 2       |
| Coding vs Intergenomic  | 200          | /      | 75000    | 25000    | 2       |
| Human vs Worm           | 200          | /      | 75000    | 25000    | 2       |
| Human Enhancers Cohn    | 500          | 500    | 20843    | 6948     | 2       |
| Human Enhancers Ensembl | 2-573        | 269    | 123872   | 30970    | 2       |
| Human Regulatory        | 71-802       | 401    | 231348   | 57713    | 3       |
| Human Nontata Promoters | 251          | /      | 27097    | 9034     | 2       |
| Human OCR Ensembl       | 71-593       | 315    | 139804   | 34952    | 2       |

2. Run genomic_classification.py to load the pretrained model under the folder "Pretrained_models" and train DNASwan. You need to choose a task name from 
**
task_names = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "demo_human_or_worm",
    "demo_mouse_enhancers",
    "demo_coding_inter",
    "drosophila_enhancers_stark",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_ocr_ensembl"
]
**
The optimal hyperparameters for each dataset are fixed in **config_gb.yaml**.
