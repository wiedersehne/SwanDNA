# SwanDNA
## Variant Effect Prediction. 
### To pretrain a model you need to follow the steps:
1. Download GRCH38 from http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz. (3.1G)
2. Run **generate_pretrain_human.py** in `./data/`. Sequence length [1k, 5k, 10k, 20k] and numbers(200k) are required. You need to load two data files, **hg38.fa** and **chromosomes.csv**, for this task.
3. Run **pretraining.py** with the generated data. Configurations of different lengths shall be changed accordingly in **config.yaml**.
### To fine-tune a pretrained model, you need to:
1. Run **generate_ve_data.py** in `./data/` to save data. The sequence length is required. A total of 97,922 sequence will be extracted. **ve_df.csv** needs to be loaded.
2. Run **ve_classification.py** to load the pretrained model under the folder `./Pretrained_models/` and train SwanDNA.
### Experimental Results

| lengths | SwanDNA w/o | Enformer | DeepSEA | Nystromformer | Linformer | Transformer | Mega | S4 |
| ------- | ----------- | -------- | ------- | ------------- | --------- | ----------- | ---- | -- |
| 1kbp    | 70.74       | 64.13    | 52.93   | 70.86         | 62.5      | 52.93       | **71.48** | 69.26 |
| 5kbp    | **71.04**   |          | /       | 70.23         | 59.68     | 52.93           | 56.60 | 65.80 |
| 10kbp   | **71.61**   |          | /       | 69.12         | 60.05     | /           | 51.98 | 62.13 |
| 20kbp   | **71.90**   |          | /       | 50.00         | 50.00     | /           | 50.01 | 52.21 |

## OCRs prediciton in plants.
### To pretrain a model you need to follow the steps:
1. Download reference genome files from https://plantdeepsea-toturial2.readthedocs.io/en/latest/08-Statistics.html
2. Run **plant_download.ipynb** in `./data/plant_data/`to download training data.
3. Run **run.sh** in `./data/plant_generate/` to save data. The sequence length is required.
4. Run **pretraining.py** with the generated data. Configurations of different lengths shall be changed accordingly in **config_plant.yaml**.
### To fine-tune a pretrained model, you need to:
1. Run **run.sh** in `./data/plant_generate/` to generate data. The plant name is required.
2. Run **plant_classification.py** to load the pretrained model under the folder `./Pretrained_models/` and train SwanDNA.

### Experimental Results
| Plant               | A.thaliana | B.distachyon | O.sativa-MH | O.sativa-ZS | S.bicolor | S.italica | Z.mays |
| ------------------- | ---------- | ------------ | ----------- | ----------- | --------- | --------- | ------ |
| Number of OCR labels                | 19         | 9            | 15          | 15          | 14        | 9         | 19     |
| DeepSEA             | 92.02      | 92.88        | 92.95       | 92.19       | 96.24     | 94.04     | 96.64  |
| Nystromformer       | 89.22      | 90.86        | 89.08       | 88.10       | 94.50     | 91.61     | 90.74  |
| Linformer           | 70.56      | 83.50        | 79.28       | 80.43       | 87.30     | 84.64     | 80.82  |
| Transformer         | 64.96      | 82.53        | 78.79       | 78.62       | 85.15     | 84.24     | 63.02  |
| Mega                | 85.37      | 88.68        | 85.43       | 85.51       | 91.99     | 88.41     | 84.74  |
| S4                  | 85.82      | 90.70        | 88.30       | 87.84       | 93.95     | 90.84     | 92.87  |
| SwanDNA w/o         | 92.09      | 93.15        | 92.85       | 92.15       | 96.32     | 93.98     | 96.64  |
| SwanDNA w/ (1kbp)   | 92.24      | 93.57        | 93.42       | 92.81       | 96.41     | 94.33     | 97.07  |
| SwanDNA w/ (10kbp)  | 92.45      | 93.77        | 93.70       | 93.11       | 96.74     | 94.71     | 97.21  |
| SwanDNA w/ (50kbp)  | 92.81      | 93.79        | 93.83       | 93.28       | 96.68     | 94.79     | 97.31  |
| SwanDNA w/ (100kbp) | **93.22**  | **94.10**    | **93.99**   | **93.56**   | **96.88** | **95.08** | **97.32** |

## Public Benchmark: GenomicBechmarks.
### To pretrain a model you need to follow the steps:
1. Download GRCH38 from http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz. (3.1G)
2. Run **generate_pretrain_human.py** in `./data/`. Sequence length (100k) and numbers (200k) are required.
3. Run **pretraining.py** with the generated data. Configurations of different lengths shall be changed accordingly in **config_gb.yaml**.
The hyperparameters of pretraining is in supplementaty document.
### To fine-tune a pretrained model and conduct classification of GenomicBenchmarks, you need to:
1. run **genomic_benchmark.py** in `./data/` to download the datasets. You need to install the bechmark using ```pip install genomic-benchmarks```. The details of the datasets are shown in the table below. More details can be found in their github, https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks.

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

2. Run **genomic_classification.py** to load the pretrained model under the folder **Pretrained_models/SwanDNA_GRCH38_100000_144_256.pt** and train SwanDNA. More specifically, you need firstly to choose a task name from the list below.
```
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
```
Then, specify the task in the main function. The following example shows how to run human_ocr_ensembl: ```classify_main(cfg, "human_ocr_ensembl")```
The optimal hyperparameters for each dataset are fixed in **config_gb.yaml**. You can also refer to the supplementary document.

### Experimental Results
| Dataset                  | CNN | Transformer | HyenaDNA | SwanDNA |
| ------------------------ | ------- | --------------- | ------------ | ------- |
| Mouse Enhancers          | 69.0    | 80.1            | 84.3         | **85.95** |
| Coding vs Intergenomic   | 87.6    | 88.8            | 87.6         | **92.85** |
| Human vs Worm            | 93.0    | 95.6            | 96.5         | **96.65** |
| Human Enhancers Cohn     | 69.5    | 70.5            | 73.8         | **73.97** |
| Human Enhancers Ensmbl   | 68.9    | 83.5            | 89.2         | **90.32** |
| Human Regulatory         | 93.3    | 91.5            | 93.8         | **94.04** |
| Human Nontata Promoters  | 84.6    | 87.7            | 96.6         | **97.62** |
| Human OCR Ensembl        | 68.0    | 73.0            | **80.9**     | 77.52   |
| Average                  | 79.2    | 83.8            | 87.8         | **88.62** |
