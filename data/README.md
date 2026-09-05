# Dataset Repository & Download Instructions

This folder contains network traffic datasets used for training and evaluating intrusion detection models.

## Datasets Supported

### 1. KDD Cup 1999 / NSL-KDD Dataset
- **Description**: Standard benchmark dataset for evaluating network intrusion detection systems. Contains 41 traffic features per sample and 40 connection classes (Normal + 39 attack types across DoS, Probe, R2L, and U2R categories).
- **Download**:
  - Download `kddcup.data_10_percent.gz` or `NSL-KDD` from the official UNB website: [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html).
  - Extract and place `kdd_dataset.csv` or `kdd_reduced.csv` into this `data/` directory.

### 2. CICIDS-2017 Dataset
- **Description**: Modern network intrusion dataset capturing benign and recent common attacks (PortScan, DDoS, Web Attacks, Infiltration) collected over 5 working days.
- **Download**:
  - Download PCAP / CSV files from [UNB CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html).
  - Extract CSV files into `data/CICIDS-2017/`.

### 3. Synthetic Test Generator
- If raw external datasets are unavailable, you can instantly generate a synthetic test dataset using the built-in script:
  ```bash
  python dataset_generator.py
  ```
  This creates `data/synthetic_test_dataset.csv` with 10,000 samples and customizable attack ratios.
