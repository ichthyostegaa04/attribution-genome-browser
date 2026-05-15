# attribution-genome-browser
Explainable AI pipeline for genotype-to-phenotype models by mapping attribution tensors onto chromosomal coordinates to identify significant genomic peaks associated with model prediction. 

## Directory structure

Place **exactly one file** in each input folder:

```
IGV/
├── IGV.ipynb
│
├── tensor/          ← your .npy tensor   (samples × SNPs × classes)
├── bed_file/        ← your .bed file     (must contain header)
├── annotation/      ← annotation.gtf
├── class/           ← class.tsv  (tsv file where each lines represents one class name that's in the same order that classes are ordered in tensor)
│
└── chr_plots/       ← output PNGs  (auto-created on first run)
```

The notebook auto-detects all files — no path editing needed.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/attribution-genome-browser.git
cd /path_to_IGV_folder
```
NOTE: IGV should be your current directory

### 2. Install dependencies

```bash
pip install numpy pandas scipy matplotlib statsmodels ipywidgets
```

### 3. Download gene annotation (one-time)

Download GTF anottation file and put it in annotation folder. There's already a reduced annotation file present for demonstration purposes, but if you wish to load a full annotation file, replace the current annotation file with your own.

To use existing file:
gunzip ./annotation/annotation_reduced.gtf.gz

### 4. Add your data

```
tensor/     →  your_tensor.npy     shape: (samples, SNPs, classes)
bed_file/   →  your_snps.bed       columns: (header, tab-separated)
```
There're already tensor and bed file present for demonstration purposes. Replace with your own.

To use existing tensor:
unzip tensor.npz && rm tensor.npz

> The tensor and BED must have the **same SNP order**: axis 1 of the tensor = rows of the BED file.

### 5. Configure & run

Open `IGV.ipynb` in Jupyter.  


Then **Kernel → Restart & Run All**.

> **Any number of classes is supported.** If `CLASS_NAMES` doesn't match the tensor,  
> labels and colors are auto-generated automatically.

---

## Output

| Output | Description |
|---|---|
| `chr_plots/chr1.png` … `chr22.png` | Smoothed attribution profiles per chromosome, dark background |
| Peak table | Significant peaks per chromosome per class (FDR & IDR, printed in notebook) |
| IGV browser | Interactive scrollable viewer (inside the notebook) |

---

## Methods

### Smoothing — Gaussian on genomic grid
SNP-level attributions are projected onto a uniform genomic grid (default **100 kb** resolution) via linear interpolation, then smoothed with a 1D Gaussian filter (**σ = 7** grid units = 700 kb). This ensures the smoothing kernel is physically uniform regardless of local SNP density.

### Peak detection — FDR
Candidate peaks are found with `scipy.signal.find_peaks`. Each peak's z-score is converted to a two-tailed p-value and corrected with **Benjamini-Hochberg** at α = 0.05.

### Peak detection — IDR (multi-scale rank stability)
For each candidate peak, its normalised rank is computed within windows of increasing genomic size (**80 kb → 10 Mb**, step 500 kb). A peak is significant if it ranks in the top 20% on at least **30% of tested scales**. This rewards peaks dominant at multiple resolutions, reducing false positives from isolated noise spikes.

The method yielding more significant peaks is automatically selected for IGV display.

---

## Parameters (Section 0)

| Parameter | Default | Description |
|---|---|---|
| `GRID_STEP` | 100 000 bp | Genomic grid resolution |
| `SIGMA` | 7 | Gaussian sigma (grid units) |
| `FDR_ALPHA` | 0.05 | BH significance threshold |
| `FDR_HEIGHT_MULT` | 0.5 | FDR peak height = mean + mult × std |
| `IDR_HEIGHT_MULT` | 0.3 | IDR candidate height threshold |
| `IDR_SCALE_MIN_KB` | 80 | Smallest IDR window (kb) |
| `IDR_SCALE_MAX_MB` | 10 | Largest IDR window (Mb) |
| `IDR_SCALE_STEP_KB` | 500 | IDR window step (kb) |
| `IDR_MIN_FRACTION` | 0.3 | Min fraction of scales for IDR significance |
| `IDR_RANK_THRESH` | 0.80 | Normalised rank threshold |
| `IGV_DEFAULT_WINDOW` | 5 000 000 bp | Default IGV view width |

---

## Input format details

**Tensor** (`.npy`):
- Shape: `(n_samples, n_snps, n_classes)`
- Values: multiplicative attribution scores (baseline ≈ 1.0, e.g. raw Integrated Gradients)

**BED file** (tab-separated, header):
```
chrom   start     end
chr1    752566    752567
chr1    768448    768449
chr2    491575    491576
...
```
Your bed file must have a header.

**GTF annotation**: standard GENCODE GTF filtered to `gene` feature rows only.

---

## License

MIT
