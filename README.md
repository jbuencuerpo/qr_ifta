# qr_ifta


The implementation of Iterative Fourier Transform Algorithm (IFTA) is under the src/ folder and also preamble.py as typical imports used in the notebooks. Under Notebooks there are example notebook to compare the IFTA vs the  Gaussian random field (GRF). The figures are saved by default under the figure folder.


```
├── figures
├── notebooks
│   ├── 00-Comparison with Gaussian Random Field (Yu 2017).ipynb
│   └── 01-K-Space Desing 100 nm GaAs.ipynb
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── IFTA.py
    └── preamble.py
```    

## Requirements
```
numpy
matplotlib
jupyter
```
## Optional 
```
matplotlib-scalebar (Plotting)
skimage (Saving the QR as .png)
```
[![DOI](https://zenodo.org/badge/274538091.svg)](https://zenodo.org/badge/latestdoi/274538091)

## Publication:
*Engineering the reciprocal space for ultrathin GaAs solar cells*,

Jeronimo Buencuerpo, Jose M.Llorens, Jose M.Ripalda, Myles A.Steiner, and Adele C.Tamboli

(https://doi.org/10.1016/j.optlastec.2021.107224)
