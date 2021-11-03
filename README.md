[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4972801.svg)](https://doi.org/10.5281/zenodo.4972801)

#  ECCE_DIK - article supplementary

---

## Purpose
This repository serves as a supplementary material for the article "Righteousness in Early Christian Literature: Distant Reading and Textual Networks", published in *Annali di storia dell’esegesi*. 

It contains scripts, data and figures. The scripts are in Python 3 programming language and mainly have a form of Jupyter notebooks. All our analyses aim at being fully reproducible and we invite other scholars to reuse our code and data for their analyses. If you reuse the code, please, use the citation below and refer to the article.


## Citation

Kaše, V., Nikki, N., and Glomb, T. (2021). ECCE_DIK: (Version v1.2). Zenodo. http://doi.org/10.5281/zenodo.4972801

---
## Authors
* Vojtěch Kaše
* Nina Nikki
* Tomáš Glomb

## License
CC-BY-SA 4.0, see attached License.md

---
# How to use this repository

* download or clone the repository
* create virtual environment and connect to it using the first two executible cells in `1_DATA_OVERVIEW.ipynb`
* in all jupyter notebooks, always check that you are connected to the `ecce_venv` kernel
* (alternatively, if you do not wish to use the virtual environment, make sure that you have installed all required python packages within the `requirements.txt` file: `pip install -r requiremnts.txt`)

---
## Scripts 
The scripts are in the `scripts` subfolder and their numbers and titles should be self-explanatory:
* `1_DATA_OVERVIEW.ipynb` offers general overview of the dataset, especially in respect to term frequencies over time.
* `2_COMPARING-AUTHORS.ipynb` analyzes the corpus on a level of individual authors, which are compared by means of several different methods anchored in vector semantics and formal network analysis.
* `textnet.py` contains functions which are used to create, analyze and visualize textual networks.
