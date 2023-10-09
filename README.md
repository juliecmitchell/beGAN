beGAN is a code for generating beta hairpin sequences of variable residue lengths like 14-mer, 16-mer, 18-mer and 20-mer peptides. 
It is provided as a Jupyter notebook and as a python file. 
To run either, there are a number of dependencies, including pytorch, numpy, pandas, propy, etc.

To create the beGAN environment and activate it using conda

*`conda create --name beGAN`


`conda activate beGAN`

To obtain the beGAN code repository in your local machine.

*`git clone https://github.com/juliecmitchell/beGAN.git`


To install required dependencies

*`conda install seaborn scikit-learn ipywidgets`


To run the code using the command line python

*`cd beGAN`
*`python beGAN_Pauling33k_run.py`

This code will generate 16-mer beta-hairpin peptide sequences with corresponding GP scores.

To run the code interactively using jupyter-lab

*`conda install -c conda-forge jupyterlab`



*`jupyter-lab`

Run `beGAN_Pauling33k_run.ipynb` interactively.

Features for the beGAN model were collected using AAindex matrix:
[https://www.genome.jp/aaindex/]
* Kawashima, S., Pokarowski, P., Pokarowska, M., Kolinski, A., Katayama, T., and Kanehisa, M.; AAindex: amino acid index database, progress report 2008. Nucleic Acids Res. 36, D202-D205 (2008)

Amino Acid indices can be extracted using the Propy3 package during model training.

`pip install propy3`

[https://github.com/MartinThoma/propy3]


