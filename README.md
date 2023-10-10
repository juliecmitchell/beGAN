beGAN is a code for generating beta-hairpin sequences of variable residue lengths like 14-mer, 16-mer, 18-mer, and 20-mer peptides. 
It is provided as a Jupyter Notebook and as a Python file. 
To run either, there are a number of dependencies, including pytorch, numpy, pandas, propy, etc.

To create the beGAN environment and activate it using conda

* `conda create --name beGAN`


* `conda activate beGAN`

To obtain the beGAN code repository in your local machine.

* `git clone https://github.com/juliecmitchell/beGAN.git`


To install required dependencies

* `conda install seaborn scikit-learn ipywidgets`


To run the code using the command line python

* `cd beGAN`
* `python beGAN_Pauling33k_run.py`

This code will generate 16-mer beta-hairpin peptide sequences with corresponding GP scores.

To run the code interactively using jupyter-lab

* `conda install -c conda-forge jupyterlab`



* `jupyter-lab`

Run `beGAN_Pauling33k_run.ipynb` interactively.

Features for the beGAN model were collected using AAindex matrix:
[https://www.genome.jp/aaindex/]
* Kawashima, S., Pokarowski, P., Pokarowska, M., Kolinski, A., Katayama, T., and Kanehisa, M.; AAindex: amino acid index database, progress report 2008. Nucleic Acids Res. 36, D202-D205 (2008)

Amino Acid indices can be extracted using the Propy3 package during model training.

`pip install propy3`

[https://github.com/MartinThoma/propy3]

Validations:

3D structures of the beta-hairpin peptide sequences can be further validated using AlphaFold2 and ESMFold. 
* [https://github.com/google-deepmind/alphafold] | [https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb]

Jumper, J., R. Evans, A. Pritzel, T. Green, M. Figurnov, O. Ronneberger, K. Tunyasuvunakool, R. Bates, A. Žídek, A. Potapenko, A. Bridgland, C. Meyer, S.A.A. Kohl, A.J. Ballard, A. Cowie, B. Romera-Paredes, S. Nikolov, R. Jain, J. Adler, T. Back, S. Petersen, D. Reiman, E. Clancy, M. Zielinski, M. Steinegger, M. Pacholska, T. Berghammer, S. Bodenstein, D. Silver, O. Vinyals, A.W. Senior, K. Kavukcuoglu, P. Kohli, and D. Hassabis. 2021. Highly accurate protein structure prediction with AlphaFold. Nature. 596:583–589.

* [https://github.com/facebookresearch/esm] | [https://esmatlas.com/resources?action=fold]

Hie, B., S. Candido, Z. Lin, O. Kabeli, R. Rao, N. Smetanin, T. Sercu, and A. Rives. 2022. A high-level programming language for generative protein design. Synthetic Biology

ML-predicted solubility can be tested using Peptide-bio:

* [https://peptide.bio]
  
Ansari, M., and A.D. White. 2023. Serverless Prediction of Peptide Properties with Recurrent Neural Networks. J. Chem. Inf. Model. 63:2546–2553.
