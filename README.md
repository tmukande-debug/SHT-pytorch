# SHT-pytorch

This is a Pytorch Version for the paper on KDD 2022.
<Lianghao Xia, Chao Huang, Chuxu Zhang (2022). Self-Supervised Hypergraph Transformer for Recommender Systems.>

python=3.6.12
pytorch= 1.6.0
numpy=1.16.0
scipy=1.5.2

Notes:

1. Please run the <SHT.py> for training.
2. Please modify the data path in DataHandler.py for training.
3. Please create a Folder named 'Model' under your current path for saving the results.
4. You can use Google Colab to quickly review and run the code by uploading the <SHT_notebook.ipynb>. 
5. The result is a little lower than the TF version (-0.002~-0.003). The reason might be the CUDA in pytorch is float.32. Please use .cuda() to set the Parameters.
6. I modified the Learning rate from 1e-3 to 1.5e-3.
7. There is a huge difference on performance if the parameters are not set with '.cuda()'.
7. Happy coding.
