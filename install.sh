kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p kaggle_data
conda create -n transformers python pandas tqdm jupyter
conda activate transformers
conda install pytorch cudatoolkit=10.0 -c pytorch
conda install pytorch cpuonly -c pytorch
conda install -c anaconda scipy
conda install -c anaconda scikit-learn
pip install pytorch-transformers
pip install tensorboardX
pip install --upgrade nbdime
conda install -c conda-forge ipywidgets
conda install -c conda-forge jupyterlab
#install apex from https://www.github.com/nvidia/apex to use fp16 training.

