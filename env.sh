apt update
apt install bc
pip install ./whls/torch-2.0.0+cpu-cp38-cp38-linux_x86_64.whl
pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
conda install tokenizers -y
pip install ./ckpts/whls/unicore-0.0.1+cu117torch2.0.0-cp38-cp38-linux_x86_64.whl
pip install opt_einsum -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install depthcharge -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install spectrum_utils -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib_venn -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pandas -y
conda install tensorboard -y
conda install scikit-learn -y