# CUDA + CuDNN 対応の NVIDIA イメージ
#FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
#FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
#FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
#FROM nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04
FROM tensorflow/tensorflow:2.14.0-gpu


# 作業ディレクトリ
WORKDIR /app

# Python & pip をインストール（Ubuntuベースなので自分で入れる）
RUN apt update && apt install -y \
    libgl1-mesa-glx \
    build-essential \
    libsndfile1 \
    libatlas-base-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libglib2.0-0 \
    libgtk2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \    
    curl \
    git



RUN pip install --upgrade pip setuptools wheel



# 必要なPythonパッケージをインストール
RUN pip install \
    torch torchvision torchinfo torchviz \
    scikit-learn pandas numpy==1.26.4 matplotlib opencv-python \
    optuna seaborn IPython tqdm \
    transformers timm \
    json5 \
    black ruff mypy \
    japanize_matplotlib==1.1.3


# 実行コマンド（必要に応じて変更）
CMD ["bash"]


