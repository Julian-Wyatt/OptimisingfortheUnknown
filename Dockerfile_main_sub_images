ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

# Pull the Docker image | 拉取镜像
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Set environment variables and compilation options | 设置环境变量和编译选项
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error | 避免公钥错误
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional) Use a mirror to speed up downloads | 可选方案，利用镜像加快下载速度
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//https:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages | 安装所需的软件包
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a group and user for the algorithm | 创建用户组和用户
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN cat /etc/group

# Create directories and set ownership | 创建目录并设置所有权
RUN mkdir -p /opt/algorithm /input /output
RUN chown algorithm:algorithm /opt/algorithm /input /output

# Switch to the created user and set working directory | 切换用户并设置工作目录
USER algorithm
WORKDIR /opt/algorithm

# Update PATH to include user-specific binary directory | 更新PATH变量以包含用户特定的二进制目录
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# Install or upgrade pip and pip-tools in user space | 在用户空间中安装或升级pip和pip-tools
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

# Copy the requirements file and install dependencies | 复制requirements文件并安装依赖
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

# Copy the Python scripts and model files | 复制Python脚本和模型文件
COPY --chown=algorithm:algorithm core/process.py /opt/algorithm/

COPY --chown=algorithm:algorithm model.py /opt/algorithm/
COPY --chown=algorithm:algorithm best_model.pt /opt/algorithm/

# Generate RCNN images
RUN python -m process

# Set the entry point to run the process script | 设置入口点以运行process脚本
ENTRYPOINT python -m process $0 $@
