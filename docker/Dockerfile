
ARG CUDA="9.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update --fix-missing -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev 

RUN apt-get install --reinstall python3-apt -y

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN conda install cython
RUN pip install requests ninja yacs cython matplotlib opencv-python tqdm numpy



# Install PyTorch 1.0 Nightly
ARG CUDA
RUN conda install pytorch-nightly cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya

# Install TorchVision 
RUN conda install torchvision


# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && git checkout 55796a04ea770029a80cf5933cc5c3f3f6fa59cf\
 && python setup.py build develop



# install python-pcl binding and it's dependencies
RUN apt update
RUN apt-get update --fix-missing

RUN apt-get install -y pkg-config freeglut3-dev build-essential  qt-sdk linux-libc-dev libudev-dev  mpi-default-dev openmpi-bin openmpi-common libpcap-dev

RUN apt-get install -y wget
RUN wget http://launchpadlibrarian.net/209530212/libeigen3-dev_3.2.5-4_all.deb
RUN dpkg -i libeigen3-dev_3.2.5-4_all.deb

RUN apt-get install -y libflann-dev libboost1.58-all-dev libusb-1.0-0-dev libusb-1.0
RUN apt-get install -y libqhull* libqhull-dev #libxmu-dev libxi-dev 


RUN apt-get update -y
RUN apt-get install libpcl-dev -y
RUN git clone https://github.com/strawlab/python-pcl \
    && cd python-pcl \
    && python setup.py build_ext -i \
    && python setup.py install


#install argoverse-api and download map
RUN git clone https://github.com/argoai/argoverse-api \
    && cd argoverse-api \
    && pip install -e .\
    && cd ..

RUN wget https://s3.amazonaws.com/argoai-argoverse/hd_maps.tar.gz\
    && tar -xf hd_maps.tar.gz\
    && mv map_files argoverse-api

RUN pip install sklearn pyquaternion uuid

#For visualization
RUN apt-get install -y libglib2.0-0 libxext6 libsm6 libxrender1 gtk3.0
RUN conda install -y -c conda-forge jsoncpp=0.10.6 qt pyvtk pyside envisage mesalib mayavi\
    && conda install -y -c ajsrk traitsui=6.0.0 \
    && conda install -c viscid-hub vtk

RUN rm libeigen3-dev_3.2.5-4_all.deb

#For kitti dataset support(deprecated)
#RUN cd pykitti
#RUN python setup.py install
#RUN cd ..
