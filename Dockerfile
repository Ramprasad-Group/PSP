# Use the official Ubuntu image as a parent image
FROM ubuntu:latest

# Avoid interactive dialog during package installations
ARG DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update -y && \
    apt-get install -y vim wget git

# Set environment variables
ENV CONDA_HOME=/opt/conda
ENV PATH=$CONDA_HOME/bin:$PATH

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_HOME && \
    rm miniconda.sh

# Set the working directory
WORKDIR /opt

# Create a Conda environment and install dependencies
RUN /opt/conda/bin/conda create -n myenv -y python=3.8
RUN /opt/conda/bin/conda init bash
RUN echo "conda activate myenv" >> ~/.bashrc
ENV PATH=$CONDA_HOME/envs/myenv/bin:$PATH

# Install additional Python packages directly
RUN conda install -n myenv -y -c anaconda scipy=1.7 pandas=1.5 'numpy<1.23.0'
RUN /opt/conda/envs/myenv/bin/pip install rdkit
RUN conda install -n myenv -y -c conda-forge openbabel
RUN conda install -n myenv -y anaconda::networkx anaconda::tqdm anaconda::tabulate

## Install packmol
# Clone Packmol repository
RUN apt-get install -y build-essential gfortran
RUN git clone https://github.com/m3g/packmol.git /opt/packmol
WORKDIR /opt/packmol
RUN make

# Set the PACKMOL_EXEC environment variable
ENV PACKMOL_EXEC=/opt/packmol/packmol

## Install pysimm
WORKDIR /opt
RUN git clone -b 1.1 --single-branch https://github.com/polysimtools/pysimm 
# Set up PYTHONPATH
ENV PYTHONPATH=$PYTHONPATH:/opt/pysimm
# Set up PATH
ENV PATH=$PATH:/opt/pysimm/bin

## Install ambertools
RUN conda install -n myenv -y -c conda-forge ambertools
ENV ANTECHAMBER_EXEC=/opt/conda/envs/myenv/bin/antechamber

## Install PSP
RUN git clone https://github.com/Ramprasad-Group/PSP.git
WORKDIR /opt/PSP
RUN /opt/conda/envs/myenv/bin/python setup.py install

# Set up default Python to /opt/conda/envs/myenv/bin/python
RUN echo 'export PATH=/opt/conda/envs/myenv/bin:$PATH' >> /etc/profile.d/python.sh && \
    echo 'alias python=/opt/conda/envs/myenv/bin/python' >> /etc/profile.d/python.sh

# Set HOME as working directory
WORKDIR /root

# Copy test files to /root
RUN cp -r /opt/PSP/test/ /root/

# Set the default command to run your application
CMD ["bash"]
