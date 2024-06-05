
FROM ubuntu:22.04

#RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
# Install packages from apt
RUN apt-get update -y && apt-get install -y sudo && apt-get install -y python3 python3-pip cmake git build-essential 

RUN pip install --upgrade pip
RUN pip install numpy scipy matplotlib \
    tqdm PyYAML \ 
    setuptools cmake ninja scikit-build \
    casadi

# Install pytorch 
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Create the necessary directories
RUN mkdir -p /external /devel

# Install acados

#COPY ./acados /app/external/acados



WORKDIR /app/external
RUN git clone https://github.com/acados/acados.git \
    && cd acados \
    && git submodule update --recursive --init \
    && mkdir -p build \
    && cd build \
    && cmake -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON -DACADOS_PYTHON=ON ..  \
    && make install -j4

# RUN rm -rf acados/build
# RUN mkdir -p acados/build

# RUN cd acados/build && cmake -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON ..

# RUN cd acados/build && make install -j4

RUN pip install -e acados/interfaces/acados_template

# Install l4casadi
RUN pip install --upgrade setuptools
RUN pip install l4casadi --no-build-isolation



# Link to shared libraries (acados)
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/external/acados/lib" >> /root/.bashrc
RUN echo "export ACADOS_SOURCE_DIR=/app/external/acados" >> /root/.bashrc

# Link to python packages
RUN echo "export PYTHONPATH=$PYTHONPATH:/app/devel/safe-mpc/src" >> /root/.bashrc

# USER $USERNAME
# WORKDIR /home/$USERNAME/

# # Copy the script into the container
# COPY init_container.sh .
# # Make the script executable
# RUN chmod +x init_container.sh
WORKDIR /app

RUN apt install -y git
# Execute the script when the container starts
CMD ["tail", "-f", "/dev/null"]



# build with  docker build --progress=plain -t mpc-dock .
# start it with docker run  -it --name mycontainername --mount type=bind,source="$(pwd)",target=/app/devel/safe-mpc mpc-dock bash,
# sicne pwd used the command has to be lauched inside the safe-mpc folder
# to open other shells use docker exec -it mycontainername bash



