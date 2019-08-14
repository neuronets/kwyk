FROM debian:stretch-slim
ARG DEBIAN_FRONTEND="noninteractive"
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"

RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
        libgomp1 \
        python3 \
        # python3-h5py \
        # python3-numpy \
        python3-pip \
        # python3-scipy \
        python3-setuptools \
        python3-wheel \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s $(which python3) /usr/local/bin/python

COPY [".", "/opt/nobrainer"]
RUN pip3 install --no-cache-dir --editable /opt/nobrainer[cpu]

ENV FREESURFER_HOME="/opt/nobrainer/third_party/freesurfer"
ENV PATH="$FREESURFER_HOME/bin:$PATH"

WORKDIR /data
ENTRYPOINT ["nobrainer_bwn"]
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@gmail.com>"
