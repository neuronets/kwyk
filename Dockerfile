FROM ubuntu:16.04
ARG DEBIAN_FRONTEND="noninteractive"
ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgomp1 \
        python3 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://bootstrap.pypa.io/pip/3.5/get-pip.py | python3 - \
    && ln -s $(which python3) /usr/local/bin/python

WORKDIR /opt/kwyk
COPY [".", "."]
RUN pip install --no-cache-dir --editable .[cpu] \
    # Install the saved models.
    && curl -fsSL https://github.com/patrick-mcclure/nobrainer/tarball/master \
    | tar xz --strip=1 --wildcards '*/saved_models'

ENV FREESURFER_HOME="/opt/kwyk/freesurfer"
ENV PATH="$FREESURFER_HOME/bin:$PATH"

WORKDIR /data
ENTRYPOINT ["kwyk"]
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@gmail.com>"
