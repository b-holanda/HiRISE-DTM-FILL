FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311

COPY ./environment.yaml /tmp/environment.yaml

RUN conda env update -n base -f /tmp/environment.yaml && \
    conda clean -afy
