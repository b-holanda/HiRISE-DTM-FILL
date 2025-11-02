FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1

COPY environment.yml /tmp/environment.yml

RUN conda env update -n base -f /tmp/environment.yml && \
    conda clean -afy
