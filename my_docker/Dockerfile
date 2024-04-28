FROM pytorch/pytorch

RUN apt-get update
RUN apt-get install -y gcc

# Use a 'large' base container to show-case how to load tensorflow or pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
#Copy all the ressources need (e.g the wight of you neural network)
COPY --chown=user:user resources /opt/app/resources

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

#Copy your python scripts
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user mynetwork.py /opt/app/
ENTRYPOINT ["python", "inference.py"]



