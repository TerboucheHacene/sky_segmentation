# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38-cu113-ubuntu20.04-sagemaker


ENV PATH="/opt/ml/code:${PATH}"

# install poetry
RUN pip install poetry

# configure poetry to disable virtualenvs
RUN poetry config virtualenvs.create false --local

# copy the poetry.lock and pyproject.toml files to the working directory
COPY pyproject.toml poetry.lock /opt/ml/code/


# set the working directory to the code directory
WORKDIR /opt/ml/code

# install dependencies
RUN poetry install --no-root

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code
COPY sky_segmentation /opt/ml/code

# install the sky_segmentation package
RUN poetry install --only-root


# this environment variable is used by the SageMaker PyTorch container to determine our user code directory
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# This environment variable is used by the SageMaker PyTorch container to determine our program entry point for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container
COPY scripts/train_sagemaker.py /opt/ml/code/train_sagemaker.py

ENV SAGEMAKER_PROGRAM train_sagemaker.py
