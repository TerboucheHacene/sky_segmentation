#!/usr/bin/env bash

# Build docker image, push to ECR for use with SageMaker

# exit when any command fails
set -e

# the only input argument is the version tag; generally we use the commit hash
commithash=$1
image="highlightsv2-train"

if [ "$commithash" == "" ]
then
    echo "Usage: $0 <commithash>"
    echo "Need to provide this commit hash as it is written to the Docker image,"
    echo " so we know what version of the code the Docker image was built with!"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)
echo "account $account"

if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration (default to us-east-1 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}
echo "region $region"


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $account.dkr.ecr.$region.amazonaws.com

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$region.amazonaws.com

# Get CodeArtifact Authorization Token to pass to docker build
codeartifact_auth_token=$(aws codeartifact get-authorization-token --query authorizationToken --region $region --domain prod-package-management-domain --domain-owner 570340546327 --duration-seconds 1800 --output text)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build --build-arg HLV2TRAIN_COMMIT_HASH=${commithash} --build-arg CODEARTIFACT_AUTH_TOKEN=${codeartifact_auth_token} --build-arg REGION=${region} -t ${image} .
docker tag ${image} ${fullname}

docker push ${fullname}

echo "BUILD AND PUSH COMPLETE!"
