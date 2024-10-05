#!/usr/bin/env bash
rm datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Validation\ Set/predictions_docker.csv
rm -rf datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Validation\ Set/images/processed_images

# Get the absolute path to the script 
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Build the Docker image
./build.sh $1

# Generate a random string to use as a volume label suffix
# VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
VOLUME_SUFFIX=$(openssl rand -hex 12)

# Set the memory limit for the algorithm image to 8GB 
# The current memory limit on Grand Challenge is 30GB, but this can be configured in the algorithm settings
MEM_LIMIT="8g"

# Create a Docker volume with a unique name 
# The volume name is cldetection_alg_2024-output-$VOLUME_SUFFIX, where $VOLUME_SUFFIX is the random string generated earlier
docker volume create cldetection_alg_2024-output-$VOLUME_SUFFIX

# Run the Docker container with specified restrictions 
# --network="none": Disable networking in the container
# --cap-drop="ALL": Drop all Linux capabilities in the container
# --security-opt="no-new-privileges": Prevent the container from gaining new privileges
# --shm-size="128m": Set the shared memory size to 128MB
# --pids-limit="256": Limit the number of processes in the container to 256
# --name="test_container": Name the container "test_container"
# -v /data/XHY/CL-Detection2024/dataset/Validation\ Set/images/:/input/images/lateral-dental-x-rays/:
#     Mount the local directory for validation images to the container's /input/images/lateral-dental-x-rays/ directory
# -v cldetection_alg_2024-output-$VOLUME_SUFFIX:/output/:
#     Mount the Docker volume to the container's /output/ directory
# cldetection_alg_2024: The name of the Docker image to run. If the ./build.sh script changes the image name, update this accordingly
#--gpus all \
docker run \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        --name="test_container" \
        -v /Users/jules/Documents/landmark_detection/datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Validation\ Set/images:/input/images/lateral-dental-x-rays/ \
        -v cldetection_alg_2024-output-$VOLUME_SUFFIX:/output/ \
        cldetection_alg_2024

# Copy the prediction file from the container to the host
docker cp test_container:/output/predictions.csv /Users/jules/Documents/landmark_detection/datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Validation\ Set/predictions_docker.csv

# Remove the container 
docker rm test_container

# Remove the Docker volume 
docker volume rm cldetection_alg_2024-output-$VOLUME_SUFFIX
