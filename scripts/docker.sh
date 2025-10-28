#!/bin/bash

# Build the Docker image
docker build -t f1tenth-rl:latest .

# Run the Docker container
docker run -it \
    --name f110-rl \
    --gpus all \
    --network=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD:/home/F1Tenth-RL \
    f1tenth-rl:latest

# Note: --rm flag will automatically clean up the container after exit