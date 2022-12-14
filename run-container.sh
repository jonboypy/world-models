#!/bin/bash

docker run \
    --mount type=bind,source="$(pwd)",target=/world-models \
    --gpus all \
    -it \
    -u $(id -u):$(id -g) \
    --name world-models \
    -w /world-models \
    --rm \
    world-models:latest \
    bash