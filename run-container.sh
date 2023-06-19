#!/bin/bash

docker run \
    --mount type=bind,source="$(pwd)",target=/world-models \
    --gpus all \
    -it \
    --privileged \
    --name world-models \
    -w /world-models \
    --rm \
    -u $UID:$GID \
    -v="/etc/group:/etc/group:ro" \
    -v="/etc/passwd:/etc/passwd:ro" \
    -v="/etc/shadow:/etc/shadow:ro" \
    -v="/etc/sudoers:/etc/sudoers:ro" \
    --shm-size=16G \
    --network=host \
    world-models:latest \
    bash
