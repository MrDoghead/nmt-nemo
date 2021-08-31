#!/bin/bash

nvidia-docker run --shm-size=3g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host -v $PWD:/workspace/nmt-nemo nmt-nemo bash
