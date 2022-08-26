#!/bin/bash

model_url="https://github.sydney.edu.au/EarthByte/EarthBytePlateMotionModel-ARCHIVE.git"

function sync_model {
    if [ ! -d "EarthBytePlateMotionModel-ARCHIVE" ]; then
        echo "Downloading plate model repository..."
        git clone "${model_url}" || return
    fi
    cd "EarthBytePlateMotionModel-ARCHIVE" || return

    echo "Updating plate model repository..."
    git fetch --all --prune && git checkout master && git pull
    cd .. || return
}

sync_model
