#!/bin/bash

rds_address="research-data-int.sydney.edu.au:/rds/PRJ-STELLAR/Program4/chris_phd/Clennett_data/"
model_url="https://github.sydney.edu.au/EarthByte/EarthBytePlateMotionModel-ARCHIVE.git"

function main {
    sync_rds || return
    sync_model || return

    return 0
}

function sync_rds {
    read -rp "Enter UniKey: " unikey
    echo "Synchronising ${unikey}@${rds_address}..."
    rsync -avz --no-perms --no-times \
        "${unikey}@${rds_address}" \
        "Clennett_data"
    return
}

function sync_model {
    if [ ! -d "EarthBytePlateMotionModel-ARCHIVE" ]; then
        echo "Downloading plate model repository..."
        git clone "${model_url}" || return
    fi
    cd "EarthBytePlateMotionModel-ARCHIVE" || return

    echo "Updating plate model repository..."
    git fetch --all --prune && git checkout "Clennett-updates" && git pull
    cd .. || return
}

main
