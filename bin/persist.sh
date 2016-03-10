#!/bin/bash

if [[ "$#" -ne 1 ]]; then
    echo "Usage:  persist.sh <alluxio_file>"
    exit 1
fi


START=$(date +%s)
alluxio fs persist $1
END=$(date +%s)
echo 'Time to UFS: ' $(($END-$START))
