#!/bin/bash

PART=debug
STARTNUM=0

if [[ "$#" -ne 1 ]]; then
    echo "Usage:  clear.sh <epoch>"
    exit 1
fi

rm -rf $HOME/work/jc/$1
rm -rf $HOME/work/log/$1
rm ${1}*.log
rm ${1}*.conf
rm ${1}.lock
