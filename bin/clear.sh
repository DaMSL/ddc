#!/bin/bash

PART=debug
STARTNUM=0
CACHE_DB=$HOME/work/${1}_cache.rdb
CACHE_LOG=$HOME/work/${1}_cache.log
#DB_LOG=$HOME/work/${1}.log 

if [[ "$#" -ne 1 ]]; then
    echo "Usage:  clear.sh <epoch>"
    exit 1
fi

rm -rf $HOME/work/jc/$1
rm -rf $HOME/work/log/$1

FILE_LIST=($CACHE_DB $CACHE_LOG ${1}.log ${1}.conf ${1}.lock)

for i in "${FILE_LIST[@]}"
do
  echo "Checking ${i}"
  if [[ -e "${i}" ]]; then
    echo "  Clearing ${i}"
    rm ${i}
  fi
done
