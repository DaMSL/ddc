#!/bin/bash

if [[ $# == 0 ]]; then
  echo 'Usage setenv.sh <appl_name>'
  exit 1
fi

NAME=$1

if [[ ! -e "${NAME}_AlluxioService.lock" ]]; then
  echo "Alluxio not running (lock file not found)"
else
  ALLUX_MASTER=$(cut -d, -f1 ${NAME}_AlluxioService.lock)
  echo "Setting Alluxio alias and master location as ${ALLUX_MASTER}"
  export ALLUXIO_MASTER_ADDRESS=${ALLUX_MASTER}
  alias afs='alluxio fs'
fi

if [[ ! -e "${1}_RedisService.lock" ]]; then
  echo "Redis not running (lock file not found)"
else
  echo "Setting Redis alias"
  module load redis
  export REDIS_MASTER=$(cut -d, -f1 ${NAME}_RedisService.lock)
  alias redis='redis-cli -h ${REDIS_MASTER}'
fi

