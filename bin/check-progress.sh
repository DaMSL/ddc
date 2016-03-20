#!/bin/bash

REDIS_HOST=$(cut -d, -f1 ${1}_RedisService.lock)
TOTAL_COMPLETE_SIM=$(ls -1 -R $HOME/work/jc/${1}/*/*.dcd | wc -l)
RECENT_SIM=$(redis-cli -h ${REDIS_HOST} llen completesim)
QUEUED_JOBS=$(redis-cli -h ${REDIS_HOST} llen jcqueue)
ME=$(whoami)
RUNNING_SIM=$(squeue -u ${ME} | grep sw- | wc -l)
NUM_CTL_ITR=$(ls -1 $HOME/work/log/${1}/cw* | wc -l)

echo
echo "REDIS_HOST     = ${REDIS_HOST}"
echo "TOTAL Sim Exec = ${TOTAL_COMPLETE_SIM}"
echo "Completed Sim  = ${RECENT_SIM}"
echo "Running Sim    = ${RUNNING_SIM}"
echo "Jobs in QUEUE  = ${QUEUED_JOBS}"
echo "Total Calc CTL = ${NUM_CTL_ITR}" 
echo
