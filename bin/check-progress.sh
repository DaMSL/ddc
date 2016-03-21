#!/bin/bash

ME=$(whoami)

echo
REDIS_HOST=$(cut -d, -f1 ${1}_RedisService.lock)
echo "REDIS_HOST     = ${REDIS_HOST}"
TOTAL_COMPLETE_SIM=$(ls -1 -R $HOME/work/jc/${1}/*/*.dcd | wc -l)
echo "TOTAL Sim Exec = ${TOTAL_COMPLETE_SIM}"
RECENT_SIM=$(redis-cli -h ${REDIS_HOST} llen completesim)
echo "Completed Sim  = ${RECENT_SIM}"
RUNNING_SIM=$(squeue -u ${ME} | grep sw- | wc -l)
echo "Running Sim    = ${RUNNING_SIM}"
QUEUED_JOBS=$(redis-cli -h ${REDIS_HOST} llen jcqueue)
echo "Jobs in QUEUE  = ${QUEUED_JOBS}"
NUM_CTL_ITR=$(ls -1 $HOME/work/log/${1}/cw* | wc -l)
echo "Total Calc CTL = ${NUM_CTL_ITR}" 
echo
