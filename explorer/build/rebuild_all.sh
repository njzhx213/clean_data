#!/bin/bash
# Rebuild every explorer extract from H200_profiling_portal/clean_data (tag -> tl -> wall -> opp). Idempotent.
set -u; cd "$(dirname "$0")"; L=../data/build.log
echo "[$(date '+%m-%d %H:%M')] tag" >> $L;  python3 build_tagdata.py  >> $L 2>&1 || echo "TAG FAILED" >> $L
echo "[$(date '+%m-%d %H:%M')] tl"  >> $L;  python3 build_tldata.py   >> $L 2>&1 || echo "TL FAILED" >> $L
echo "[$(date '+%m-%d %H:%M')] wall" >> $L; python3 build_walldata.py >> $L 2>&1 || echo "WALL FAILED" >> $L
echo "[$(date '+%m-%d %H:%M')] opp" >> $L;  python3 build_oppdata.py  >> $L 2>&1 || echo "OPP FAILED" >> $L
echo "[$(date '+%m-%d %H:%M')] DONE" >> $L
