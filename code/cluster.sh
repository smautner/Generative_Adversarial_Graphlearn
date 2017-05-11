#!/bin/sh

#$ -cwd
#$ -l h_vmem=7G
#$ -pe smp 8
#$ -R y
source ~/.bashrc
source ~/stupidbash.sh
python showroc.py

