#!/bin/bash

dir="/home/sphinx/git/master-project/Tensorflow"

for trainfile in ./build/slope_train_set*.bin; do		
	
	IFS='_' read -ra tmp <<< "$trainfile"

	queryfile="./build/slope_query_set_${tmp[3]}"
	labelfile="./build/slope_query_lab_${tmp[3]}"
	echo "python lstm.py ${trainfile}  ${queryfile} ${labelfile} ${1}"
	python lstm.py ${trainfile}  ${queryfile} ${labelfile} ${1}
done

for trainfile in ./build/sine_train_set*.bin; do		
	
	IFS='_' read -ra tmp <<< "$trainfile"

	queryfile="./build/sine_query_set_${tmp[3]}"
	labelfile="./build/sine_query_lab_${tmp[3]}"
	echo "python lstm.py ${trainfile}  ${queryfile} ${labelfile} ${1}"
	python lstm.py ${trainfile}  ${queryfile} ${labelfile} ${1}
done

for trainfile in ./build/pulse_train_set*.bin; do		
	
	IFS='_' read -ra tmp <<< "$trainfile"
	
	queryfile="./build/pulse_query_set_${tmp[3]}"
	labelfile="./build/pulse_query_lab_${tmp[3]}"
	echo "python lstm.py ${trainfile}  ${queryfile} ${labelfile} ${1}"
	python lstm.py ${trainfile}  ${queryfile} ${labelfile} ${1}
done
