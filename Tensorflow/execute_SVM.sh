#!/bin/bash

dir="/home/sphinx/git/master-project/Tensorflow"

for trainfile in ./build/slope_train_set*.bin; do		
	
	IFS='_' read -ra tmp <<< "$trainfile"

	trainlabel="./build/slope_train_lab_${tmp[3]}"
	queryfile="./build/slope_query_set_${tmp[3]}"
	labelfile="./build/slope_query_lab_${tmp[3]}"
	echo "python svm.py ${trainfile} ${trainlabel} ${queryfile} ${labelfile} ${1}"
	python svm.py ${trainfile} ${trainlabel} ${queryfile} ${labelfile} ${1}
done

for trainfile in ./build/sine_train_set*.bin; do		
	
	IFS='_' read -ra tmp <<< "$trainfile"

	trainlabel="./build/sine_train_lab_${tmp[3]}"
	queryfile="./build/sine_query_set_${tmp[3]}"
	labelfile="./build/sine_query_lab_${tmp[3]}"
	echo "python svm.py ${trainfile} ${trainlabel} ${queryfile} ${labelfile} ${1}"
	python svm.py ${trainfile} ${trainlabel} ${queryfile} ${labelfile} ${1}
done

for trainfile in ./build/pulse_train_set*.bin; do		
	
	IFS='_' read -ra tmp <<< "$trainfile"
	
	trainlabel="./build/pulse_train_lab_${tmp[3]}"
	queryfile="./build/pulse_query_set_${tmp[3]}"
	labelfile="./build/pulse_query_lab_${tmp[3]}"
	echo "python svm.py ${trainfile} ${trainlabel} ${queryfile} ${labelfile} ${1}"
	python svm.py ${trainfile} ${trainlabel} ${queryfile} ${labelfile} ${1}
done
