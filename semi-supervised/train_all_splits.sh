#!/bin/bash
for i in {0..3}
do
	echo "Split $i starting"
	python train.py "$i"
	echo "Split $i ended"
done