#!/usr/bin/env bash
# random select 400 image from ILSVRC12/train
mkdir -p data/ILSVRC12
f=/share/Dataset/ILSVRC12/train
for ((i=0;i<${1:-10};++i));do
    a=$(ls $f|shuf -n1)
    b=$(ls $f/$a|shuf -n1)
    if [ $(du -b $f/$a/$b|cut -f 1) -lt 100000 ] ;then
        ((--i))
        continue
    fi 
    printf $a/$b'\n'
    cp $f/$a/$b data/ILSVRC12
done
