#!/bin/bash

#conda activate equiformer2
for i in {1..10};do
  echo $i
  timeout 3m bash ./scripts/SuperCon/train_${i}.sh
done

