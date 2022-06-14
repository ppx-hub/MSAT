#!/bin/bash

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 0 --p 1 &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 0 --p 0.993955 &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 0 --p 0.993955 --channelnorm True &
PID1=$!
wait ${PID1}
python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 0 --p 0.993955 --gamma 5 & 
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 0 --p 0.993955 --gamma 5 --lateral_inhi True &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 0 --p 0.993955 --gamma 5 --spicalib True --spi 64 &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 0 --p 0.993955 --gamma 5 --lateral_inhi True --spicalib True --spi 32 & 
PID1=$!;
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 0 --p 0.993955 --gamma 5 --lateral_inhi True --spicalib True --spi 64 & 
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 0 --p 0.993955 --gamma 5 --lateral_inhi True --spicalib True --spi 96 & 
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 0 --p 0.993955 --gamma 5 --lateral_inhi True --spicalib True --spi 128 &  
PID1=$!;
wait ${PID1}