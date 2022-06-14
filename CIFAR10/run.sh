#!/bin/bash

python /data1/hexiang/newframework/conversion3/CIFAR10/debug_converted_CIFAR10_KL.py --device 0 --p 1 &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR10/debug_converted_CIFAR10_KL.py --device 0 --p 0.995849 &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR10/debug_converted_CIFAR10_KL.py --device 0 --p 0.995849 --channelnorm True &
PID1=$!
wait ${PID1}
python /data1/hexiang/newframework/conversion3/CIFAR10/debug_converted_CIFAR10_KL.py --device 0 --p 0.995849 --gamma 5 & 
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR10/debug_converted_CIFAR10_KL.py --device 0 --p 0.995849 --gamma 5 --lateral_inhi True &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR10/debug_converted_CIFAR10_KL.py --device 0 --p 0.995849 --gamma 5 --spicalib True --spi 64 &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR10/debug_converted_CIFAR10_KL.py --device 0 --p 0.995849 --gamma 5 --lateral_inhi True --spicalib True --spi 32 & 
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR10/debug_converted_CIFAR10_KL.py --device 0 --p 0.995849 --gamma 5 --lateral_inhi True --spicalib True --spi 64 & 
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR10/debug_converted_CIFAR10_KL.py --device 0 --p 0.995849 --gamma 5 --lateral_inhi True --spicalib True --spi 96 & 
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR10/debug_converted_CIFAR10_KL.py --device 0 --p 0.995849 --gamma 5 --lateral_inhi True --spicalib True --spi 128 &  
PID1=$!
wait ${PID1}