#!/bin/bash

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 2 --p 1 --model_name resnet20 &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 2 --p 0.987506 --model_name resnet20 &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 2 --p 0.987506 --channelnorm True --model_name resnet20 &
PID1=$!
wait ${PID1}
python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 2 --p 0.987506 --gamma 5 --model_name resnet20 & 
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 2 --p 0.987506 --gamma 5 --lateral_inhi True --model_name resnet20 &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 2 --p 0.987506 --gamma 5 --spicalib True --spi 64 --model_name resnet20 &
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 2 --p 0.987506 --gamma 5 --lateral_inhi True --spicalib True --spi 32 --model_name resnet20 & 
PID1=$!;
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 2 --p 0.987506 --gamma 5 --lateral_inhi True --spicalib True --spi 64 --model_name resnet20 & 
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 2 --p 0.987506 --gamma 5 --lateral_inhi True --spicalib True --spi 96 --model_name resnet20 & 
PID1=$!
wait ${PID1}

python /data1/hexiang/newframework/conversion3/CIFAR100/debug_converted_CIFAR100_KL.py --device 2 --p 0.987506 --gamma 5 --lateral_inhi True --spicalib True --spi 128 --model_name resnet20 &  
PID1=$!;
wait ${PID1}