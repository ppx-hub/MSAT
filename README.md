- For BeyesianOptimization

please use `scipy==1.7.3`, or you will meet `Queue is empty error` 

> https://github.com/fmfn/BayesianOptimization/issues/270




# Spiking Calibration

1. 关于周期均值的计算--滑动计算与统计所有
2. 关于针对性地设置余量--使用贝叶斯优化近似一个最优？
3. 关于使用校正后依然存在SIN的问题：猜测是剩余小的负值造成的


输出KL散度
all-2-256：   4.082165    0.7867879746835443
all-3-256：   8.355107    0.786689082278481
-spicalib:   57.317192    0.7862935126582279
-LIPooling: 345.866302    0.7800632911392406
-gamma:    1085.394409    0.770371835443038
-p-norm:   4912.201172    0.7286392405063291



## trial-1
VGG16,200,seed16
use: 
Conv_2: sin_radio: 0.0000000
Conv_5: sin_radio: 0.0000182
Conv_9: sin_radio: 0.0000592
Conv_12: sin_radio: 0.0001347
Conv_16: sin_radio: 0.0000680
Conv_19: sin_radio: 0.0001490
Conv_22: sin_radio: 0.0001037
Conv_26: sin_radio: 0.0001314
Conv_29: sin_radio: 0.0001990
Conv_32: sin_radio: 0.0003317
Conv_36: sin_radio: 0.0005908
Conv_39: sin_radio: 0.0005029
Conv_42: sin_radio: 0.0018800

no:
Conv_2: sin_radio: 0.0000000
Conv_5: sin_radio: 0.0011344
Conv_9: sin_radio: 0.0132627
Conv_12: sin_radio: 0.0076944
Conv_16: sin_radio: 0.0048643
Conv_19: sin_radio: 0.0291539
Conv_22: sin_radio: 0.0167017
Conv_26: sin_radio: 0.0195789
Conv_29: sin_radio: 0.0187041
Conv_32: sin_radio: 0.0258775
Conv_36: sin_radio: 0.0327076
Conv_39: sin_radio: 0.3923548
Conv_42: sin_radio: 0.1221108