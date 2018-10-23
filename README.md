# AMNet: Memorability Estimation with Attention
A PyTorch implementation of our paper [AMNet: Memorability Estimation with Attention](https://arxiv.org/abs/1804.03115)
by Jiri Fajtl, Vasileios Argyriou, Dorothy Monekosso and Paolo Remagnino. This paper will be presented 
at [CVPR 2018](http://cvpr2018.thecvf.com/).

==> [AMNet](https://github.com/ok1zjf/AMNet)  

## Path & Cmd

### 服务器文件路径

`/media/Data/qipeng/modified_complete_images`
`/home/qipeng/PicMemorability/AMNet-Rumor`

### 本机路径

`/Users/snow/snow_学习/4-研究生/实验室/AMNet/AMNet/_expt`

### Train Cmd

`python3.5 main.py --train-batch-size 256 --test-batch-size 256 --cnn ResNet50FC --dataset lamem --dataset-root /media/Data/qipeng/modified_complete_images/AMNet-Rumor/lamem/ --train-split train_0 --val-split val_0 --lstm-steps 4`

每次实验前：

- 保存上一次的
  - 日志
  - 模型文件
- 更换gpu设备
- 改实验名
- 改训练输出结果、预测输出结果的文件夹
- 同步代码

### Test Cmd

`python3.5 main.py --test --dataset lamem --dataset-root /media/Data/qipeng/modified_complete_images/AMNet-Rumor/lamem/ --cnn ResNet50FC --model-weights /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Train-Output/lamem_ResNet50FC_lstm3_train_1/weights_54.pkl --test-split 'test_*'`

### Predict Cmd - Rumor / Nonrumor

`python3.5 main.py --cnn ResNet50FC --model-weights /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Train-Output/expt3/lamem_ResNet50FC_lstm2_train_0/weights_54.pkl --eval-images /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/images/rumor --csv-out memorabilities-expt3-rumor.txt --att-maps-out /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/expt3/att_maps/rumor --lstm-steps 2`

`python3.5 main.py --cnn ResNet50FC --model-weights /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Train-Output/expt3/lamem_ResNet50FC_lstm2_train_0/weights_54.pkl --eval-images /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/images/nonrumor --csv-out memorabilities-expt3-nonrumor.txt --att-maps-out /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/expt3/att_maps/nonrumor --lstm-steps 2`

### Predict Cmd - Sample

`python3.5 main.py --cnn ResNet50FC --model-weights /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Train-Output/lamem_ResNet50FC_lstm3_train_1/weights_54.pkl --eval-images /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/images/sample --csv-out memorabilities-sample.txt --att-maps-out /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/att_maps/sample`

## Expt Log

| **实验序号** | 日期      | 主题            | 训练数据             | 超参数                                                | Accuracy | AUC     | 备注                        |
| :----------: | --------- | --------------- | -------------------- | ----------------------------------------------------- | -------- | ------- | --------------------------- |
|    **0**     | 1016-1017 | baseline        | shuffle（不严格1:1） | lstm_steps: 3<br />batch_size: 222<br />epoch: 55     | 0.75963  | 0.84084 |                             |
|    **1**     | 1022      | baseline-可视化 | **train_0, val_0**   | lstm_steps: 3<br />**batch_size: 256**<br />epoch: 55 | 0.75352  | 0.83668 | 召回率高                    |
|    **2**     | 1023      | lstm_steps=1    | train_0, val_0       | **lstm_steps: 1**<br />batch_size: 256<br />epoch: 55 | 0.75537  | 0.84413 | 准确率高                    |
|    **3**     | 1023      | lstm_steps=2    | train_0, val_0       | **lstm_steps: 2**<br />batch_size: 256<br />epoch: 55 | 0.755    | 0.83577 | 观察epoch图，**可能过拟合** |
|    **4**     | 1023      | lstm_steps=4    | train_0, val_0       | **lstm_steps: 4**<br />batch_size: 256<br />epoch: 55 |          |         |                             |

### Expt 0

```
				precision    recall  f1-score   support

          0       0.76      0.75      0.76      2700
          1       0.76      0.77      0.76      2700

avg / total       0.76      0.76      0.76      5400
```

### Expt 1: LSTM3

**Accuracy**

```
				precision    recall  f1-score   support

          0       0.80      0.68      0.73      2700
          1       0.72      0.83      0.77      2700

avg / total       0.76      0.75      0.75      5400
```

### Expt 2: LSTM1

**Accuracy**

```
				precision    recall  f1-score   support

          0       0.72      0.83      0.77      2700
          1       0.80      0.68      0.74      2700

avg / total       0.76      0.76      0.75      5400
```

### Expt 3: LSTM2

**Accuracy**

```
				precision    recall  f1-score   support

          0       0.73      0.80      0.77      2700
          1       0.78      0.71      0.74      2700

avg / total       0.76      0.76      0.75      5400
```



## TODO

### Debug

- 预测时有bug

### Learn

- Linux命令
  - nohup, >>, &
  - kill, ps
  - top
  - ls -lR|grep "^-"|wc -l
- pytorch
  - 多块gpu尝试