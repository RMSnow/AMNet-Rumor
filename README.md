# AMNet: Memorability Estimation with Attention
A PyTorch implementation of our paper [AMNet: Memorability Estimation with Attention](https://arxiv.org/abs/1804.03115)
by Jiri Fajtl, Vasileios Argyriou, Dorothy Monekosso and Paolo Remagnino. This paper will be presented 
at [CVPR 2018](http://cvpr2018.thecvf.com/).

==> [AMNet](https://github.com/ok1zjf/AMNet)  

## 实验流程

1. Train

   开始之前：

   - `amnet.py`：（1）修改实验序号（2）修改AMNet-Train-Output文件夹
   - `config.py`：修改GPU设备
   - 服务器：（1）同步代码（2）创建AMNet-Train-Output文件夹
   - 修改Cmd命令的参数，如`--lstm-steps`等

   训练完成：

   - 保存可视化结果
   - 保存log日志文件
   - 保存模型文件`.pkl`、训练日志`.csv`

2. Predict

   开始之前：

   - 服务器：创建AMNet-Predict文件夹
   - 修改Cmd命令的参数，如`--lstm-steps`等

   预测完成：

   - 保存部分att_maps图

     `scp qipeng@10.25.0.232:/media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/images/rumor/0a* /Users/snow/snow_学习/4-研究生/实验室/AMNet/AMNet/att_maps/expt4/rumor`

     `scp qipeng@10.25.0.232:/media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/images/nonrumor/0* /Users/snow/snow_学习/4-研究生/实验室/AMNet/AMNet/att_maps/expt4/nonrumor`

   - 保存预测结果：`mem`文件

     `scp qipeng@10.25.0.232:/home/qipeng/PicMemorability/AMNet-Rumor-Baseline/*expt4* /Users/snow/snow_学习/4-研究生/实验室/AMNet/AMNet/_expt`

3. Eval

   - 记录实验日志：
     - Accuracy：sklearn分类报告
     - AUC
     - `README.md`：可视化图片添加
   - 分析`eval-expt*.csv`，保存相应的图片 

4. 每日结束

   - git同步代码

## Path & Cmd

### 服务器文件路径

`/media/Data/qipeng/modified_complete_images`
`/home/qipeng/PicMemorability/AMNet-Rumor`

### 本机路径

`/Users/snow/snow_学习/4-研究生/实验室/AMNet/AMNet/_expt`

### Train Cmd

`python3.5 main.py --train-batch-size 256 --test-batch-size 256 --cnn ResNet50FC --dataset lamem --dataset-root /media/Data/qipeng/modified_complete_images/AMNet-Rumor/lamem/ --train-split train_0 --val-split val_0 --lstm-steps 6`

### Predict Cmd - Rumor / Nonrumor

`python3.5 main.py --cnn ResNet50FC --model-weights /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Train-Output/expt1/lamem_ResNet50FC_lstm3_train_0/weights_35.pkl --eval-images /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/images/rumor --csv-out memorabilities-expt1.2-rumor.txt --att-maps-out /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/expt1.2/att_maps/rumor --lstm-steps 3 --gpu 1`

`python3.5 main.py --cnn ResNet50FC --model-weights /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Train-Output/expt1/lamem_ResNet50FC_lstm3_train_0/weights_35.pkl --eval-images /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/images/nonrumor --csv-out memorabilities-expt1.2-nonrumor.txt --att-maps-out /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/expt1.2/att_maps/nonrumor --lstm-steps 3 --gpu 3`

`python3.5 main.py --cnn ResNet50FC --model-weights /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Train-Output/expt4/lamem_ResNet50FC_lstm4_train_0/weights_35.pkl --eval-images /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/images/rumor --csv-out memorabilities-expt4.2-rumor.txt --att-maps-out /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/expt4.2/att_maps/rumor --lstm-steps 4 --gpu 1`

`python3.5 main.py --cnn ResNet50FC --model-weights /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Train-Output/expt4/lamem_ResNet50FC_lstm4_train_0/weights_35.pkl --eval-images /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/images/nonrumor --csv-out memorabilities-expt4.2-nonrumor.txt --att-maps-out /media/Data/qipeng/modified_complete_images/AMNet-Rumor/AMNet-Predict/expt4.2/att_maps/nonrumor --lstm-steps 4 --gpu 3`

## Expt Log

| **实验序号** | 日期      | 主题            | 训练数据             | 超参数                                                | Accuracy | AUC     | 备注                        |
| :----------: | --------- | --------------- | :------------------- | ----------------------------------------------------- | -------- | ------- | --------------------------- |
|    **0**     | 1016-1017 | baseline        | shuffle（不严格1:1） | lstm_steps: 3<br />batch_size: 222<br />epoch: 55     | 0.75963  | 0.84084 |                             |
|    **1**     | 1022      | baseline-可视化 | **train_0, val_0**   | lstm_steps: 3<br />**batch_size: 256**<br />epoch: 55 | 0.75352  | 0.83668 | 召回率高                    |
|    **2**     | 1023      | lstm_steps=1    | train_0, val_0       | **lstm_steps: 1**<br />batch_size: 256<br />epoch: 55 | 0.75537  | 0.84413 | 准确率高                    |
|    **3**     | 1023      | lstm_steps=2    | train_0, val_0       | **lstm_steps: 2**<br />batch_size: 256<br />epoch: 55 | 0.755    | 0.83577 | 观察epoch图，**可能过拟合** |
|    **4**     | 1023      | lstm_steps=4    | train_0, val_0       | **lstm_steps: 4**<br />batch_size: 256<br />epoch: 55 | 0.73389  | 0.82929 |                             |
|    **5**     | 1024      | lstm_steps=5    | train_0, val_0       | **lstm_steps: 5**<br />batch_size: 256<br />epoch: 55 |          |         |                             |
|    **6**     | 1024      | lstm_steps=6    | train_0, val_0       | **lstm_steps: 6**<br />batch_size: 256<br />epoch: 55 |          |         |                             |
|   **2.2**    | 1024      | weights_35      |                      |                                                       | 0.76111  | 0.838   |                             |
|   **3.2**    | 1024      | weights_35      |                      |                                                       | 0.75685  | 0.843   | 减少epoch有效               |
|   **1.2**    | 1024      | weights_35      |                      |                                                       | 0.74167  | 0.838   |                             |
|   **4.2**    | 1024      | weights_35      |                      |                                                       | 0.74815  | 0.840   | 减少epoch有效               |

### 部分规律

|     Epoch = 54     |  AUC  | 0-precision | 1-precision | 0-recall | 1-recall | 0-f1 | 1-f1 |
| :----------------: | :---: | :---------: | :---------: | :------: | :------: | :--: | :--: |
|    Expt2: LSTM1    | 0.844 |    0.72     |    0.80     |   0.83   |   0.68   | 0.77 | 0.74 |
|    Expt3: LSTM2    | 0.836 |    0.73     |    0.78     |   0.80   |   0.71   | 0.77 | 0.74 |
|    Expt1: LSTM3    | 0.837 |    0.80     |    0.72     |   0.68   |   0.83   | 0.73 | 0.77 |
|    Expt4: LSTM4    | 0.829 |    0.69     |    0.81     |   0.86   |   0.61   | 0.76 | 0.69 |
|   **Epoch = 35**   |       |             |             |          |          |      |      |
|   Expt2.2: LSTM1   | 0.838 |    0.77     |    0.75     |   0.74   |   0.78   | 0.76 | 0.76 |
| **Expt3.2: LSTM2** | 0.843 |    0.73     |    0.79     |   0.82   |   0.69   | 0.77 | 0.74 |
|   Expt1.2: LSTM3   | 0.838 |    0.82     |    0.69     |   0.62   |   0.87   | 0.71 | 0.77 |
| **Expt4.2: LSTM4** | 0.840 |    0.71     |    0.80     |   0.83   |   0.67   | 0.77 | 0.73 |

### Expt 0

**Accuracy**

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

![1](https://raw.githubusercontent.com/RMSnow/AMNet-Rumor/master/_expt/expt1-training-epoch54.png)

![1](https://raw.githubusercontent.com/RMSnow/AMNet-Rumor/master/_expt/expt1-eval-epoch54.png)

### Expt 2: LSTM1

**Accuracy**

```
				precision    recall  f1-score   support

          0       0.72      0.83      0.77      2700
          1       0.80      0.68      0.74      2700

avg / total       0.76      0.76      0.75      5400
```

![1](https://raw.githubusercontent.com/RMSnow/AMNet-Rumor/master/_expt/expt2-training-epoch54.png)

![1](https://raw.githubusercontent.com/RMSnow/AMNet-Rumor/master/_expt/expt2-eval-epoch54.png)

### Expt 3: LSTM2

**Accuracy**

```
				precision    recall  f1-score   support

          0       0.73      0.80      0.77      2700
          1       0.78      0.71      0.74      2700

avg / total       0.76      0.76      0.75      5400
```

![1](https://raw.githubusercontent.com/RMSnow/AMNet-Rumor/master/_expt/expt3-training-epoch54.png)

![1](https://raw.githubusercontent.com/RMSnow/AMNet-Rumor/master/_expt/expt3-eval-epoch54.png)

### Expt4: LSTM4

```
				precision    recall  f1-score   support

          0       0.69      0.86      0.76      2700
          1       0.81      0.61      0.69      2700

avg / total       0.75      0.73      0.73      5400
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