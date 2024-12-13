# Model 1
### Targets: Start with the model from the session 7 class.
### Results: 
- Parameters: 13.8k
- Best Train Accuracy: 98.95
- Best Test Accuracy: 99.41 (15th Epoch)
### Analysis: The model is a simple CNN with 7 layers. It uses dropout to prevent overfitting. The model is able to achieve high accuracy on the test set, but it is not able to achieve the accuracy of 99.41% with less than 8K parameters.

## Code for Model 1

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
```

## Console Log for Model 1

```
PS E:\AI\github\era-v3-s7-cnn-cloud> python src/train.py
cpu
----------------------------------------------------------------
        Layer (type)               Output Shape         Param # 
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144 
              ReLU-2           [-1, 16, 26, 26]               0 
       BatchNorm2d-3           [-1, 16, 26, 26]              32 
           Dropout-4           [-1, 16, 26, 26]               0 
            Conv2d-5           [-1, 32, 24, 24]           4,608 
              ReLU-6           [-1, 32, 24, 24]               0 
       BatchNorm2d-7           [-1, 32, 24, 24]              64 
           Dropout-8           [-1, 32, 24, 24]               0 
            Conv2d-9           [-1, 10, 24, 24]             320 
        MaxPool2d-10           [-1, 10, 12, 12]               0 
           Conv2d-11           [-1, 16, 10, 10]           1,440 
             ReLU-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
          Dropout-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,304
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 13,808
Trainable params: 13,808
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.06
Params size (MB): 0.05
Estimated Total Size (MB): 1.12
----------------------------------------------------------------
CUDA Available? False
EPOCH: 0
Loss=0.01941436156630516 Batch_id=937 Accuracy=91.28: 100%|█████████████████████████████████████████████████████████████████████████████| 938/938 [02:13<00:00,  7.01it/s] 

Test set: Average loss: 0.0572, Accuracy: 9821/10000 (98.21%)

EPOCH: 1
Loss=0.03962118551135063 Batch_id=937 Accuracy=97.69: 100%|█████████████████████████████████████████████████████████████████████████████| 938/938 [02:07<00:00,  7.38it/s] 

Test set: Average loss: 0.0518, Accuracy: 9824/10000 (98.24%)

EPOCH: 2
Loss=0.005107395816594362 Batch_id=937 Accuracy=98.19: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:04<00:00,  7.52it/s] 

Test set: Average loss: 0.0303, Accuracy: 9902/10000 (99.02%)

EPOCH: 3
Loss=0.048854272812604904 Batch_id=937 Accuracy=98.31: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:00<00:00,  7.81it/s] 

Test set: Average loss: 0.0272, Accuracy: 9913/10000 (99.13%)

EPOCH: 4
Loss=0.002012245124205947 Batch_id=937 Accuracy=98.56: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:00<00:00,  7.81it/s] 

Test set: Average loss: 0.0279, Accuracy: 9912/10000 (99.12%)

EPOCH: 5
Loss=0.005592297296971083 Batch_id=937 Accuracy=98.64: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:01<00:00,  7.70it/s] 

Test set: Average loss: 0.0233, Accuracy: 9920/10000 (99.20%)

EPOCH: 6
Loss=0.021483028307557106 Batch_id=937 Accuracy=98.64: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:01<00:00,  7.73it/s] 

Test set: Average loss: 0.0233, Accuracy: 9924/10000 (99.24%)

EPOCH: 7
Loss=0.026516681537032127 Batch_id=937 Accuracy=98.74: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:03<00:00,  7.59it/s] 

Test set: Average loss: 0.0229, Accuracy: 9924/10000 (99.24%)

EPOCH: 8
Loss=0.022202573716640472 Batch_id=937 Accuracy=98.75: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:07<00:00,  7.33it/s] 

Test set: Average loss: 0.0219, Accuracy: 9934/10000 (99.34%)

EPOCH: 9
Loss=0.02282341569662094 Batch_id=937 Accuracy=98.89: 100%|█████████████████████████████████████████████████████████████████████████████| 938/938 [02:03<00:00,  7.57it/s] 

Test set: Average loss: 0.0238, Accuracy: 9923/10000 (99.23%)

EPOCH: 10
Loss=0.014295128174126148 Batch_id=937 Accuracy=98.89: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:02<00:00,  7.63it/s] 

Test set: Average loss: 0.0220, Accuracy: 9926/10000 (99.26%)

EPOCH: 11
Loss=0.0047846161760389805 Batch_id=937 Accuracy=98.88: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [02:01<00:00,  7.73it/s] 

Test set: Average loss: 0.0221, Accuracy: 9935/10000 (99.35%)

EPOCH: 12
Loss=0.009756717830896378 Batch_id=937 Accuracy=98.88: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:10<00:00,  7.17it/s] 

Test set: Average loss: 0.0207, Accuracy: 9937/10000 (99.37%)

EPOCH: 13
Loss=0.19465979933738708 Batch_id=937 Accuracy=98.95: 100%|█████████████████████████████████████████████████████████████████████████████| 938/938 [02:01<00:00,  7.71it/s] 
Test set: Average loss: 0.0197, Accuracy: 9938/10000 (99.38%)

EPOCH: 14
EPOCH: 14
Loss=0.052712198346853256 Batch_id=937 Accuracy=98.94: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:04<00:00,  7.54it/s] ████████| 938/938 [02:04<00:00,  7.54it/s]

Test set: Average loss: 0.0206, Accuracy: 9941/10000 (99.41%)


===================
Results:
Parameters: 13.8k
Best Train Accuracy: 98.95
Best Test Accuracy: 99.41 (15th Epoch)
===================
PS E:\AI\github\era-v3-s7-cnn-cloud>
```
