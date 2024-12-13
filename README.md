# Model 1
### Targets: 
 - Start with the model from the session 7 class.
### Results: 
- Parameters: 13.8k
- Best Train Accuracy: 98.95
- Best Test Accuracy: 99.41 (15th Epoch)
### Analysis: 
 - The model is a simple CNN with 7 layers. It uses dropout to prevent overfitting. The model is able to achieve high accuracy on the test set, but it is not able to achieve the accuracy of 99.41% with less than 8K parameters.
 - The model is able to achieve 99.41% accuracy on the test set with 13.8k parameters.
 - The model is able to achieve 98.95% accuracy on the train set with 13.8k parameters.


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


# Model 2
### Targets: 
 - Improve the model to reduce the number of parameters less than 8K.
 - Key changes made to reduce parameters while maintaining performance:
   - Reduced initial channels from 16 to 8 in the first layer
   - Reduced channels in subsequent layers proportionally
   - Removed one convolution layer (original convblock7)
   - Adjusted the architecture to maintain receptive field while using fewer parameters
   - Justification for changes:
     - Channel reduction: The original model had more channels than necessary in early layers. By starting with 8 channels and gradually increasing, we maintain feature extraction capability while reducing parameters.
     - Layer removal: Removed one redundant convolution layer as the receptive field was already sufficient with the remaining layers.          
     - Maintained key architectural elements:
       - Kept the dropout for regularization
       - Preserved batch normalization for stable training
       - Kept the GAP layer for spatial dimension reduction
       - Maintained the basic structure of conv->relu->batchnorm->dropout
     - These changes should reduce the parameter count to under 8K while maintaining the model's ability to achieve ~99.4% accuracy on MNIST. The reduced architecture still has sufficient capacity to learn the required features for digit classification.


### Results: 
 - Parameters: 5.0k
 - Best Train Accuracy: 98.36   
 - Best Test Accuracy: 99.08 (14th Epoch)
### Analysis: 
 - The model is a simple CNN with 7 layers. It uses dropout to prevent overfitting. The model is able to achieve high accuracy on the test set, but it is not able to achieve the accuracy of 99.41% with less than 8K parameters.
 - The model is able to achieve 99.08% accuracy on the test set with 5.0k parameters.
 - The model is able to achieve 98.36% accuracy on the train set with 5.0k parameters.

## Code for Model 2

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
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 8

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)        
        x = self.convblock7(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

```

## Console Log for Model 2

```
PS E:\AI\github\era-v3-s7-cnn-cloud> python src/train.py
cpu
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,152
              ReLU-6           [-1, 16, 24, 24]               0
       BatchNorm2d-7           [-1, 16, 24, 24]              32
           Dropout-8           [-1, 16, 24, 24]               0
            Conv2d-9            [-1, 8, 24, 24]             128
        MaxPool2d-10            [-1, 8, 12, 12]               0
           Conv2d-11           [-1, 12, 10, 10]             864
             ReLU-12           [-1, 12, 10, 10]               0
      BatchNorm2d-13           [-1, 12, 10, 10]              24
          Dropout-14           [-1, 12, 10, 10]               0
           Conv2d-15             [-1, 12, 8, 8]           1,296
             ReLU-16             [-1, 12, 8, 8]               0
      BatchNorm2d-17             [-1, 12, 8, 8]              24
          Dropout-18             [-1, 12, 8, 8]               0
           Conv2d-19             [-1, 12, 8, 8]           1,296
             ReLU-20             [-1, 12, 8, 8]               0
      BatchNorm2d-21             [-1, 12, 8, 8]              24
          Dropout-22             [-1, 12, 8, 8]               0
        AvgPool2d-23             [-1, 12, 1, 1]               0
           Conv2d-24             [-1, 10, 1, 1]             120
================================================================
Total params: 5,048
Trainable params: 5,048
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.57
Params size (MB): 0.02
Estimated Total Size (MB): 0.60
----------------------------------------------------------------
CUDA Available? False
EPOCH: 0
Loss=0.03700254112482071 Batch_id=937 Accuracy=84.31: 100%|███| 938/938 [01:24<00:00, 11.13it/s]        

Test set: Average loss: 0.0873, Accuracy: 9772/10000 (97.72%)

EPOCH: 1
Loss=0.03686046972870827 Batch_id=937 Accuracy=96.06: 100%|██████████| 938/938 [01:09<00:00, 13.43it/s]

Test set: Average loss: 0.0762, Accuracy: 9770/10000 (97.70%)

EPOCH: 2
Loss=0.11288551241159439 Batch_id=937 Accuracy=96.95: 100%|██████████| 938/938 [01:27<00:00, 10.72it/s] 

Test set: Average loss: 0.0499, Accuracy: 9843/10000 (98.43%)

EPOCH: 3
Loss=0.163455992937088 Batch_id=937 Accuracy=97.34: 100%|████████████| 938/938 [01:22<00:00, 11.43it/s] 

Test set: Average loss: 0.0430, Accuracy: 9865/10000 (98.65%)

EPOCH: 4
Loss=0.024510761722922325 Batch_id=937 Accuracy=97.69: 100%|█████████| 938/938 [01:24<00:00, 11.09it/s] 

Test set: Average loss: 0.0401, Accuracy: 9860/10000 (98.60%)

EPOCH: 5
Loss=0.052453331649303436 Batch_id=937 Accuracy=97.78: 100%|█████████| 938/938 [01:22<00:00, 11.33it/s] 

Test set: Average loss: 0.0388, Accuracy: 9865/10000 (98.65%)

EPOCH: 6
Loss=0.18465520441532135 Batch_id=937 Accuracy=97.91: 100%|██████████| 938/938 [01:23<00:00, 11.23it/s] 

Test set: Average loss: 0.0334, Accuracy: 9896/10000 (98.96%)

EPOCH: 7
Loss=0.36025211215019226 Batch_id=937 Accuracy=97.92: 100%|██████████| 938/938 [01:22<00:00, 11.42it/s] 

Test set: Average loss: 0.0317, Accuracy: 9903/10000 (99.03%)

EPOCH: 8
Loss=0.007759375497698784 Batch_id=937 Accuracy=98.13: 100%|█████████| 938/938 [01:21<00:00, 11.54it/s] 

Test set: Average loss: 0.0361, Accuracy: 9882/10000 (98.82%)

EPOCH: 9
Loss=0.011883535422384739 Batch_id=937 Accuracy=98.11: 100%|█████████| 938/938 [01:21<00:00, 11.50it/s] 

Test set: Average loss: 0.0336, Accuracy: 9881/10000 (98.81%)

EPOCH: 10
Loss=0.2088194191455841 Batch_id=937 Accuracy=98.21: 100%|███████████| 938/938 [01:22<00:00, 11.37it/s] 

Test set: Average loss: 0.0306, Accuracy: 9896/10000 (98.96%)

EPOCH: 11
Loss=0.01457920391112566 Batch_id=937 Accuracy=98.19: 100%|██████████| 938/938 [01:21<00:00, 11.52it/s] 

Test set: Average loss: 0.0288, Accuracy: 9907/10000 (99.07%)

EPOCH: 12
Loss=0.07502304762601852 Batch_id=937 Accuracy=98.20: 100%|██████████| 938/938 [01:26<00:00, 10.84it/s] 

Test set: Average loss: 0.0326, Accuracy: 9897/10000 (98.97%)

EPOCH: 13
Loss=0.007396047003567219 Batch_id=937 Accuracy=98.32: 100%|█████████| 938/938 [01:23<00:00, 11.19it/s] 

Test set: Average loss: 0.0274, Accuracy: 9908/10000 (99.08%)

EPOCH: 14
Loss=0.05654190108180046 Batch_id=937 Accuracy=98.36: 100%|██████████| 938/938 [01:24<00:00, 11.16it/s] 

Test set: Average loss: 0.0311, Accuracy: 9898/10000 (98.98%)


===================
Results:
Parameters: 5.0k
Best Train Accuracy: 98.36
Best Test Accuracy: 99.08 (14th Epoch)
===================
PS E:\AI\github\era-v3-s7-cnn-cloud> 
```


