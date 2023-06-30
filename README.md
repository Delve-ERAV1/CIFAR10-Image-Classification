# ERAV1
## S9
### CIFAR10 Image Classification

```
Write a new network that
has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
total RF must be more than 44
one of the layers must use Depthwise Separable Convolution
one of the layers must use Dilated Convolution
use GAP (compulsory):- add FC after GAP to target #of classes (optional)
use albumentation library and apply:
horizontal flip
shiftScaleRotate
coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
make sure you're following code-modularity (else 0 for full assignment) 
upload to Github
Attempt S9-Assignment Solution.
Questions in the Assignment QnA are:
copy and paste your model code from your model.py file (full code) [125]
copy paste output of torch summary [125]
copy-paste the code where you implemented albumentation transformation for all three transformations [125]
copy paste your training log (you must be running validation/text after each Epoch [125]
Share the link for your README.md file. [200]
```


1. Imports: The necessary libraries and modules are imported, including PyTorch, NumPy, TorchVision, Albumentations, Matplotlib, and other relevant modules.

2. Helper Functions:
   - `get_stats(trainloader)`: This function calculates the per-channel mean and standard deviation of the dataset using the provided `trainloader`.
   - `get_loader(transform=None, train=True)`: This function returns a `DataLoader` object for loading the dataset, with an optional transformation applied.
   - `get_summary(model, device)`: This function prints the summary of the model architecture using the `torchsummary` package.
   - `get_device()`: This function returns the device type used for training the model (e.g., "cuda" for GPU or "cpu" for CPU).

3. Custom Dataset Class: The code defines a custom dataset class called `Cifar10SearchDataset`, which inherits from `torchvision.datasets.CIFAR10`. It overrides the `__getitem__` method to apply transformations to the images using Albumentations.

4. Transformation Functions:
   - `get_train_transform(mu, sigma)`: This function returns a composition of data augmentation and normalization transformations for the training set, using the provided mean (`mu`) and standard deviation (`sigma`).
   - `get_test_transform(mu, sigma)`: This function returns a composition of normalization transformations for the test set, using the provided mean (`mu`) and standard deviation (`sigma`).

5. Model Architecture:
   The code defines a custom neural network model called `Net`. The model consists of several convolutional blocks, transition blocks, skip connections, and final blocks. It uses depthwise separable convolutions and skip connections to improve performance. The `forward` method defines the forward pass of the model.

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 3, 32, 32]              27
       BatchNorm2d-2            [-1, 3, 32, 32]               6
              ReLU-3            [-1, 3, 32, 32]               0
            Conv2d-4           [-1, 32, 34, 34]              96
       BatchNorm2d-5           [-1, 32, 34, 34]              64
              ReLU-6           [-1, 32, 34, 34]               0
depthwise_separable_conv-7           [-1, 32, 34, 34]               0
            Conv2d-8           [-1, 32, 34, 34]             288
       BatchNorm2d-9           [-1, 32, 34, 34]              64
             ReLU-10           [-1, 32, 34, 34]               0
           Conv2d-11           [-1, 32, 36, 36]           1,024
      BatchNorm2d-12           [-1, 32, 36, 36]              64
             ReLU-13           [-1, 32, 36, 36]               0
depthwise_separable_conv-14           [-1, 32, 36, 36]               0
           Conv2d-15           [-1, 32, 36, 36]             288
      BatchNorm2d-16           [-1, 32, 36, 36]              64
             ReLU-17           [-1, 32, 36, 36]               0
           Conv2d-18           [-1, 32, 38, 38]           1,024
      BatchNorm2d-19           [-1, 32, 38, 38]              64
             ReLU-20           [-1, 32, 38, 38]               0
depthwise_separable_conv-21           [-1, 32, 38, 38]               0
           Conv2d-22           [-1, 32, 18, 18]           9,216
           Conv2d-23           [-1, 32, 18, 18]             288
      BatchNorm2d-24           [-1, 32, 18, 18]              64
             ReLU-25           [-1, 32, 18, 18]               0
           Conv2d-26           [-1, 32, 20, 20]           1,024
      BatchNorm2d-27           [-1, 32, 20, 20]              64
             ReLU-28           [-1, 32, 20, 20]               0
depthwise_separable_conv-29           [-1, 32, 20, 20]               0
           Conv2d-30           [-1, 32, 20, 20]             288
      BatchNorm2d-31           [-1, 32, 20, 20]              64
             ReLU-32           [-1, 32, 20, 20]               0
           Conv2d-33           [-1, 32, 22, 22]           1,024
      BatchNorm2d-34           [-1, 32, 22, 22]              64
             ReLU-35           [-1, 32, 22, 22]               0
depthwise_separable_conv-36           [-1, 32, 22, 22]               0
           Conv2d-37           [-1, 32, 22, 22]             288
      BatchNorm2d-38           [-1, 32, 22, 22]              64
             ReLU-39           [-1, 32, 22, 22]               0
           Conv2d-40           [-1, 64, 24, 24]           2,048
      BatchNorm2d-41           [-1, 64, 24, 24]             128
             ReLU-42           [-1, 64, 24, 24]               0
depthwise_separable_conv-43           [-1, 64, 24, 24]               0
           Conv2d-44           [-1, 64, 24, 24]           2,048
      BatchNorm2d-45           [-1, 64, 24, 24]             128
             ReLU-46           [-1, 64, 24, 24]               0
      BatchNorm2d-47           [-1, 64, 24, 24]             128
   skipConnection-48           [-1, 64, 24, 24]               0
      BatchNorm2d-49           [-1, 64, 24, 24]             128
           Conv2d-50           [-1, 64, 11, 11]          36,864
      BatchNorm2d-51           [-1, 64, 11, 11]             128
           Conv2d-52           [-1, 64, 11, 11]             576
      BatchNorm2d-53           [-1, 64, 11, 11]             128
             ReLU-54           [-1, 64, 11, 11]               0
           Conv2d-55           [-1, 32, 13, 13]           2,048
      BatchNorm2d-56           [-1, 32, 13, 13]              64
             ReLU-57           [-1, 32, 13, 13]               0
depthwise_separable_conv-58           [-1, 32, 13, 13]               0
           Conv2d-59           [-1, 32, 13, 13]             288
      BatchNorm2d-60           [-1, 32, 13, 13]              64
             ReLU-61           [-1, 32, 13, 13]               0
           Conv2d-62           [-1, 32, 15, 15]           1,024
      BatchNorm2d-63           [-1, 32, 15, 15]              64
             ReLU-64           [-1, 32, 15, 15]               0
depthwise_separable_conv-65           [-1, 32, 15, 15]               0
           Conv2d-66           [-1, 32, 15, 15]             288
      BatchNorm2d-67           [-1, 32, 15, 15]              64
             ReLU-68           [-1, 32, 15, 15]               0
           Conv2d-69           [-1, 64, 17, 17]           2,048
      BatchNorm2d-70           [-1, 64, 17, 17]             128
             ReLU-71           [-1, 64, 17, 17]               0
depthwise_separable_conv-72           [-1, 64, 17, 17]               0
           Conv2d-73           [-1, 64, 17, 17]           4,096
      BatchNorm2d-74           [-1, 64, 17, 17]             128
             ReLU-75           [-1, 64, 17, 17]               0
      BatchNorm2d-76           [-1, 64, 17, 17]             128
   skipConnection-77           [-1, 64, 17, 17]               0
      BatchNorm2d-78           [-1, 64, 17, 17]             128
           Conv2d-79             [-1, 64, 8, 8]          36,864
      BatchNorm2d-80             [-1, 64, 8, 8]             128
           Conv2d-81             [-1, 64, 8, 8]             576
      BatchNorm2d-82             [-1, 64, 8, 8]             128
             ReLU-83             [-1, 64, 8, 8]               0
           Conv2d-84           [-1, 64, 10, 10]           4,096
      BatchNorm2d-85           [-1, 64, 10, 10]             128
             ReLU-86           [-1, 64, 10, 10]               0
depthwise_separable_conv-87           [-1, 64, 10, 10]               0
           Conv2d-88           [-1, 64, 10, 10]             576
      BatchNorm2d-89           [-1, 64, 10, 10]             128
             ReLU-90           [-1, 64, 10, 10]               0
           Conv2d-91          [-1, 128, 12, 12]           8,192
      BatchNorm2d-92          [-1, 128, 12, 12]             256
             ReLU-93          [-1, 128, 12, 12]               0
depthwise_separable_conv-94          [-1, 128, 12, 12]               0
           Conv2d-95          [-1, 128, 12, 12]           1,152
      BatchNorm2d-96          [-1, 128, 12, 12]             256
             ReLU-97          [-1, 128, 12, 12]               0
           Conv2d-98          [-1, 256, 14, 14]          32,768
      BatchNorm2d-99          [-1, 256, 14, 14]             512
            ReLU-100          [-1, 256, 14, 14]               0
depthwise_separable_conv-101          [-1, 256, 14, 14]               0
          Conv2d-102          [-1, 256, 14, 14]          16,384
     BatchNorm2d-103          [-1, 256, 14, 14]             512
            ReLU-104          [-1, 256, 14, 14]               0
     BatchNorm2d-105          [-1, 256, 14, 14]             512
  skipConnection-106          [-1, 256, 14, 14]               0
       AvgPool2d-107            [-1, 256, 1, 1]               0
          Conv2d-108            [-1, 100, 1, 1]          25,600
          Conv2d-109             [-1, 10, 1, 1]           1,000
================================================================
Total params: 198,153
Trainable params: 198,153
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 17.62
Params size (MB): 0.76
Estimated Total Size (MB): 18.39
----------------------------------------------------------------
```

6. Training and Testing:
   - `train` function: This function is used for training the model. It iterates over the training data, performs forward and backward passes, updates the optimizer, and calculates training losses and accuracies.
   - `test` function: This function is used for evaluating the model on the test data. It calculates the test loss and accuracy.

7. Training Loop:
   The code includes a loop that trains and tests the model for a specified number of epochs (`EPOCHS`). It initializes the model, optimizer, and learning rate scheduler. In each epoch, it calls the `train` and `test` functions and updates the learning rate using the `OneCycleLR` scheduler.

# Model Documentation

This documentation provides an overview of the implementation details and usage instructions for the `Net` model, which is designed for image classification on the CIFAR-10 dataset. The model architecture consists of several convolutional blocks, skip connections, and transition blocks.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Dilation](#Dilation)
- [Depthwise Separable Convolution](#Depthwise-Separable-Convolution)
- [OneCycle Learning Rate Schedule](#OnceCycle)
- [Training](#training)
- [Evaluation](#evaluation)
- [Misclassified Images](#misclassified)

## Model Architecture<a name="model-architecture"></a>

The `Net` model is defined in the code provided. It consists of the following major components:

### Convolution Blocks
1. `convblock1`: This block includes three depthwise separable convolutional layers with 32 output channels each. It forms the initial part of the network.
2. `convblock2`: This block includes three depthwise separable convolutional layers with 32 and 64 output channels, respectively.
3. `convblock3`: This block includes three depthwise separable convolutional layers with 32 and 64 output channels, respectively.
4. `convblock4`: This block includes three depthwise separable convolutional layers with 64, 128, and 256 output channels, respectively.

### Transition Blocks
1. `transblock1`: This block includes a convolutional layer that with an output of 32 channels and downsamples the spatial dimensions by a factor of 2.
2. `transblock2`: This block includes a batch normalization layer followed by a convolutional layer with an output of 32 channels and downsamples the spatial dimensions by a factor of 2.
3. `transblock3`: This block includes a batch normalization layer followed by a convolutional layer with an output of 64 channels and downsamples the spatial dimensions by a factor of 2.

### Skip Connections
1. `skipConnection1`: This skip connection connects the output of `convblock1` to `convblock2` by applying a 1x1 convolution and batch normalization with a stride of 2.
2. `skipConnection2`: This skip connection connects the output of `convblock2` to `convblock3` by applying a 1x1 convolution and batch normalization with a stride of 2.
3. `skipConnection3`: This skip connection connects the output of `convblock3` to `convblock4` by applying a 1x1 convolution and batch normalization with a stride of 2.

### Final Block
The final block consists of two 1x1 convolutional layers that reduce the number of channels to 100 and 10, respectively. It is followed by a global average pooling layer (`nn.AvgPool2d`) and a reshape operation to convert the output to the desired shape.

## Usage

To use the `Net` model, follow these steps:

1. Instantiate the model: `model = Net()`
2. Move the model to the desired device: `model.to(device)`

The model is now ready to be used for training or evaluation.

## Training

To train the `Net` model, the provided code includes a `train` function that performs the following steps:

1. Set the model to training mode: `model.train()`
2. Iterate over the training data batches.
3. Zero the gradients: `optimizer.zero_grad()`
4. Forward pass: Compute the predicted output using the model: `y_pred = model(data)`
5. Compute the loss: `loss = F.nll_loss(y_pred, target)`
6. Backpropagation: Compute gradients and update model weights: `loss.backward(), optimizer.step()

```python
model = Net().to(device)

train_losses, train_acc = [], []
test_losses, test_acc = [], []

EPOCHS = 60

model =  Net().to(device)
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=60, steps_per_epoch=len(trainloader), verbose=True)

    train(model, device, trainloader, optimizer, [test_losses, test_acc])
    scheduler.step()
    test(model, device, testloader, [train_losses, train_acc])
```

```
EPOCH: 288
Loss=0.5912972688674927 Batch_id=390 Accuracy=78.19: 100%|██████████| 391/391 [00:16<00:00, 23.37it/s]

Test set: Average loss: 0.0015, Accuracy: 46907/50000 (93.81%)

EPOCH: 289
Loss=0.4554990828037262 Batch_id=390 Accuracy=78.30: 100%|██████████| 391/391 [00:16<00:00, 23.37it/s]

Test set: Average loss: 0.0015, Accuracy: 46677/50000 (93.35%)

EPOCH: 290
Loss=0.6932679414749146 Batch_id=390 Accuracy=78.48: 100%|██████████| 391/391 [00:16<00:00, 23.28it/s]

Test set: Average loss: 0.0015, Accuracy: 46915/50000 (93.83%)

EPOCH: 291
Loss=0.5505960583686829 Batch_id=390 Accuracy=78.39: 100%|██████████| 391/391 [00:17<00:00, 22.69it/s]

Test set: Average loss: 0.0014, Accuracy: 46958/50000 (93.92%)

EPOCH: 292
Loss=0.5868149995803833 Batch_id=390 Accuracy=77.93: 100%|██████████| 391/391 [00:17<00:00, 22.23it/s]

Test set: Average loss: 0.0015, Accuracy: 46807/50000 (93.61%)

EPOCH: 293
Loss=0.674118161201477 Batch_id=390 Accuracy=78.26: 100%|██████████| 391/391 [00:17<00:00, 22.36it/s]

Test set: Average loss: 0.0015, Accuracy: 46925/50000 (93.85%)

EPOCH: 294
Loss=0.7092744708061218 Batch_id=390 Accuracy=78.37: 100%|██████████| 391/391 [00:16<00:00, 23.25it/s]

Test set: Average loss: 0.0014, Accuracy: 46975/50000 (93.95%)

EPOCH: 295
Loss=0.45869630575180054 Batch_id=390 Accuracy=78.35: 100%|██████████| 391/391 [00:16<00:00, 23.14it/s]

Test set: Average loss: 0.0014, Accuracy: 46942/50000 (93.88%)

EPOCH: 296
Loss=0.6513633131980896 Batch_id=390 Accuracy=78.32: 100%|██████████| 391/391 [00:17<00:00, 22.01it/s]

Test set: Average loss: 0.0015, Accuracy: 46822/50000 (93.64%)

EPOCH: 297
Loss=0.7121884822845459 Batch_id=390 Accuracy=78.25: 100%|██████████| 391/391 [00:17<00:00, 23.00it/s]

Test set: Average loss: 0.0014, Accuracy: 47029/50000 (94.06%)

EPOCH: 298
Loss=0.6097805500030518 Batch_id=390 Accuracy=78.54: 100%|██████████| 391/391 [00:17<00:00, 22.95it/s]

Test set: Average loss: 0.0015, Accuracy: 46924/50000 (93.85%)

EPOCH: 299
Loss=0.727354109287262 Batch_id=390 Accuracy=78.05: 100%|██████████| 391/391 [00:16<00:00, 23.16it/s]

Test set: Average loss: 0.0014, Accuracy: 46933/50000 (93.87%)
```

## Dilation

Dilation is a technique used in convolutional neural networks (CNNs) to increase the receptive field of filters without increasing the number of parameters. It introduces gaps between the kernel elements, allowing the filters to capture features with larger spatial extents. In the `Net` model, dilation is not explicitly used. However, if you want to incorporate dilation in your model, you can modify the convolutional layers accordingly.

To apply dilation in a convolutional layer, you can specify the `dilation` parameter when defining the layer. This parameter determines the spacing between the kernel elements. For example, to introduce a dilation factor of 2 in a convolutional layer, you can set `dilation=2`. Higher dilation factors result in larger receptive fields, which can capture more global information.

In the `Net` model code, you can add the `dilation` parameter to the desired convolutional layers, such as `convblock1`, `convblock2`, etc. Experimenting with different dilation factors and observing the impact on the model's performance can help you determine the optimal settings for your specific task.

![dilate](https://github.com/Delve-ERAV1/S9/assets/11761529/9bde9393-dbc5-4c2d-adba-80aa06cdb114)


## Depthwise Separable Convolution

Depthwise separable convolution is a technique that decomposes a standard convolutional layer into two separate operations: a depthwise convolution and a pointwise convolution. This approach reduces the computational cost and the number of parameters while still capturing spatial and channel-wise dependencies effectively.

In the `Net` model, depthwise separable convolution is used in the convolutional blocks (`convblock1`, `convblock2`, etc.) to extract features from the input data. The depthwise separable convolution is performed as follows:

1. Depthwise Convolution: A depthwise convolution applies a separate convolutional filter to each input channel. This operation captures spatial correlations within each channel independently.

2. Pointwise Convolution: A pointwise convolution, also known as a 1x1 convolution, performs a 1x1 convolution on the output of the depthwise convolution. It combines the information from different channels and projects them into a new feature space.

By employing depthwise separable convolutions, the model reduces the computational complexity while maintaining a good representation capacity. This allows the model to learn discriminative features effectively while being more efficient in terms of memory and computation.

![DSC](https://github.com/Delve-ERAV1/S9/assets/11761529/c2a0aa5c-726f-43fe-8931-0210fffb11d8)

## OneCycle Learning Rate Schedule

The OneCycle learning rate (LR) schedule is a technique commonly used in training deep neural networks. It dynamically adjusts the learning rate during training to accelerate convergence and improve generalization. This technique involves gradually increasing the learning rate to a maximum value and then gradually decreasing it back to a small value.

Here's a high-level overview of the OneCycle LR schedule:

1. Setting the Learning Rate Range: Initially, you define a range of learning rates. This range typically spans several orders of magnitude, starting from a small LR value (e.g., 1e-5) to a larger LR value (e.g., 1e-1).

2. Defining the Number of Steps: You determine the total number of training steps or epochs required for your training process.

3. Calculating the LR Schedule: Using the total number of steps and the learning rate range, you can calculate the LR schedule. The schedule involves a phase of increasing LR, followed by a phase of decreasing LR.

4. Increasing Phase: In the increasing phase, the LR gradually increases from the lower bound to the upper bound of the learning rate range. This allows the model to quickly explore different regions of the loss landscape and escape potential local minima.

5. Decreasing Phase: In the decreasing phase, the LR is gradually decreased from the upper bound back to the lower bound. This helps the model converge and refine its parameters.

By applying the OneCycle LR schedule, we can potentially achieve faster convergence, better generalization, and improved performance for your deep learning models. The specific implementation details, such as the LR range, step count, and LR schedule, may vary depending on the specific task and dataset.


# Misclassified Images
![misclass](https://github.com/Delve-ERAV1/S9/assets/11761529/148833ea-62f6-475c-82f1-56c93818b514)
