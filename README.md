# Dual-Phenological Convolutional Networks for Paddy Rice Mapping in Cloud-Prone Regions

leveraging composite Sentinel 2 optical imagery from both the transplanting period and the peak period of paddy rice.  The goal is that even if observations are available for only one of the two periods, paddy rice mapping can still arrive relateively high accuracy.

To minimize the impact of missing values in the input data, we implemented a mask average pooling layer and a loss function that considers only valid observations .
## The flow diagram
![flow](./src/assets/Flow.jpg)
## The visualization comparsion
![comparison](./src/assets/comparsion.png)
## The accuracy metrics of 5-fold cross validation
![accuracy](./src/assets/accuracy.png)


