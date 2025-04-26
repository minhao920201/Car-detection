# Car-detection
## Dataset Preparation
The code currently supports CCD. The dataset needs to be prepared under the folder `data/.`  
Please refer to the [CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset) repo for downloading and deployment.
## Pre-requirement
Install the packages that we need for the following program.
```
pip install -r requirements.txt
```
## Preprocessing
Remove noise from images, convert to grayscale, and unify image sizes.
+ [Preprocessing](https://github.com/minhao920201/Car-detection/blob/main/preprocessing.py)
## Train Model
Building CNN+LSTM model to identify traffic accidents.
+ [Model](https://github.com/minhao920201/Car-detection/blob/main/CNNLSTM.py)

**Loss Function**

![image](https://github.com/minhao920201/Car-detection/blob/main/demo/loss.png)  
**Predict test case**

![image](https://github.com/minhao920201/Car-detection/blob/main/demo/classfication%20report.jpg)
