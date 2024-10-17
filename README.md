# FSRCNN (fast Super Resolution Convolutional Neural Network)
## Introduction :-
Though SRCNN is already faster than most previous learning-based methods, the processing speed on large images is still unsatisfactory. For example, to upsample an 240×240 image by a factor of 3, the speed of the original SRCCNN is about 1.32 fps, which is far from real time (24 fps). 

There are two inherent limitations of SRCNN that restricts its runnnign speed :-
1. Pre-processing step, the original LR image needs to be upsampled to the desired size using bicubic interpolation to form the input. Thus the computation complexity of SRCNN grows quadratically with the spatial size of the HR image. For the upscaling factor n, the computational cost of convolution with the interpolated LR image will be n^2 times of that for the original LR one.
   
2. The second restriction lies on the costly non-linear mapping step.

So, to solve above limitations of SRCNN , FSRCNN comes into existance:-

 1. To solve the first problem, a deconvolution layer is used to replace the bicubic interpolation. To further ease the computational burden, the deconvolution layer is placed at the end of the network, then the computational complexity is only proportional to the spatial size of the original LR image.
    
 2. For the second problem, a shrinking and an expanding layer is placed at the beginning and at the end of the mapping layer separately to restrict mapping in a low-dimensional feature space.

## FSRCNN Architecture :-

![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-12%20233823.png)

FSRCNN can be decomposed into five parts – feature extraction, shrinking, mapping, expanding and deconvolution.
The first four parts are convolution layers, while the last one is a deconvolution layer.

- ### Feature Extraction :-

  FSRCNN performs feature extraction on the original LR image without interpolation.
  The first layer can be represented as Conv(5, d, 1).

- ### Shrinking :-

  We add a shrinking layer after the feature extraction layer to reduce the LR feature dimension d. We fix the filter size to be f2 = 1.

  n2 = s << d

  The second layer can be represented as Conv(1, s, d).
  This strategy greatly reduces the number of parameters

- ### Non-Linear Mapping :-
  
  We use multiple 3 × 3 layers to replace a single wide one.
  
  To be consistent, all mapping layers contain the same number of filters n3 = s.
  
  Then the non-linear mapping part can be represented as m × Conv(3, s, s). Here, m = number of mapping layers.

- ### Expanding :-

  The expanding layer acts like an inverse process of the shrinking layer.

  If we generate the HR image directly from these low-dimensional features, the final restoration quality will be poor. Therefore, we add an expanding layer after the mapping part to expand the HR feature dimension.

  The expanding layer is Conv(1, d, s)

- ### Deconvolution :-
  
  The last part is a deconvolution layer, which upsamples and aggregates the previous features with a set of deconvolution filters. The deconvolution can be regarded as an inverse operation of the convolution.
  For convolution, the filter is convolved with the image with a stride k, and the output is 1/k times of the input. Contrarily, if we exchange the position of the input and output, the output will be k times of the input.
  We take advantage of this property to set the stride k = n, which is the desired upscaling factor.

  The deconvolution layer is DeConv(9, 1, d).

- ### PReLU :-

  For the activation function after each convolution layer, we use of Parametric Rectified Linear Unit (PReLU)

## Step By Step Guide :-

- ### Datalaoding:-

  We use *'BSD100'* dataset here .
  
  *'image_SRF_2'* as training set , *'image_SRF_3'* as validation set and *'image_SRF_4'* as testing set .
  
  Here, no preprocessing is required like that of bicubic interpolation.

  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20010754.png)

  This is the only transformation that we apply here.

- ### FSRCNN model :-

  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20011204.png)
  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20011228.png)

  - Our first layer is *feature extraction* layer which consists of *56 filters* of size *5 * 5 * 3*
  - Our second layer is *Shrinking layer* which consists of *12 filters* of size *1 * 1 * 56*.
  - Our third layer is *mapping layer* is a set of *4 layers*, each consisting of *12 filters* of size *3 * 3 * 12*.
  - our fourth layer is *expansion layer* which consists of *56 filters* of size *1 * 1 * 12*.
  - Our last layer is *Deconvolutional layer* which consists of 3 filters as we want RGB image , size of *9 * 9 * 56* and upsacling factor here is *2* for *'image_SRF_2'(training dataset)*, upscaling factor for *'image_SRF_4'(test set)* is *3*.

- ## Loss Calculation :-

  Here, we are using both *MSE* and *Perceptual loss*.

  Perceptual loss is a type of loss function that aims to capture the perceptual differences between images , rather than just pixel wise differences. Perceptual differences refer to how humans visually perceive 
  distinctions between images rather than just mathematical difference in pixel values. Two images can mathematically different but look very similar to humans or vice versa.

  So, perceptual loss function aims to capture these differences by using features from pre trained network. Here we are using 'VGG-16 pre trained network'.
  
  > - The target HR image and generated HR image are passed through pre trained network.
  > - The network processes both images , producing feature maps at various layers.
  > - For each selected layer , the feature maps of generated image and target image are compared. The compariosn is usually done by MSE between feature maps.
  > - The losses from multiple layers are often combined. Earlier layer captures low level features and deeper layers capture high level semantic information.
  > - The perceptual loss is often combined with pixel wise MSE loss for final optimization objective.
  
  ![](https://github.com/Srishti002/SRCNN/blob/main/Screenshot%202024-10-12%20030818.png)

  When using VGG-16 for perceptual loss, we typically only use the convolutional layers, discarding the fully connected layers. This allows us to input images of various sizes.

  We're only using the first 16 layers of VGG-16. This includes several convolutional and max pooling layers, up to 'relu3_3'.

  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20020227.png)

- ## Training and Testing :-
  
  - epochs = 20
  - Batch size = 1
  - Optimizer = Adam
  - Learning rate = 0.001
  - Loss = MSE + Perceptual
    
    ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20020936.png)

    ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20021003.png)

- Training Set result :

  ![](https://github.com/Srishti002/FSRCNN/blob/main/label_0.png)
  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20225521.png)

  ![](https://github.com/Srishti002/FSRCNN/blob/main/label_2.png)
  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20230422.png)

  ![](https://github.com/Srishti002/FSRCNN/blob/main/label_3.png)
  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20230548.png)

  ![](https://github.com/Srishti002/FSRCNN/blob/main/label_4.png)
  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20230658.png)

  ![](https://github.com/Srishti002/FSRCNN/blob/main/label_5.png)
  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-13%20230817.png)

- Test Set Result :

  ![]()
  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-14%20010813.png)

  ![]()
  ![](https://github.com/Srishti002/FSRCNN/blob/main/Screenshot%202024-10-14%20010910.png)

  ![]()
  ![]()

  

  
  
