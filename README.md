# FSRCNN
## Introduction :-
Though SRCNN is already faster than most previous learning-based methods, the processing speed on large images is still unsatisfactory. For example, to upsample an 240×240 image by a factor of 3, the speed of the original SRCNN is about 1.32 fps, which is far from real time (24 fps). 

There are two inherent limitations of SRCNN that restricts its runnnign speed :-
1. Pre-processing step, the original LR image needs to be upsampled to the desired size using bicubic interpolation to form the input. Thus the computation complexity of SRCNN grows quadratically with the spatial size of the HR image. For the upscaling factor n, the computational cost of convolution with the interpolated LR image will be n^2 times of that for the original LR one.
   
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
  
  
