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
