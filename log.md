# 26 Septembre (2023):
## History:
- Created separate branch
- Created separate virtual enviornment
- Downloaded openVino
    - ![Intel Problem](logResources/0_IntelProblem.png)
    - ![openVino as virtual enviornment](logResources/1_openVino.png)

- Found out that openVino is only for inference (i.e. running, but not training)
    - (https://www.youtube.com/watch?v=kY9nZbX1DWM)
    - Hence useful for running the model given a certain file of weights, but not for training

- Found out that I can run Caffe based models without GPU SUPPORT (hence without CUDA). 
    - https://github.com/bytedeco/javacpp-presets/issues/219
    - ![No CUDA needed?](logResources/2_CudaProblem.png)
    - ![No CUDA needed 2?](logResources/2_CudaProblem2.png)
    - ![Getting Caffe to run only on a CPU](logResources/3_CaffeOnlyCPU.png)

## Ideas:
- Get a pre-trained model for showcasing purposes then start building upon it in phase 2
    - Need to start exploring it and looking at its source code in phase 1 so I could get 
        straight to work and enchancements in phase 2

## Closest goals/ views:
- download pre-trained model and set-it up to work on my mac -> see how good is it
