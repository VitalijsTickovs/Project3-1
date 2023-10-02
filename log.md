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

- Starting to follow the instructions to download the MobilenetSSD model (https://github.com/chuanqi305/MobileNet-SSD)
    - I will need to request access to the weights
    - Installed SSD repository 
    - Stuck on installing Caffe
        - Looks like I caqn try to install it with anaconda? I don't wanna use brew. 
        - I am not sure whether it will cause any issues because it differs from standard installation, but I will just have to try.
            - ![Caffe installable via anaconda?](logResources/4_CaffeAnaconda.png)
            - According to digital assitant I might not be able to modify files as I want (e.g. enable only CPU) so it recommends downloading from source

## Ideas:
- Get a pre-trained model for showcasing purposes then start building upon it in phase 2
    - Need to start exploring it and looking at its source code in phase 1 so I could get 
        straight to work and enchancements in phase 2

## Closest goals/ views:
- download pre-trained model and set-it up to work on my mac -> see how good is it

# 1 Octobre (2023):
## History
1. Tried to install Caffe via anaconda only to find out that there is no package avilable for osx system. Try linux? 
2. Decided not to try linux version
3. Trying to install via official instructions and homebrew
    - let hdf5 be on anaconda python
    - installed all required brew libraries except for szip (couldn't find it on brew, assume part of anacondas hdf5)
    - installed command line tools xcode
    - installed 2 additional libraries:
        - command "--with-python" became obsolete now it is in python by default
        - ![alternative to "--with-python"](logResources/5_withPython.png)
        - No boost-python available only boost-python3
    - do the checks using otool from https://gist.github.com/kylemcdonald/0698c7749e483cd43a0e 
    - go to caffe makefile.config and modify it to use CPU and anaconda3 path
4. Tried to "make all". Ran into errors with C++ version of protoc. Possibly because of conflicting anaconda and brew? (https://github.com/BVLC/caffe/issues/6527)
5. Tried to ignore brew and go via anaconda:
    - change the PATH and CPATH variables to anaconda base bin and include directories
    - Also changed the protoc version headers to C++11 
        - ![Previous head parameters](logResources/6_headOld.png)
        - ![Current head parameters](logResources/6_headNew.png)
    - had to instal glog 
    - had to install opencv
6. I next error occurs then I am
7. To many conflicts and too slow via conda installation so going back to brew

## Useful sources:
- Another caffe installation guide (https://gist.github.com/kylemcdonald/0698c7749e483cd43a0e)

## Objectives:
- Try to get back to brew and exclude conda
    - change PATH and CPATH variables to brew bin and brew include?
    - change back the flags to normal?
    - install python-boost3?


### 2 Octobre (2023)
## History:
1. Trying to comeback to brew
    - bin is folder with all the binary executables allowing to execute certain commands and programs on computer
    - brew install binaries to local bin direcotries on computer
    - there are multiple bin directories on computer
    - to see them in finder press: cmd + shift + .
2. How to change $PATH variable:
    - https://stackoverflow.com/questions/15872666/how-to-remove-entry-from-path-on-mac
3. Change the CXXFLAGS in makefile to -std=c++14
4. Conclusions:
    - the repository is written in old c++ version
    - potetntially need to downgrade all the brew packages especially protobuf
5. Was able to resolve error by removing the anaconda protoc and protobuf 
    - anaconda was conflicting with brew when runnign "make all"
    - ![Thanks to chinese guys and chatgpt "which -a protoc" command](logResources/5_withPython.png)
6. Uninstalled opencv and protobuff in preparation for installing the downgraded versions
7. Installed protobuf@3 via brew which is more suitable for the release of the caffe repository
    - NOW I HAVE ONE ERROR LESS!!!
8. Problem is that now I am missing opencv2 while current opencv is v4 and overrieds my protobuf@3 with newer protobuf version
9. Tired to use opencv with protobuf and protobuf@3
    - unlinked protobuf and linked protobuf@3 so no protoc errors
    - still doesn't like opencv -> i.e. requests opencv2
10. Tired to install opencv2 via brew by editing .rb file but failed because of sha1 being depreciated (https://docs.brew.sh/Checksum_Deprecation)
    -  thanks to this guy for explanations ![Brew old package installation](logResources/8_brewOldPackages.png)



## Objectives:
- Reinstall and downgrade protobuf version??? -> done, using protobuf@3 and it is working
- manually try to add opencv@2?
