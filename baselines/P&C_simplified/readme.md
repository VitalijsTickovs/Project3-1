# Guide on which files are used where
### 1) Basic autoencoder
#### Main files
- main
    - file for running autoencoder model
- dataset
    - get the skeleton data in a X and Y array formats to feed into a network and do some pre-processing
- model 
    - contains code of autoencoder model based on pytorch 
#### Additional files
- visualData
    - a helper ipynb file allowing to visualise some of the mechanisms in "dataset"
- visualModel 
    - a helper ipynb file allowing to visualise some of the mechanisms in "model"
- util
    - testing different methods for "model" (e.g. training procedure, testing procedure etc.)

### 2) Other
