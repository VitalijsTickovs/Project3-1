### Plan
- Start simple:
    - very simple autoencoder baseline model:
        - 3 inputs:
            1. left hand coordinates
            2. right hand coordinates
            3. center of object frame coordinates
        - doesn't take into account object type
        - can't extend the model weight matrices to add a new object
        - output:
            - predicted movement sequence 
    - simple assumptions:
        1. single object (extension to multi needed)
        2. single human (extension to multi needed)

### Log
##### 19 Nov 2023
- Decided to use PyTorch instead of TensorFlow because is supposedly easier to debug and less steep learning curve

##### 22 Nov 2023
Progress:
- May actually be possible to use intel extnesion (simple changes to the optimizer type in code the rest of the changes remain the same)
    - Nevermind it is for linux and windows OS
    - No options, left with CPUs
- Recomended 2 layers for encoders and decoders
- Multiple inputs only needed when multiple different data sources (e.g. visual and audio)
    - Solution 1: multiple input layers + pooling
    - Solution 2: flatten + CNN

- Work with group:
    - json skeleton dataset
    - more annotations for YOLO
- Work alone:
    - Prepared pytorch based autoencoder trainign and test methods
    - Explored data:
        - Need to predict the movement in the future; options:
            1. Predict based on 4 seconds, but look only 1 second forward
            2. Predict based on 4 seconds interval; 4 seconds forward
                - if ran out of data fill up the space with the same frame (i.e. terminal grame)
                - if not countinue 
            - conclusions: 
                - don't like 2.  
                - option 1. reminds me of online learning
        - can leave "none" as test set (although questionable because high quality data)