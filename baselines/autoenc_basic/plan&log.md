### Plan
- Start simple:
    - very simple autoencoder baseline model:
        - 3 inputs:
            1. left hand coordinates
            2. right hand coordinates
            3. center of object frame coordinates
        - doesn't take into accoutn object type
        - can't extend the model weight matrices to add a new object
        - output:
            - predicted movement sequence 
    - simple assumptions:
        1. single object
        2. single human 

### Log
##### 19 Nov 2023
- Decided to use PyTorch instead of TensorFlow because is supposedly easier to debug and less steep learning curve