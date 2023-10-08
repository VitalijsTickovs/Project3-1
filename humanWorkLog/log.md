# Log of all of my activity in the repository (history)
## 3 Octobre 2023
### History
1. After leaving mobilenetSSD focused on YOLO
2. Set up YOLO by following the instructions
3. Able to train it although slow even with only 3 epochs
4. Gave up on trying to convert result object into something visualisable

### Close objectives
1. Look at tracking ability and whether it is possible to extract the data in real time


## 8 Octobre 2023
1. Created code for tracking the objects both from video and in real-time 
2. Looks like tracking with .mov file types is very slow (because able to track in real-time much faster when source=0)
3. Need to explore further what are different optional tracking algorithms such as "bytetrack". Looks like it comes from other YOLO models such as YOLOX. 
4. IDEA: Adressing the limitation of camera not being able to see everything (limited view):
    - memorising and using inference
        - example:
            - if A to B is 5 cm
            - and B to C is 4 cm
            - then A to C is 9 cm
5. Started thinking about how to store the distances between the objects:
    - something like a graph, but needs to be computatinally efficient to iterate through
        - maybe start with simple arrays instead of creating a whole graph structure
    - Distances between objects on an image might not be representative of the real-world distances. How to deal with this?
        - 2 options come to mind:
            - use the scale data (e.g. how small is the object in comparison to its known size -> use pre-exisiting knowldge) 
            - use depth camera
6. Trying to understand xyxy  and xywh formates of result object from YOLO
    - do not understna dthe xywh format (unable to create correct bounding boxes)
    - successfuly creation of boxes when using xyxy format
