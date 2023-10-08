import cv2 as cv
from ultralytics import YOLO
import math

def predictAndCenter(path):
    """Method for calculating distances between midpoints"""
    # get the frames
    frame = cv.imread(path)

    # Load a model
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(path)  # return a list of Results objects

    # Process results list
    idx = 0
    for result in results:
        # print the arrays of results from result object
        print(path, ":")
        print(result.boxes.xyxy)
        print()

        # draw the midpoint and the boxes
        for xyxy in result.boxes.xyxy:
            frame = cv.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,0,255), 2)
            midX = int((xyxy[0]+xyxy[2])/2)
            midY = int((xyxy[1]+xyxy[3])/2)
            frame = cv.circle(frame, (midX, midY), radius=5, color=(0,0,255), thickness=-1)
        idx+=1
    return frame

def computeMidpoints(path, visualise=False):
    # Load a model
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

    # Run inference on an images
    results = model(path)  # return a list of results

    # get the xyxy
    xyxys = results[0].boxes.xyxy # there is only one element because we are inputting only 
                                    # one image into model inference

    # compute midpoints and store them
    midPoints = [ [0]*2 for i in range(len(xyxys))]
    if (visualise): print("midPoints: ", midPoints)
    counter = 0
    for xyxy in xyxys:
        midX = int((xyxy[0]+xyxy[2])/2)
        midY = int((xyxy[1]+xyxy[3])/2)
        midPoints[counter][0] = midX
        midPoints[counter][1] = midY
        counter+=1
    if (visualise): print("midPoints filled: ", midPoints)
    checkMidPoints(visualise, midPoints, path)
    return results, midPoints


def checkMidPoints(bool, midpoints, path):
    """Check where the midpoints are showing up to see if they were calculated correctly"""
    if bool:
        frame = cv.imread(path)
        for arr in midpoints:
            frame = cv.circle(frame, (arr[0], arr[1]), radius=5, color=(0,0,255), thickness=-1)
        cv.imshow("midpoints frame", frame)
        cv.waitKey(0) 
        cv.destroyAllWindows()

def computeDistances(path, visualise=False):
    """Compute distances between midpoints of the detected objects"""
    # get the midpoints
    results, midPoints = computeMidpoints(path)

    # create array to store them
    distances = [ [0.0]*len(midPoints) for i in range(len(midPoints))]
    if (visualise): 
        print("distances:", distances)
        print()
    
    # iterate through detected objects and compute distances
    idxA = 0
    for pointA in midPoints:
        idxB = 0
        for pointB in midPoints:
            distance = math.dist(pointA, pointB)   
            distances[idxA][idxB] = distance
            idxB+=1
        idxA+=1

    if (visualise): print("distances filled:", distances)
    return results, distances

def getNameLookup(distances, visualise=False):
    """generate arrays so that we can look up a  name of the object e.g. distance value in cell (1,3) is a distance between 
    nameRow[1] and nameCol[3]"""
    # init. empty lookup arrays
    namesRow = ["..."]*len(distances)
    namesCol = ["..."]*len(distances)

    # assign template names
    for i in range(len(distances)):
        namesRow[i] = "entity"+str(i)
        namesCol[i] = "entity"+str(i)

    if(visualise):
        print("namesRow", namesRow)
        print()
        print("namesCol", namesCol)

    # return lookup arrays
    return namesRow, namesCol

def getTypeLookup(results, namesRow, visualise=False):
    """Generate array for looking up entity types"""
    # create empty array for type lookup [[class_id, class_name], [class_id, class_name], ...]
    typeNames = [[0]*2 for i in range(len(namesRow))]
    typeClsIdx = 

    # fill the array with class names and class ids corresponding to namesRow index (which is the same as namesCol)
    counter = 0
    for idxCls in results[0].boxes.cls:
        className = results[0].names[int(idxCls)]
        types[counter][0] = int(idxCls)
        types[counter][1] = className

        counter+=1

    if (visualise): print(types)
    return types


## Executable code:
results, distances = computeDistances("detect/bus.jpg")
namesRow, namesCol = getNameLookup(distances)
types = getTypeLookup(results, namesRow)