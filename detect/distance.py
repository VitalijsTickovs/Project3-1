import cv2 as cv
from ultralytics import YOLO

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
    print("midPoints: ", midPoints)
    counter = 0
    for xyxy in xyxys:
        midX = int((xyxy[0]+xyxy[2])/2)
        midY = int((xyxy[1]+xyxy[3])/2)
        midPoints[counter][0] = midX
        midPoints[counter][1] = midY
        counter+=1
    print("midPoints filled: ", midPoints)
    checkMidPoints(visualise, midPoints, path)
    return midPoints


def checkMidPoints(bool, midpoints, path):
    """Check where  the midpoints are showing up to see if they were calculated correctly"""
    if bool:
        frame = cv.imread(path)
        for arr in midpoints:
            frame = cv.circle(frame, (arr[0], arr[1]), radius=5, color=(0,0,255), thickness=-1)
        cv.imshow("midpoints frame", frame)
        cv.waitKey(0) 
        cv.destroyAllWindows()


## Executable code:
computeMidpoints("detect/bus.jpg")
