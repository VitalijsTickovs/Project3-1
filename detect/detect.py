import glob
import cv2 as cv
from ultralytics import YOLO


def predictAndVisualise(arr):
    # get the frames
    idx = 0
    frame = cv.imread(arr[idx])

    # Load a model
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(arr)  # return a list of Results objects

    # Process results list
    idx = 0
    for result in results:
        # print the arrays of results from result object
        print(arr[idx], ":")
        print(result.boxes.xywh)
        print()

        # draw the rectangle boxes to verify that I understna dthe formatting
        for xyxy in result.boxes.xyxy:
            cv.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,0,255), 2)
        idx+=1
    return frame

def predictAndCenter(arr):
    # get the frames
    idx = 0
    frame = cv.imread(arr[idx])

    # Load a model
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(arr)  # return a list of Results objects

    # Process results list
    idx = 0
    for result in results:
        # print the arrays of results from result object
        print(arr[idx], ":")
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

def predictSingle(path:str):
    img = cv.imread(path)
    cv.imshow(path, img)
    print("\n")
    cv.waitKey(0)
    cv.destroyAllWindows()

frame = predictAndCenter(["detect/bus.jpg"])
cv.imshow("frame", frame)
cv.waitKey(0) 
cv.destroyAllWindows()