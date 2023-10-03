import glob
import cv2 as cv
from ultralytics import YOLO


def predictMultiple(arr):
    # Load a model
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(arr)  # return a list of Results objects

    # Process results list
    for result in results:
        print(result.boxes)  # Boxes object for bbox outputs

def predictSingle(path:str):
    img = cv.imread(path)
    cv.imshow(path, img)
    print("\n")
    cv.waitKey(0)
    cv.destroyAllWindows()

predictMultiple(["bus.jpg"])