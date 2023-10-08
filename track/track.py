from ultralytics import YOLO

# define necessary methods
def runTrack(option, pathToFile=None):
    # running inference on file
    if option == 1:
        results = model.track(source=pathToFile, show=True, tracker="bytetrack.yaml")
    # running inference on video stream
    else: # option == 1:
        results = model.track(source=0, show=True, tracker="bytetrack.yaml")

# loading pre-trained model saved at pt
model = YOLO('yolov8n.pt')

runTrack(1, "track/fish.mov")

