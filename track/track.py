from ultralytics import YOLO

# define necessary methods
def runTrack(option, pathToFile=None):
    # running inference on file
    if option == 1:
        results = model.track(source=pathToFile, show=True, tracker="bytetrack.yaml")
    # running inference on video stream
    else: # option == 1:
        results = model.track(source=0, show=True, tracker="bytetrack.yaml", conf=0.2)


# loading pre-trained model saved at pt
model = YOLO('track/Weights/last.pt')

runTrack(0)

