from ultralytics import YOLO

# define necessary methods
def runTrack(option, pathToFile=None):
    # running inference on file
    if option == 1:
        results = model.track(source=pathToFile, show=True, tracker="bytetrack.yaml", conf=0.1)
    # running inference on video stream
    else: # option == 1:
        results = model.track(source=0, show=True, tracker="bytetrack.yaml", conf=0.1)


# loading pre-trained model saved at pt
model = YOLO('Models/box_training_with_string/best.pt')

# Best weights:
# alpaca1: 'Models/box_training_with_string/best.pt'
# 

runTrack(1, pathToFile="local_env/videos/box_short.mov")

