if __name__ == '__main__':
    # Your main code here
    from ultralytics import YOLO

    # Load a model
    model = YOLO("best.pt")
    
    # Use the model
    results = model.train(data="C:\GitHub\Project3-1\local_env\data.yaml", epochs=300, imgsz=640, device='0')  # train the model
