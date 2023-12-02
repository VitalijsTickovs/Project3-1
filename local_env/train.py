if __name__ == '__main__':
    # Your main code here
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolov8n.yaml")
    
    # Use the model
    results = model.train(data="local_env\data.yaml", epochs=200, imgsz=640, patience=20, device='0')  # train the model
