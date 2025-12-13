from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data=r"C:\Users\PC\Desktop\screw3.0\dataset\data.yaml",
    epochs=50,
    imgsz=640,
    device="cpu"
)
