from ultralytics import YOLO

model = YOLO(r"C:\Users\PC\Desktop\screw3.0\weights\best.pt")

results = model.predict(
    source=[
        r"C:\Users\PC\Desktop\screw3.0\apply\test_screw.jpg"
    ],
    imgsz=640,
    conf=0.5,
    device='cpu',
    save=True,
    save_txt=True,
    save_conf=True
)
