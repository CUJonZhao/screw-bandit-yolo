from ultralytics import YOLO
import os
import csv

model_path = r"C:\Users\PC\Desktop\model_screw3.0\weights\best.pt"
images_dir = r"C:\Users\PC\Desktop\screw3.0\dataset\images\train"
labels_dir = r"C:\Users\PC\Desktop\screw3.0\dataset\labels\train"

model = YOLO(model_path)

out_csv = "state.csv"
f = open(out_csv, "w", newline="")
writer = csv.writer(f)
writer.writerow(["filename", "gt", "pred", "TP", "FP", "FN", "avg_conf"])

for img_name in os.listdir(images_dir):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(images_dir, img_name)
    results = model.predict(img_path, save=False, verbose=False)[0]

    pred_boxes = results.boxes
    pred_count = len(pred_boxes)
    pred_confs = pred_boxes.conf.cpu().numpy().tolist() if pred_count > 0 else []
    avg_conf = sum(pred_confs)/len(pred_confs) if pred_confs else 0

    txt = img_name.rsplit(".", 1)[0] + ".txt"
    label_path = os.path.join(labels_dir, txt)
    gt_count = len(open(label_path).readlines()) if os.path.exists(label_path) else 0

    TP = min(gt_count, pred_count)
    FP = max(0, pred_count - gt_count)
    FN = max(0, gt_count - pred_count)

    writer.writerow([img_name, gt_count, pred_count, TP, FP, FN, avg_conf])

f.close()
print("done")
