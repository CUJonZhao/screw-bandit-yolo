import pandas as pd

df = pd.read_csv("state.csv")

total_images = len(df)
total_gt = df["gt"].sum()
total_pred = df["pred"].sum()
total_tp = df["TP"].sum()
total_fp = df["FP"].sum()
total_fn = df["FN"].sum()
avg_conf = df["avg_conf"].mean()

fn_per_image = total_fn / total_images if total_images > 0 else 0
fp_per_image = total_fp / total_images if total_images > 0 else 0
fn_rate = total_fn / total_gt if total_gt > 0 else 0
fp_rate = total_fp / total_pred if total_pred > 0 else 0

summary = pd.DataFrame({
    "total_images": [total_images],
    "total_gt": [total_gt],
    "total_pred": [total_pred],
    "total_TP": [total_tp],
    "total_FP": [total_fp],
    "total_FN": [total_fn],
    "avg_conf": [avg_conf],
    "FN_per_image": [fn_per_image],
    "FP_per_image": [fp_per_image],
    "FN_rate_over_GT": [fn_rate],
    "FP_rate_over_pred": [fp_rate]
})

summary.to_csv("error_summary.csv", index=False)

with open("error_overview.txt", "w") as f:
    f.write("TOTAL IMAGES: " + str(total_images) + "\n")
    f.write("TOTAL GT: " + str(total_gt) + "\n")
    f.write("TOTAL PRED: " + str(total_pred) + "\n")
    f.write("TOTAL TP: " + str(total_tp) + "\n")
    f.write("TOTAL FP: " + str(total_fp) + "\n")
    f.write("TOTAL FN: " + str(total_fn) + "\n")
    f.write("AVG CONF: " + str(avg_conf) + "\n")
    f.write("FN PER IMAGE: " + str(fn_per_image) + "\n")
    f.write("FP PER IMAGE: " + str(fp_per_image) + "\n")
    f.write("FN RATE OVER GT: " + str(fn_rate) + "\n")
    f.write("FP RATE OVER PRED: " + str(fp_rate) + "\n")

print("done")
