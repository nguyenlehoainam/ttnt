import numpy as np

# e) Cấu trúc dữ liệu phù hợp để lưu: list of dict (mỗi dòng 1 record)
data = [
    {"stt": 1, "area": 30.0,     "price": 448.524},
    {"stt": 2, "area": 32.4138,  "price": 509.248},
    {"stt": 3, "area": 34.8276,  "price": 535.104},
    {"stt": 4, "area": 37.2414,  "price": 551.432},
    {"stt": 5, "area": 39.6552,  "price": 623.418},
    {"stt": 6, "area": 42.0690,  "price": 625.992},
    {"stt": 7, "area": 44.4828,  "price": 655.248},
    {"stt": 8, "area": 46.8966,  "price": 701.377},
    {"stt": 9, "area": 50.0,     "price": None},   # cần dự đoán
]

train = [d for d in data if d["price"] is not None]
x = np.array([d["area"] for d in train], dtype=float)
y = np.array([d["price"] for d in train], dtype=float)

# f) Hồi quy tuyến tính y = w1*x + w0 (công thức OLS)
x_mean, y_mean = x.mean(), y.mean()
w1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
w0 = y_mean - w1 * x_mean

# Dự đoán mẫu 9
x9 = next(d["area"] for d in data if d["stt"] == 9)
y9_pred = w1 * x9 + w0

print(f"Phương trình hồi quy: y = {w1:.6f} * x + {w0:.6f}")
print(f"Dự đoán giá cho mẫu 9 (diện tích = {x9}): {y9_pred:.3f}")
