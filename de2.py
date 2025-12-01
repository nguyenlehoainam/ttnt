import numpy as np

# c) Cấu trúc dữ liệu phù hợp: list of dict (mỗi dòng là 1 record)
data = [
    {"stt": 1, "x1": 1, "x2": 1, "t": 0},   # 0: Xanh
    {"stt": 2, "x1": 8, "x2": 3, "t": 0},   # 0: Xanh
    {"stt": 3, "x1": 2, "x2": 7, "t": 1},   # 1: Chín
    {"stt": 4, "x1": 8, "x2": 8, "t": 1},   # 1: Chín
    {"stt": 5, "x1": 9, "x2": 9, "t": 1},   # 1: Chín
    {"stt": 6, "x1": 9, "x2": 2, "t": None} # cần dự đoán
]

train = [d for d in data if d["t"] is not None]
sample6 = next(d for d in data if d["stt"] == 6)

X = np.array([[d["x1"], d["x2"]] for d in train], dtype=float)
y = np.array([d["t"] for d in train], dtype=int)
x6 = np.array([sample6["x1"], sample6["x2"]], dtype=float)

# (khuyến nghị) chuẩn hoá Min-Max để học ổn định
xmin, xmax = X.min(axis=0), X.max(axis=0)
Xn = (X - xmin) / (xmax - xmin)
x6n = (x6 - xmin) / (xmax - xmin)

class Perceptron:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def step(self, z):
        return 1 if z >= 0 else 0  # hàm ngưỡng

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1], dtype=float)
        self.b = 0.0

        for epoch in range(1, self.epochs + 1):
            errors = 0
            for xi, ti in zip(X, y):
                pred = self.step(self.w @ xi + self.b)
                update = self.lr * (ti - pred)
                if update != 0:
                    self.w += update * xi
                    self.b += update
                    errors += 1
            if errors == 0:  # hội tụ
                return epoch
        return self.epochs

    def predict_one(self, x):
        return self.step(self.w @ x + self.b)

model = Perceptron(lr=0.1, epochs=1000)
used_epochs = model.fit(Xn, y)
pred6 = model.predict_one(x6n)

label_text = "0 (Xanh)" if pred6 == 0 else "1 (Chín)"

print("Epochs used:", used_epochs)
print("w =", model.w, "b =", model.b)
print(f"Mẫu 6 (x1={x6[0]}, x2={x6[1]}) => dự đoán lớp:", label_text)
