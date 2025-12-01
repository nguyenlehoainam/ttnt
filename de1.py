import numpy as np

# a) Cấu trúc dữ liệu lưu trữ phù hợp: list[dict]
data = [
    {"stt": 1, "x1": 2,  "x2": 1, "t": 0},     # Từ chối
    {"stt": 2, "x1": 3,  "x2": 4, "t": 0},     # Từ chối
    {"stt": 3, "x1": 10, "x2": 2, "t": 1},     # Duyệt
    {"stt": 4, "x1": 12, "x2": 8, "t": 1},     # Duyệt
    {"stt": 5, "x1": 4,  "x2": 8, "t": 1},     # Duyệt
    {"stt": 6, "x1": 8,  "x2": 3, "t": None},  # Cần dự báo
]

# Tách train và mẫu cần dự báo
train = [d for d in data if d["t"] is not None]
sample6 = next(d for d in data if d["stt"] == 6)

# Chuyển sang ma trận X (n_samples, n_features) và vector y
X = np.array([[d["x1"], d["x2"]] for d in train], dtype=float)
y = np.array([d["t"] for d in train], dtype=int)
x6 = np.array([sample6["x1"], sample6["x2"]], dtype=float)

# (Khuyến nghị) Chuẩn hoá Min-Max để học ổn định hơn
xmin, xmax = X.min(axis=0), X.max(axis=0)
Xn = (X - xmin) / (xmax - xmin)
x6n = (x6 - xmin) / (xmax - xmin)

class Perceptron:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def _step(self, z: float) -> int:
        return 1 if z >= 0 else 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        for epoch in range(1, self.epochs + 1):
            errors = 0
            for xi, ti in zip(X, y):
                pred = self._step(self.w @ xi + self.b)
                update = self.lr * (ti - pred)
                if update != 0:
                    self.w += update * xi
                    self.b += update
                    errors += 1
            if errors == 0:
                return epoch  # hội tụ sớm
        return self.epochs

    def predict_one(self, x: np.ndarray) -> int:
        return self._step(self.w @ x + self.b)

# Huấn luyện và dự báo
model = Perceptron(lr=0.1, epochs=1000)
used_epochs = model.fit(Xn, y)
pred6 = model.predict_one(x6n)

meaning = "Duyệt (Uy tín)" if pred6 == 1 else "Từ chối (Rủi ro cao)"

print("Perceptron trained.")
print(f"Epochs used: {used_epochs}")
print(f"Weights w: {model.w}, bias b: {model.b:.4f}")
print(f"Sample 6: x1={x6[0]}, x2={x6[1]}  =>  predicted t={pred6}  =>  {meaning}")
