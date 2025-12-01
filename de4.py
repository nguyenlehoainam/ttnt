import math
from collections import Counter, defaultdict

# g) Cấu trúc dữ liệu phù hợp: list of dict (mỗi dòng 1 record)
data = [
    {"TT": 1, "Income": "Cao",  "Credit": "Tốt", "Collateral": "Có",    "Decision": "Yes"},
    {"TT": 2, "Income": "Cao",  "Credit": "Xấu", "Collateral": "Có",    "Decision": "Yes"},
    {"TT": 3, "Income": "Thấp", "Credit": "Tốt", "Collateral": "Không", "Decision": "No"},
    {"TT": 4, "Income": "TB",   "Credit": "Tốt", "Collateral": "Có",    "Decision": "Yes"},
    {"TT": 5, "Income": "Thấp", "Credit": "Xấu", "Collateral": "Không", "Decision": "No"},
    {"TT": 6, "Income": "TB",   "Credit": "Xấu", "Collateral": "Có",    "Decision": "No"},
    {"TT": 7, "Income": "Cao",  "Credit": "Tốt", "Collateral": "Không", "Decision": "Yes"},
    {"TT": 8, "Income": "TB",   "Credit": "Tốt", "Collateral": "Không", "Decision": "Yes"},
]

sample9 = {"TT": 9, "Income": "Thấp", "Credit": "Tốt", "Collateral": "Có"}  # cần dự đoán


def entropy(rows, label_key="Decision"):
    counts = Counter(r[label_key] for r in rows)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def majority_label(rows, label_key="Decision"):
    return Counter(r[label_key] for r in rows).most_common(1)[0][0]


def information_gain(rows, attr, label_key="Decision"):
    base = entropy(rows, label_key)
    groups = defaultdict(list)
    for r in rows:
        groups[r[attr]].append(r)

    total = len(rows)
    remainder = 0.0
    for g in groups.values():
        remainder += (len(g) / total) * entropy(g, label_key)

    return base - remainder


def id3(rows, attrs, label_key="Decision"):
    labels = [r[label_key] for r in rows]
    # Nếu tất cả cùng nhãn -> lá
    if len(set(labels)) == 1:
        return labels[0]

    # Nếu hết thuộc tính -> trả về nhãn đa số
    if not attrs:
        return majority_label(rows, label_key)

    # Chọn thuộc tính có gain lớn nhất
    best_attr = max(attrs, key=lambda a: information_gain(rows, a, label_key))
    node = {
        "attr": best_attr,
        "default": majority_label(rows, label_key),
        "children": {}
    }

    # Chia nhánh theo giá trị của thuộc tính best_attr
    groups = defaultdict(list)
    for r in rows:
        groups[r[best_attr]].append(r)

    remaining_attrs = [a for a in attrs if a != best_attr]
    for val, subset in groups.items():
        node["children"][val] = id3(subset, remaining_attrs, label_key)

    return node


def predict(tree, x):
    # tree có thể là nhãn (str) hoặc node (dict)
    while isinstance(tree, dict):
        attr = tree["attr"]
        val = x.get(attr)
        if val in tree["children"]:
            tree = tree["children"][val]
        else:
            return tree["default"]  # gặp giá trị chưa có trong train
    return tree


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "=>", tree)
        return
    attr = tree["attr"]
    print(indent + f"[{attr}] (default={tree['default']})")
    for val, child in tree["children"].items():
        print(indent + f"  - {attr} = {val}:")
        print_tree(child, indent + "    ")


# h) Huấn luyện ID3 và dự đoán mẫu 9
attrs = ["Income", "Credit", "Collateral"]
tree = id3(data, attrs)

print("=== Cây ID3 học được ===")
print_tree(tree)

pred9 = predict(tree, sample9)
print("\n=== Dự đoán mẫu 9 ===")
print("Mẫu 9:", sample9)
print("=> Decision =", pred9)
