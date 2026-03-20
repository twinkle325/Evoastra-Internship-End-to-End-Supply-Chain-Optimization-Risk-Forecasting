
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ================= LOAD DATA =================
df = pd.read_csv("DataCoSupplyChain_clean.csv", encoding="cp1252")

print("Shape:", df.shape)

# ================= REMOVE LEAKAGE =================
df = df.drop(columns=[
    "Delivery Status",
    "Customer Email",
    "Customer Password",
    "Product Image",
    "Customer Fname",
    "Customer Lname"
], errors='ignore')

# ================= FEATURE ENGINEERING =================

# 1. Shipping delay (MOST IMPORTANT)
df["shipping_delay"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]

# 2. Discount flag
df["is_discounted"] = (df["Order Item Discount Rate"] > 0).astype(int)

# 3. High value order
df["high_value_order"] = (df["Sales"] > df["Sales"].median()).astype(int)

# ================= ENCODING =================
cat_cols = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Customer Segment",
    "Category Name",
    "Type"
]

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# ================= FEATURES =================
features = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "shipping_delay",
    "Order Item Discount Rate",
    "Order Item Quantity",
    "Sales",
    "Benefit per order",
    "is_discounted",
    "high_value_order",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Customer Segment",
    "Category Name",
    "Type"
]

X = df[features]
y = df["Late_delivery_risk"]

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# Save dataset
df[features + ["Late_delivery_risk"]].to_csv("dataco_features.csv", index=False)