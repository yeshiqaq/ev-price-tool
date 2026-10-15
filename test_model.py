import joblib

model = joblib.load(
    "models/price_model_v1.joblib"
)

print("Model loaded successfully!")
print("Model type:", type(model))

if hasattr(model, "feature_names_in_"):
    print("Input features:")
    print(model.feature_names_in_)