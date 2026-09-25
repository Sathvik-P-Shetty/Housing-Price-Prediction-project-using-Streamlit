import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ---------------------------------------
# Load Dataset
# ---------------------------------------

df = pd.read_csv("Housing.csv")

# ---------------------------------------
# Encode Categorical Columns
# ---------------------------------------

encoders = {}

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
pickle.dump(encoders, open("encoder.pkl","wb"))

# ---------------------------------------
# Split Dataset
# ---------------------------------------

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# Models
# ---------------------------------------

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
}

best_model = None
best_score = -999

print("="*60)

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred)**0.5

    print(f"\n{name}")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")

    if r2 > best_score:
        best_score = r2
        best_model = model

print("\nBest Model:", best_model.__class__.__name__)
print("Best R2 Score:", best_score)

pickle.dump(best_model, open("best_model.pkl","wb"))

print("\nModel Saved Successfully!")