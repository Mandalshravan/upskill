import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("C:/Users/MANDAL/Desktop/upskill/datafile (1).csv")

# Strip any leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Encode categorical variables
le_crop = LabelEncoder()
le_state = LabelEncoder()
df["Crop"] = le_crop.fit_transform(df["Crop"])
df["State"] = le_state.fit_transform(df["State"])

# Features and target
X = df.drop("Yield (Quintal/ Hectare)", axis=1)
y = df["Yield (Quintal/ Hectare)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" Mean Squared Error: {mse:.2f}")
print(f" R^2 Score: {r2:.2f}")
