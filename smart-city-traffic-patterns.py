import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar  # You can replace this with Indian calendar if needed

# Load test dataset
df = pd.read_csv("C:/Users/MANDAL/Desktop/upskill/test_BdBKkAj.csv")

# Convert DateTime to datetime object
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract features from test data
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Hour'] = df['DateTime'].dt.hour
df['Weekday'] = df['DateTime'].dt.weekday
df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)

# Add holiday flag
cal = calendar()
holidays = cal.holidays(start=df['DateTime'].min(), end=df['DateTime'].max())
df['Is_Holiday'] = df['DateTime'].dt.normalize().isin(holidays).astype(int)

print("Test Data Preview:")
print(df.head())

# ✅ LOAD TRAINING DATA (missing before)
train_df = pd.read_csv("C:/Users/MANDAL/Desktop/upskill/train_aWnotuB.csv")

# Process training data
train_df['DateTime'] = pd.to_datetime(train_df['DateTime'])
train_df['Hour'] = train_df['DateTime'].dt.hour

# Plot for Junction 1
junction_1 = train_df[train_df['Junction'] == 1]
junction_1.groupby('Hour')['Vehicles'].mean().plot(kind='line', title='Average Vehicles by Hour - Junction 1')
plt.xlabel('Hour of Day')
plt.ylabel('Average Vehicles')
plt.grid(True)
plt.show()
