import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas_profiling

from matplotlib import rcParams
import warnings


warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 10, 6
np.random.seed(42)

data = pd.read_csv('./data/pima_indians_diabetes.csv')
# Eight inputs: 
#   time_pregnant_no,
#   plasma_concentration,
#   diastolic_blood_pressure,
#   triceps_skinfold_thickness,
#   serum_insulin,
#   bmi,
#   diabetes_pedigree,
#   age
# One target:
#   class (0 or 1)
X = data.drop('class', axis=1)
Y = data['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, stratify=Y, test_size=0.10, random_state=42
)

classifier = RandomForestClassifier(n_estimators=100) # set number of trees
classifier.fit(X_train, Y_train) # train the model

Y_pred = classifier.predict(X_test)
print('Accuracy:', accuracy_score(Y_test, Y_pred))
