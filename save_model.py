import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv("Crop_recommendation.csv")

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X = df.drop('label', axis=1).values
y = df['label'].values

pipeline = make_pipeline(StandardScaler(), GaussianNB())
pipeline.fit(X, y)

artifact = {"pipeline": pipeline, "label_encoder": le}
with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("model.pkl saved successfully!")
