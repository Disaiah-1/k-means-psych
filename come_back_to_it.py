import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("/Users/zainabnda-isaiah/Data visualization/human_cognitive_performance.csv")

# Show nulls and initial size
print(df.isnull().sum())

# Drop NaNs only in the columns you need
df = df.dropna(subset=["Age", "Caffeine_Intake", "Memory_Test_Score"])


# Extract and scale features
xx = df[["Age", "Caffeine_Intake", "Memory_Test_Score"]].values
scaler = MinMaxScaler(feature_range=(1, 10))
xx = scaler.fit_transform(xx)

X = xx[:, [0, 1]]

# Target: Memory_Test_Score (column 2)
Y = xx[:, 2]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.95)
print(X_train[:5])
print(Y_train[:5])

#Elbow method
wwcss = []
for i in range(1, 11):
    kmean = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmean.fit(X_train, Y_train)
    wwcss.append(kmean.inertia_)

# Plot
# plt.plot(range(1, 11), wwcss, marker='o')
# plt.title("Elbow Graph")
# plt.xlabel("Number of Clusters")
# plt.ylabel("WCSS")
# plt.grid(True)
# plt.show()

#Last elbow on the graph landed on numer 5, so 5 is the optiman number of clusters. Making the clustering algorithm.
#When initializing centroids, make sure you assign them to the same name as in the elbow method to override their original number which is 10. If not you will have 10 centroids in your final output.
kmean = KMeans(n_clusters=20, init='k-means++', random_state=0)
ymodel = kmean.fit_predict(X_train)

# Create a new DataFrame for plotting
train_df = pd.DataFrame(X_train, columns=["Age", "Caffeine_Intake"])
train_df["Memory_Test_Score"] = Y_train
train_df["label3"] = ymodel

# Create 3D scatter plot
trace1 = go.Scatter3d(
    x=train_df['Age'],
    y=train_df['Caffeine_Intake'],
    z=train_df['Memory_Test_Score'],
    mode='markers',
    marker=dict(
        color=train_df['label3'],
        size=5,
        opacity=0.8,
        line=dict(width=0.5)
    )
)

data = [trace1]
layout = go.Layout(
    title='Clusters (Training Data)',
    scene=dict(
        xaxis=dict(title='Caffeine Intake'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Memory Test Score')
    )
)
fig = go.Figure(data=data, layout=layout)
fig.show()

centroids = kmean.cluster_centers_

trace2 = go.Scatter3d(
    x=centroids[:, 0],  # Age
    y=centroids[:, 1],  # Caffeine Intake
    z=[np.mean(Y_train)] * len(centroids),  # Use average memory score for Z position (or compute properly if needed)
    mode='markers',
    marker=dict(
        color='red',
        size=7,
        symbol='x',
        opacity=1.0
    ),
    name='Centroids'
)

# Plot
fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()