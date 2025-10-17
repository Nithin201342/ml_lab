import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/jiss-sngce/CO_3/main/jkcars.csv')
print(df.head())
print(df.shape, "\n", df.info())

# Create a new DataFrame consisting of three columns 'Volume', 'Weight', 'CO2'.
new_data = df[['Volume', 'Weight', 'CO2']]

sil_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
    kmeans.fit(new_data)
    score = silhouette_score(new_data, kmeans.labels_)
    sil_scores.append(score)
    
silhouette_df = pd.DataFrame({'K': range(2, 11), 'Silhouette_score': sil_scores})
print(silhouette_df)

plt.xlabel("Number of clusters")
plt.ylabel("Silhoutte score")
plt.plot(silhouette_df['K'], silhouette_df['Silhouette_score'], marker='o')
plt.grid()
plt.show()