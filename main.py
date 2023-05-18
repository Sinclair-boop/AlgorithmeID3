
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Charger les données à partir du fichier CSV
data = pd.read_csv('jouer.csv')

# Convertir les données textuelles en valeurs numériques
le = LabelEncoder()
data['temps'] = le.fit_transform(data['temps'])
data['température'] = le.fit_transform(data['température'])
data['humidité'] = le.fit_transform(data['humidité'])
data['vent'] = le.fit_transform(data['vent'])
data['jouer'] = le.fit_transform(data['jouer'])

# Séparer les caractéristiques (features) et les étiquettes (labels)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Créer un classifieur ID3
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# Prédire sur de nouvelles données
new_data = np.array([[2, 1, 0, 0]])  # Exemple de nouvelles données à prédire
prediction = clf.predict(new_data)
print(le.inverse_transform(prediction))

