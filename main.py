import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Charger les données à partir du dataset diamonds.csv
df = pd.read_csv('diamonds.csv')
print("Dataset loaded successfully.")


#Afficher les données.
print(df.head())

# Afficher le shape des données
print(df.shape)

# Afficher les informations sur le dataset
print(df.info())

# Afficher les colonnes et supprimer les colonnes insignifiantes si elles existent.
print(df.columns)
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    print("Colonne supprimé")
else:
    print("Pas de colonnes insignifiantes")

# Chercher et traiter les valeurs dupliquées.
duplicates = df.duplicated().sum()
print(f"Nombre de doublons : {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Nouveau shape : {df.shape}")

# Regarder la distribution des données à l’aide d’un histogramme.
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.savefig('distribution_histogram.png')

# Chercher et traiter les valeurs manquantes
missing_values = df.isnull().sum()
print(missing_values)
if missing_values.sum() > 0:
    print("Traitement des valeurs manquantes requis (aucune detectée dans le dataset standard diamonds).")
else:
    print("Aucune valeur manquante.")



#Afficher les valeurs aberrantes à l’aide d’un boxplot.
plt.figure(figsize=(15, 8))
sns.boxplot(data=df.select_dtypes(include=np.number))
plt.xticks(rotation=45)
plt.title("Boxplot des variables numériques")
plt.savefig('outliers_boxplot.png')
print("Boxplot sauvegardé sous 'outliers_boxplot.png'")

#Quantifier les valeurs aberrantes
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Colonne {col}: {len(outliers)} ")

# Gérer les valeurs aberrantes pour l’attribut price
Q1_price = df['price'].quantile(0.25)
Q3_price = df['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price

original_shape = df.shape
df_cleaned = df[(df['price'] >= lower_bound_price) & (df['price'] <= upper_bound_price)]
print(f"Lignes supprimer pour le prix: {original_shape[0] - df_cleaned.shape[0]}")
df = df_cleaned # On met à jour le dataframe

#  Afficher le shape des données
print(df.shape)

#  Calculer le taux des valeurs aberrantes sur le dataframe
deleted_count = original_shape[0] - df.shape[0]
total_count = original_shape[0]
print(f"{deleted_count / total_count * 100:.2f}%")

# Analyse Univariée :
# Afficher les caractéristiques statistiques de l’attribut price<
print("a. Statistiques descriptives du prix:")
print(df['price'].describe())

#Calculer la dispersion des prix
print(f"b. Variance du prix : {df['price'].var()}")
print(f"   Ecart-type du prix : {df['price'].std()}")

# Afficher la courbe de densité des prix
plt.figure()
sns.kdeplot(df['price'], fill=True)
plt.title("Densité du prix")
plt.savefig('price_density.png')
print("c. Densité du prix sauvegardée sous 'price_density.png'")

# Analyse Bivariée
# Faire un lmplot du carat par rapport au prix. Que remarquez-vous
sns.lmplot(x='carat', y='price', data=df, line_kws={'color': 'red'})
plt.title("Carat vs Price")
plt.savefig('lmplot_carat_price.png')
print("a. lmplot carat vs price sauvegardé.")
# Le prix augmente avec le carat.
# Quand le carat augmente, le prix augmente aussi. C'est donc une relation positive

# Faire un lmplot du depth par rapport au prix. Que remarquez-vous
sns.lmplot(x='depth', y='price', data=df, line_kws={'color': 'red'})
plt.title("Depth vs Price")
plt.savefig('lmplot_depth_price.png')
print("b. lmplot depth vs price sauvegardé.")
# Le depth a peu d'influence sur le prix.
# La profondeur n'a pas beaucoup d'effet sur le prix, la relation est donc faible.


# Afficher la matrice de corrélation- heat map
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig('correlation_heatmap.png')
print("c. Heatmap sauvegardée.")
# Le carat est corrélé au prix.
# Le carat a la plus forte corrélation avec le prix parmi les variables numériques.

# Est ce les coupes influent sur les prix ? faire un boxplot pour répondre à cette question
plt.figure()
sns.boxplot(x='cut', y='price', data=df)
plt.title("Cut vs Price")
plt.savefig('boxplot_cut_price.png')
print("d. Boxplot Cut vs Price sauvegardé.")
# Oui, la coupe influence le prix.
# Les différentes coupes montrent des prix différents donc la coupe a un impact sur le prix.

# est ce que les couleurs aussi influent sur les prix ? faire un boxplot pour répondre à cette question
plt.figure()
sns.boxplot(x='color', y='price', data=df)
plt.title("Color vs Price")
plt.savefig('boxplot_color_price.png')
print("e. Boxplot Color vs Price sauvegardé.")
# Oui, la couleur influence aussi le prix.
# Les couleurs différentes donnent des prix différents donc la couleur influence aussi le prix.



# Diviser les donnees en Var Explicatives (X) et Var à Expliquer (Y)
X = df.drop('price', axis=1)
Y = df['price']
print("X shape:", X.shape)
print("Y shape:", Y.shape)

#Encoder les variables qualitatives
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
print(f"Colonnes catégorielles à encoder : {list(categorical_cols)}")

# Encoder les variables qualitatives
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])
print("Variables encodée : ")
print(X.head())

# Diviser les données en training et test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)




