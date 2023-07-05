import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats


# Génération de données aléatoires
np.random.seed(0) # pour la reproductibilité
X1 = np.random.rand(100)
X2 = np.random.rand(100)
beta0 = 1
beta1 = 2
beta2 = 3
epsilon = np.random.randn(100)
Y = beta0 + beta1 * X1 + beta2 * X2 + epsilon
# creation d'une data fram
df=pd.read_excel("Classeur1.xlsx",index_col=0)
print(df)

# 2 specification du modele

# Définition des variables dépendantes et indépendantes 
X = df[["pib","avoir"]]
y = df["importation"]


# Ajout d'une constante à nos variables explicatives
X = sm.add_constant(X)

# Spécification du modèle de régression linéaire multiple
model = sm.OLS(y, X)

# 3 Estimation des paramètres

# Estimation des paramètres du modèle
results = model.fit()

# Affichage des résultats
print(results.summary())

# 4 validation des hypothèses

# H1: linéarité
plt.scatter(results.fittedvalues, y)
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs observées')
plt.title('Valeurs prédites vs Valeurs observées')
plt.show()

# indépendance des erreurs (voir la valeur de durbin-Watson)

# test d'homoscédasticité
# Calculer le test de Breusch-Pagan
bp_test = het_breuschpagan(results.resid, results.model.exog)
# Imprimer les résultats
labels = ['Statistique de test de Lagrange multiplier', 'p-valeur de LM', 'Statistique de test à base de F', 'p-valeur de F']
print(dict(zip(labels, bp_test)))

# Prédiction avec le modèle
df['pred'] = results.predict(X)
# Comparaison des valeurs prédites avec les valeurs réelles
plt.scatter(df['pred'], y)
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs réelles')
plt.title('Valeurs prédites vs Valeurs réelles')
plt.show()