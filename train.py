import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import joblib

data = pd.read_csv('games.csv')

data['average_elo'] = (data['white_rating'] + data['black_rating']) / 2
data['elo_difference'] = data['white_rating'] - data['black_rating']

features = data[['rated', 'increment_code', 'opening_eco', 'opening_ply', 'average_elo', 'elo_difference']]
target = data['winner']

features = pd.get_dummies(features)

le = LabelEncoder()
target = le.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print('Decision Tree Accuracy:', accuracy_score(y_test, dt_pred))

# k-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print('KNN Accuracy:', accuracy_score(y_test, knn_pred))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print('Naive Bayes Accuracy:', accuracy_score(y_test, nb_pred))

# Neural Network
nn = MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(50,15,5,))
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)
print('Neural Network Accuracy:', accuracy_score(y_test, nn_pred))

joblib.dump(dt, 'decision_tree_model.pkl')
joblib.dump(nn, 'neural_network_model.pkl')

new_game = pd.read_csv('new_game.csv')

new_game['average_elo'] = (new_game['white_rating'] + new_game['black_rating']) / 2
new_game['elo_difference'] = new_game['white_rating'] - new_game['black_rating']

features_new_game = new_game[['rated', 'increment_code', 'opening_eco', 'opening_ply', 'average_elo', 'elo_difference']]

features_new_game = pd.get_dummies(features_new_game)

missing_cols = {col: pd.Series(0, index=features_new_game.index) for col in X_train.columns if col not in features_new_game.columns}
missing_cols_df = pd.DataFrame(missing_cols)

features_new_game = pd.concat([features_new_game, missing_cols_df], axis=1)

features_new_game = features_new_game[X_train.columns]

for col in X_train.columns:
    if col not in features_new_game.columns:
        features_new_game[col] = 0
features_new_game = features_new_game[X_train.columns]

nn_proba_new_game = nn.predict_proba(features_new_game)

prob_columns = le.inverse_transform([i for i in range(len(dt.classes_))])
nn_probs_df = pd.DataFrame(nn_proba_new_game, columns=prob_columns)

for i in range(9):
    print(f"Expected winner: {new_game['winner'][i]}")
    print("Neural Network Probabilities:\n", nn_probs_df.iloc[i])
    print("")

data['elo_category'] = pd.cut(data['average_elo'], bins=[0, 2000, 3000], labels=['low', 'high'])

transactions = data.apply(lambda x: [x['elo_category'], x['victory_status']], axis=1).tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.001)
filtered_rules = rules[rules['antecedents'].apply(lambda x: 'high' in x or 'low' in x)]
sorted_rules = filtered_rules.sort_values(by='confidence', ascending=False)

print(sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


transactions = data.apply(lambda x: [x['elo_category'], x['opening_eco'], x['winner']], axis=1).tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
# filtered_rules = rules[rules['antecedents'].apply(lambda x: any(y in x for y in ['high', 'low'])) & rules['antecedents'].apply(lambda x: any(y.startswith('A') for y in x))]
filtered_rules = rules[rules['antecedents'].apply(lambda x: any(y in x for y in ['high', 'low'])) & rules['antecedents'].apply(lambda x: any(y == 'B20' or y == 'C70' or y == 'D06' or y == 'D07' for y in x))]
sorted_rules = filtered_rules.sort_values(by='confidence', ascending=False)

print(sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
