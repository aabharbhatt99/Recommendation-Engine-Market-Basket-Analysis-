"""collaborative filtering based recommendation system.ipynb"""

import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_excel("http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx")
df.head()
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]
df.head()
basket = (df[df['Country'] == "France"]
        .groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum().unstack().reset_index().fillna(0)
        .set_index('InvoiceNo'))
basket.head()
def encoding_units(x):
  if x <= 0:
    return 0
  if x > 0:
    return 1

basket_sets = basket.applymap(encoding_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)
basket_sets.head()
frequen_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
frequen_itemsets.head()
rules = association_rules(frequen_itemsets, metric="lift", min_threshold=1)
rules.head()
rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]
rules.head(10)


"""Real code"""

import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_excel("OnlineRetail.csv")
df.head()

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]
df.head()

basket = (df[df['Country'] == "France"]
        .groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum().unstack().reset_index().fillna(0)
        .set_index('InvoiceNo'))
basket.head()

def encoding_units(x):
  if x <= 0:
    return 0
  if x > 0:
    return 1

basket_sets = basket.applymap(encoding_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)
basket_sets.head()

"""**TRAINING DATA**"""

frequen_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
frequen_itemsets.head()

rules = association_rules(frequen_itemsets, metric="lift", min_threshold=1)
rules.head()

rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]
rules.head()

basket['ALARM CLOCK BAKELIKE GREEN'].sum()

basket['ALARM CLOCK BAKELIKE RED'].sum()

basket2 = (df[df['Country'] =="Germany"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets2 = basket2.applymap(encoding_units)
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

rules2[ (rules2['lift'] >= 4) &
        (rules2['confidence'] >= 0.5)]