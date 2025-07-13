import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Chargement des données
examination = pd.read_csv("patient-pathway/examinations.csv")
indication = pd.read_csv("patient-pathway/indications.csv")

print("Examination sample:\n", examination.head())
print("\nIndications sample:\n", indication.head())

# Nettoyage des colonnes 'indications' et 'findings'
def preprocess_split_lower(df, col_name):
    # On gère les valeurs manquantes en remplaçant par chaîne vide
    df[col_name] = df[col_name].fillna("").str.lower().str.split(",")
    return df

indication = preprocess_split_lower(indication, "indications")
examination = preprocess_split_lower(examination, "findings")

# Explosion des listes pour compter chaque indication et finding
indications_exp = indication.explode("indications")
indications_exp = indications_exp[indications_exp["indications"] != ""]

findings_exp = examination.explode("findings")
findings_exp = findings_exp[findings_exp["findings"] != ""]

# Comptage des occurrences
top_indications = indications_exp["indications"].value_counts().head(10)
top_findings = findings_exp["findings"].value_counts().head(10)

print("\nTop 10 Indications:")
print(top_indications)

print("\nTop 10 Findings:")
print(top_findings)

# Visualisation - Barplots
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.barplot(y=top_indications.index, x=top_indications.values, palette="Blues_d", orient='h')
plt.title("Top 10 Indications")

plt.subplot(1,2,2)
sns.barplot(y=top_findings.index, x=top_findings.values, palette="Greens_d", orient='h')
plt.title("Top 10 Findings")

plt.tight_layout()
plt.show()

# WordCloud
from wordcloud import WordCloud

text_ind = " ".join(indications_exp["indications"].dropna())
text_find = " ".join(findings_exp["findings"].dropna())

wc_ind = WordCloud(width=800, height=400, background_color="white").generate(text_ind)
wc_find = WordCloud(width=800, height=400, background_color="white").generate(text_find)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(wc_ind, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud Indications")

plt.subplot(1,2,2)
plt.imshow(wc_find, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud Findings")

plt.show()

