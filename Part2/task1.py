import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lecture du fichier CSV
exams = pd.read_csv("patient-pathway/examinations.csv", parse_dates=["exam_date"])

# Affichage pour vérification
print(exams.head())

# Calcul du nombre d'examens par patiente
exams_per_patient = exams.groupby("id_patient").size().reset_index(name="num_exams")

# Statistiques globales
print(exams_per_patient.describe())

# Visualisation
plt.figure(figsize=(8, 5))
sns.histplot(exams_per_patient["num_exams"], bins=10)
plt.title("Distribution du nombre d'examens par patiente")
plt.xlabel("Nombre d'examens")
plt.ylabel("Nombre de patientes")
plt.grid(True)
plt.tight_layout()
plt.show()
