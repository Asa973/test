import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

examination = pd.read_csv("patient-pathway/examinations.csv", parse_dates=["exam_date"])

sample_patients = examination['id_patient'].drop_duplicates().sample(10, random_state=52)
df_sample = examination[examination['id_patient'].isin(sample_patients)].copy()

df_sample['id_patient'] = df_sample['id_patient'].astype(str)
df_sample = df_sample.sort_values(by=['id_patient', 'exam_date'])

patient_ids = df_sample['id_patient'].unique()
patient_to_num = {pid: i for i, pid in enumerate(patient_ids)}
df_sample['patient_num'] = df_sample['id_patient'].map(patient_to_num)

plt.figure(figsize=(10, 5))

# pour différencier les patients
palette = sns.color_palette("tab10", n_colors=len(patient_ids))

# couleurs différentes par patient
sns.scatterplot(
    data=df_sample,
    x='exam_date',
    y='patient_num',
    hue='id_patient',
    palette=palette,
    s=100,
    legend=False
)

# tracer les lignes reliant les examens de chaque patient
for i, (patient_id, group) in enumerate(df_sample.groupby('id_patient')):
    plt.plot(group['exam_date'].values, group['patient_num'].values, alpha=0.6, color=palette[i])


plt.title("Parcours d'examens de plusieurs patients")
plt.xlabel("Date d'examen")
plt.ylabel("ID Patient")
plt.yticks(ticks=range(len(patient_ids)), labels=patient_ids)
plt.grid(True)
plt.tight_layout()
plt.show()


