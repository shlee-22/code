import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates

file_path = 'C: .xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Keep only the necessary columns: hip_angle, knee_angle, ankle_angle, and label
df = df[['hip_angle', 'knee_angle', 'ankle_angle', 'label']]

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X[y <= 2], y[y <= 2], test_size=0.2, random_state=42)

svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    return y_pred


print("SVM Model Evaluation:")
y_pred = evaluate_model(svm_model, X_test, y_test)

cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Normal', 'Incomplete injury', 'Complete injury'],
            yticklabels=['Normal', 'Incomplete injury', 'Complete injury'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Proportion) for Labels 0, 1, 2')
plt.show()

X_new = X[(y == 3) | (y == 4)]
y_new = y[(y == 3) | (y == 4)]
y_new_pred = svm_model.predict(X_new)

label_3_indices = (y_new == 3)
label_4_indices = (y_new == 4)

label_3_predictions = y_new_pred[label_3_indices]
label_4_predictions = y_new_pred[label_4_indices]

label_3_counts = np.bincount(label_3_predictions, minlength=3)
label_4_counts = np.bincount(label_4_predictions, minlength=3)

label_3_probabilities = (label_3_counts / label_3_counts.sum()) * 100
label_4_probabilities = (label_4_counts / label_4_counts.sum()) * 100

labels = ['Normal', 'Incomplete injury', 'Complete injury']
x = np.arange(len(labels))

fig, ax = plt.subplots()
bar_width = 0.35

bar1 = ax.bar(x - bar_width / 2, label_3_probabilities, bar_width, label='Label 3 (Implanted_no sti)', color='white',
              edgecolor='black')
bar2 = ax.bar(x + bar_width / 2, label_4_probabilities, bar_width, label='Label 4 (Implanted_sti)', color='orange',
              edgecolor='black')

for i in range(len(labels)):
    ax.text(x[i] - bar_width / 2, label_3_probabilities[i] + 1, f'{label_3_probabilities[i]:.2f}%', ha='center',
            va='bottom')
    ax.text(x[i] + bar_width / 2, label_4_probabilities[i] + 1, f'{label_4_probabilities[i]:.2f}%', ha='center',
            va='bottom')

ax.set_xlabel('Class')
ax.set_ylabel('Probability (%)')
ax.set_title('Predictions for Labels 3 and 4')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(-5, 105)
ax.legend()

ax.set_yticks(np.arange(0, 110, 10))

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm'}
label_names = ['Normal', 'Incomplete injury', 'Complete injury', 'Implanted_no sti', 'Implanted_sti']
for label in df['label'].unique():
    subset = df[df['label'] == label]
    ax.scatter(subset['hip_angle'], subset['knee_angle'], subset['ankle_angle'], c=colors[label],
               label=label_names[label])

ax.set_xlabel('Hip Angle')
ax.set_ylabel('Knee Angle')
ax.set_zlabel('Ankle Angle')
ax.set_title('3D Scatter Plot of Movement Data')
ax.legend()

ax.view_init(elev=26, azim=137)

plt.show()


def create_radar_chart(df):
    categories = ['hip_angle', 'knee_angle', 'ankle_angle']
    N = len(categories)

    overall_mean = df[categories].mean()
    overall_std = df[categories].std()

    df_standardized = df.copy()
    for category in categories:
        df_standardized[category] = (df[category] - overall_mean[category]) / overall_std[category]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)

    colors = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm'}
    labels = ['Normal', 'Incomplete injury', 'Complete injury', 'Implanted_no sti', 'Implanted_sti']

    for label in df['label'].unique():
        values = df_standardized[df_standardized['label'] == label][categories].mean().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=labels[label], color=colors[label])
        ax.fill(angles, values, color=colors[label], alpha=0.25)

    plt.title('Radar Chart for Models (Z-scores)', size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))



create_radar_chart(df)

plt.show()


def create_parallel_coordinates_plot(df):
    df_filtered = df[df['label'].isin([0, 1, 2])].copy()


    df_filtered.loc[:, 'label'] = df_filtered['label'].map(
        {0: 'Normal', 1: 'Incomplete injury', 2: 'Complete injury'}).astype(str)

    plt.figure(figsize=(10, 6))
    parallel_coordinates(df_filtered, class_column='label', cols=['hip_angle', 'knee_angle', 'ankle_angle'],
                         color=['r', 'g', 'b'])
    plt.title('Parallel Coordinates Plot for Normal, Incomplete injury, and Complete injury')
    plt.xlabel('Angles')
    plt.ylabel('Values')
    plt.legend(loc='upper right')
    plt.show()


create_parallel_coordinates_plot(df)