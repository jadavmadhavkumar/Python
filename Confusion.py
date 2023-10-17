import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Example true labels and predicted labels
true_labels = np.random.randint(5, size=10000)
predicted_labels = np.random.randint(5, size=10000)

# Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Create a figure with full-screen size
plt.figure(figsize=(1, 1))

# Create a heatmap for visualization with the custom colormap
sns.heatmap(cm, annot=True, cmap="coolwarm", fmt="d", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Maximize the plot window
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.show()

