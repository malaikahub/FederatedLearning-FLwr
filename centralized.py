import time
import numpy as np
from model import create_model
from dataset import load_datasets

# Load data
client_data, test_data = load_datasets()

x = []
y = []

for c in client_data:
    x.append(c[0])
    y.append(c[1])

x = np.concatenate(x)
y = np.concatenate(y)

model = create_model()

start_time = time.time()

history = model.fit(
    x, y,
    epochs=5,
    batch_size=32,
    validation_data=test_data,
    verbose=0
)

end_time = time.time()

print("\n===== CENTRALIZED TRAINING =====")

for i, acc in enumerate(history.history["accuracy"]):
    print(f"Epoch {i+1} Accuracy: {acc:.4f}")

final_acc = history.history["accuracy"][-1]

print("\n========= FINAL RESULTS =========")
print(f"Final Accuracy: {final_acc:.4f}")
print(f"Training Time: {end_time - start_time:.2f} seconds")