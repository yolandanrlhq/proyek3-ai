import matplotlib.pyplot as plt

# =========================
# TRAIN HEAD
# =========================
epochs_head = [1, 2, 3, 4, 5, 6, 7]

train_acc_head = [0.2715, 0.3427, 0.3537, 0.3597, 0.4038, 0.4118, 0.3948]
val_acc_head   = [0.2690, 0.3080, 0.2600, 0.2600, 0.2610, 0.2450, 0.2620]

train_loss_head = [1.5965, 1.3515, 1.3310, 1.2763, 1.2325, 1.1978, 1.1994]
val_loss_head   = [1.3963, 1.3773, 1.4511, 1.4642, 1.5061, 1.5362, 1.4536]

# =========================
# ACCURACY GRAPH
# =========================
plt.figure(figsize=(8,5))

plt.plot(
    epochs_head,
    train_acc_head,
    marker='o',
    label='Training Accuracy'
)

plt.plot(
    epochs_head,
    val_acc_head,
    marker='o',
    label='Validation Accuracy'
)

plt.title('Training dan Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.savefig('accuracy.png', dpi=300)
plt.show()

# =========================
# LOSS GRAPH
# =========================
plt.figure(figsize=(8,5))

plt.plot(
    epochs_head,
    train_loss_head,
    marker='o',
    label='Training Loss'
)

plt.plot(
    epochs_head,
    val_loss_head,
    marker='o',
    label='Validation Loss'
)

plt.title('Training dan Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig('loss.png', dpi=300)
plt.show()