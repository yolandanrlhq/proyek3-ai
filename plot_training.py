import matplotlib.pyplot as plt

# =================================================================
# DATA HISTORI TRAINING ASLI BERDASARKAN LOG TERBARU YOLANDA
# =================================================================

# 30 Epoch penuh
epochs = list(range(1, 31))

# Data Akurasi Latihan dan Validasi
train_acc = [
    0.3567, 0.3897, 0.4030, 0.4083, 0.4219, 0.4245, 0.4308, 0.4315, 0.3948, 0.4045,
    0.4033, 0.4040, 0.4068, 0.4066, 0.4094, 0.4100, 0.4084, 0.4122, 0.4121, 0.4098,
    0.5047, 0.5459, 0.5677, 0.5991, 0.6154, 0.6262, 0.6339, 0.6403, 0.6494, 0.6594
]

val_acc = [
    0.2350, 0.2396, 0.2642, 0.2640, 0.2901, 0.2901, 0.2950, 0.2966, 0.3076, 0.3079,
    0.3072, 0.3070, 0.3107, 0.3107, 0.3094, 0.3141, 0.3125, 0.3141, 0.3176, 0.3158,
    0.3807, 0.4210, 0.4633, 0.4589, 0.4959, 0.4850, 0.5150, 0.4520, 0.4835, 0.4890
]

# Data Loss Latihan dan Validasi
train_loss = [
    1.4422, 1.2972, 1.2642, 1.2460, 1.2263, 1.2208, 1.2088, 1.2078, 1.2396, 1.2257,
    1.2282, 1.2238, 1.2214, 1.2206, 1.2163, 1.2155, 1.2173, 1.2119, 1.2112, 1.2090,
    1.0796, 0.9876, 0.9381, 0.8643, 0.8290, 0.8071, 0.7832, 0.7650, 0.7432, 0.7263
]

val_loss = [
    1.5630, 1.5337, 1.4680, 1.4708, 1.4236, 1.4257, 1.4349, 1.4375, 1.3824, 1.3814,
    1.3827, 1.3803, 1.3803, 1.3783, 1.3788, 1.3782, 1.3774, 1.3781, 1.3744, 1.3802,
    1.6790, 1.3963, 1.4177, 1.3360, 1.2882, 1.3828, 1.2765, 1.5917, 1.5588, 1.5085
]

# =================================================================
# GRAFIK 1: PERKEMBANGAN AKURASI (ACCURACY) - BERSIH
# =================================================================
plt.figure(figsize=(11, 6))
plt.plot(epochs, train_acc, color='#1f77b4', marker='o', linestyle='-', linewidth=2, label='Training Accuracy')
plt.plot(epochs, val_acc, color='#ff7f0e', marker='s', linestyle='-', linewidth=2, label='Validation Accuracy')

plt.title('Grafik Progres Akurasi Model MobileNetV2', fontsize=14, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Nilai Akurasi', fontsize=12)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(1, 30)

plt.savefig('accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# =================================================================
# GRAFIK 2: PERKEMBANGAN TINGKAT ERROR (LOSS) - BERSIH
# =================================================================
plt.figure(figsize=(11, 6))
plt.plot(epochs, train_loss, color='#1f77b4', marker='o', linestyle='-', linewidth=2, label='Training Loss')
plt.plot(epochs, val_loss, color='#ff7f0e', marker='s', linestyle='-', linewidth=2, label='Validation Loss')

plt.title('Grafik Progres Loss Model MobileNetV2', fontsize=14, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Nilai Loss', fontsize=12)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(1, 30)

plt.savefig('loss.png', dpi=300, bbox_inches='tight')
plt.show()