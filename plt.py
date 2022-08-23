import matplotlib.pyplot as plt

lr_x = [1e-5, 1e-4, 2e-4, 1e-3, 2e-3]
lr_acc_sen = [0.23438, 0.5625, 0.46208, 0.60938,  0.65625]
epoch_x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
epoch_acc_sen = [0.60938, 0.71875, 0.70312, 0.67188, 0.71875, 0.70312, 0.70312, 0.76562, 0.79688]
epoch_acc_seq = [0.64062, 0.73438, 0.67188, 0.75, 0.70312, 0.6875, 0.70312, 0.75, 0.75]

alpha_x = [0.3, 0.4, 0.5, 0.6, 0.7]
alpha_acc_sen = [0.60938, 0.71875, 0.67188, 0.60938, 0.70312]
alpha_acc_seq = [0.64062, 0.76562, 0.67188, 0.57812, 0.71875]

plt.plot(lr_x, lr_acc_sen)
plt.show()

plt.plot(epoch_x, epoch_acc_sen)
plt.show()

plt.plot(epoch_x, epoch_acc_seq)
plt.show()

plt.plot(alpha_x, alpha_acc_sen)
plt.show()

plt.plot(alpha_x, alpha_acc_seq)
plt.show()