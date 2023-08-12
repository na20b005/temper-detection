num_samples = 5


for i in range(1, num_samples, 2):
    plt.subplot(num_samples, 2, i)
    plt.imshow(in_imgs[idxs[i]])
    plt.subplot(num_samples, 2, i + 1)
    plt.imshow(out_imgs[idxs[i]])