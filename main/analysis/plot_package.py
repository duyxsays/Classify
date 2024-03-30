import matplotlib.pyplot as plt
import librosa as lr
import os

def plot_category(directory, samples):
    audio_list = []
    for sample in samples:
        audio, sr = lr.load(os.path.join(directory, sample))
        audio_list.append(audio)

    plt.style.use('seaborn-dark-palette')
    nrows, ncols = 5, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 18))
    fig.suptitle(samples[0].capitalize(), size=20)

    x = 0
    for i in range(nrows):
        for j in range(ncols):
            if x == samples.__len__():
                break
            ax[i,j].set_title(samples[x])
            ax[i,j].plot(audio_list[x])
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].grid(False)
            x += 1
    plt.show()

def plot_category2(samples, sample_root):
    audio_categories = []
    for dir in samples:
        items = os.listdir(os.path.join(sample_root, dir))
        audio, sr = lr.load(os.path.join(sample_root, dir, items[0]))
        audio_categories.append(audio)


    plt.style.use('seaborn-dark-palette')
    nrows, ncols =  int(samples.__len__()/2), 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 18))
    fig.suptitle(samples[0].capitalize(), size=20)

    x = 0
    for i in range(nrows):
        for j in range(ncols):
            if x == samples.__len__():
                break
            ax[i,j].set_title(samples[x])
            ax[i,j].plot(audio_categories[x])
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].grid(False)
            x += 1
    plt.show()
