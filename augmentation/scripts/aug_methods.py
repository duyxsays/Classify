# import libraries
import matplotlib.pyplot as plt
import librosa as lr
import soundfile as sf
import os
import shutil
import seaborn as sns

# MARK: Augmentation methods ---------------------------------------------------------------
def polarity_invert_samples(input, output, category, progress, total):
    delete_ds_store(input)
    handle_directory(output)
    
    total_samples = sum_items(input); i = 0

    for sample in os.listdir(input):
        signal, sr = lr.load(os.path.join(input, sample), sr=16000, mono=True)
        
        augmented_audio = signal * -1
        output_path = os.path.join(output, 'inverted_' + sample)
        copy_path = os.path.join(output, sample)
        
        sf.write(output_path, augmented_audio, sr)
        sf.write(copy_path, signal, sr)
        
        i += 1
        write_progress(i, total_samples, " - Polarity inverting", category, progress, total)
    

def pitch_shift_samples(input, output, category, progress, total):
    handle_directory(output)
    
    semitones = [-6, -5, -4, -3, -2, -1, 1, 2, 3]
    total_samples = sum_items(input) * len(semitones); i = 0

    for sample in os.listdir(input):
        signal, sr = lr.load(os.path.join(input, sample), sr=16000, mono=True)

        for semitone in semitones:
            augmented_audio = lr.effects.pitch_shift(y=signal, sr=sr, n_steps=semitone)
            output_path = os.path.join(output, 'pitch_shift_' + str(semitone) + '_' + sample)
            sf.write(output_path, augmented_audio, sr)

            i += 1
            write_progress(i, total_samples, " - Pitch shifting", category, progress, total)

    remove_files(input) # delete files from the inverted folder


def time_stretch_samples(input, output, category, progress, total):
    handle_directory(output)
    
    rates = [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1,15, 1.2]
    total_samples = sum_items(input) * len(rates); i = 0; 

    for sample in os.listdir(input):
        signal, sr = lr.load(os.path.join(input, sample), sr=16000) 

        for rate in rates:
            augmented_audio = lr.effects.time_stretch(y=signal, rate=rate)
            output_path = os.path.join(output, 'time_stretch_' + str(rate) + '_' + sample)
            
            sf.write(output_path, augmented_audio, sr)

            i += 1
            write_progress(i, total_samples, " - Time stretching", category, progress, total)

    remove_files(input) # delete files from the shifted folder

def reduce_samples(output):
    delete_ds_store(output)

    total_samples = sum_items(output)
    i = 0
    # Read wav files, reduce to one second samples, and save them in a folder
    for category in os.listdir(output):
        category_path = os.path.join(output, category)

        j = 0
        for sample in os.listdir(category_path):
            sample_path = os.path.join(category_path, sample)
            audio, sr = lr.load(os.path.join(category_path, sample), sr=16000)
            duration = len(audio) / sr

            if duration > 1:
                audio = audio[:sr]  # Take the first second of audio

            i += 1; j += 1

            destination = os.path.join(category_path, category + '_' + str(j) + '.wav')
            sf.write(destination, audio, sr)
            
            if os.path.exists(sample_path):
                os.remove(sample_path)
            
            write_reducing(i, total_samples, "Reducing")
            

# MARK: Supporting ---------------------------------------------------------------
def handle_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        # remove existing files in the output folder
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def remove_files(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def move_data(augmented_root, destination, stretched, category, progress, total):
    if not os.path.exists(augmented_root):
            os.mkdir(augmented_root)

    if not os.path.exists(destination):
        os.mkdir(destination)
    else:
        # remove existing files in the output folder
        for file in os.listdir(destination):
            file_path = os.path.join(destination, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    total_samples = sum_items(stretched); i = 0
    # Move files from one path to another
    for file in os.listdir(stretched):
        file_path = os.path.join(stretched, file)
        if os.path.isfile(file_path):
            shutil.move(file_path, destination)
            i += 1
            write_progress(i, total_samples, " - Moving", category, progress, total)
        

def sum_items(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])

def write_progress(i, total_samples, message, category, progress, total):
    #sys.stdout.write("\r"+ category + " " + progress + "/" + total + " " + message + ": %i out of %i" % (i, total_samples))
    #sys.stdout.flush()
    spaces = 20 * ' '
    print(category + ' '  + progress + '/' + total + ' ' + message + ': ' + str(i) + ' out of ' + str(total_samples) + spaces, end='\r', flush=True)
    

def write_reducing(i, total_samples, message):
    spaces = 20 * ' '
    #sys.stdout.write("\r" + message + ": %i out of %i" % (i, total_samples))
    #sys.stdout.flush()
    print(message + ': ' + str(i) + ' out of ' + str(total_samples) + spaces, end='\r', flush=True)

def delete_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")


# MARK: Anaysis ---------------------------------------------------------------
def analyse_duration(augmented_root):
    delete_ds_store(augmented_root)
    durations = []
    
    for category in os.listdir(augmented_root):
        category_path = os.path.join(augmented_root, category)
        
        for sample in os.listdir(category_path):
            sample_path = os.path.join(category_path, sample)

            if os.path.isfile(sample_path):
                signal, sr = lr.load(sample_path, sr=16000)
                duration = len(signal) / sr
                durations.append(duration)

    plt.hist(durations, bins=5)
    plt.xlabel('Duration')
    plt.ylabel('Count')
    plt.title('Histogram of Durations')
    plt.show()

    # Plotting the box plot
    plt.boxplot(durations)
    plt.title('Box Plot of Audio Sample Durations')
    plt.ylabel('Duration (seconds)')
    plt.show()

    sns.kdeplot(durations, fill=True)
    plt.title('Kernel Density Estimate Plot of Audio Sample Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Density')
    plt.show()