# %%
# import libraries - activate augment
import aug_methods as am # own package
import os

# %% 
# list folder structure
data_dir = '/Users/duyx/Code/Classify/data/train/version2.0'

categories = am.list_data_folders(data_dir=data_dir)
# %%
# read sample directories
output_dir = '/Users/duyx/Code/Classify/main/augment/output/'
inverted = os.path.join(output_dir, 'inverted')
shifted = os.path.join(output_dir, 'pitch_shifted')
stretched = os.path.join(output_dir, 'time_stretched')

# reference to the destination folder
augmented_dir = '/Users/duyx/Code/Classify/data/augment/augmented_bass_sounds'

# %%
# run the entire augmentation process
category_count = 0
total_categories = len(categories)
print('Augmenting data...')
for category in categories:
    category_count += 1
    category_dir = os.path.join(data_dir, category)
    destination = os.path.join(augmented_dir, category)

    am.polarity_invert_samples(category_dir, inverted, category, str(category_count), str(total_categories))    # 1. Invert the samples
    am.pitch_shift_samples(inverted, shifted,  category, str(category_count), str(total_categories))            # 2. Pitch shift the samples
    am.time_stretch_samples(shifted, stretched, category, str(category_count), str(total_categories))           # 3. Time stretch the samples
    am.move_data(augmented_dir, destination, stretched, category, str(category_count), str(total_categories))   # 4. Move the augmented samples to the destination folder
    
am.reduce_samples(augmented_dir)
# %%
