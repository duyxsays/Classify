# %% 
# import libraries - env augment
import aug_methods as am # own package
import os

# %% 
# list categories
root = '/Users/duyx/Code/Classify/training_samples/'

am.delete_ds_store(root)

categories = os.listdir(root)
categories.sort()

print('Available categories: \n')
for category in categories:
    index = categories.index(category)
    print(str(index) + ': ' + category)

# %%
# select a category
index = 5 # <- select a directory/category 
selected_category = categories[index] 

print('\nChosen category: ' + selected_category)

category_dir = os.path.join(root, selected_category)

# %%
# read sample directories
inverted = '/Users/duyx/Code/Classify/augmentation/output/inverted'
shifted = '/Users/duyx/Code/Classify/augmentation/output/pitch_shifted'
stretched = '/Users/duyx/Code/Classify/augmentation/output/time_stretched/'

# reference to the destination folder
augmented_root = '/Users/duyx/Code/Classify/augmentation/data/augmented_bass_data_test'
destination = os.path.join(augmented_root, selected_category)

# %%
# run the augmentation methods
am.polarity_invert_samples(category_dir, inverted, selected_category, "1", "1")              # 1. Invert the samples
am.pitch_shift_samples(inverted, shifted, selected_category, "1", "1")                       # 2. Pitch shift the samples
am.time_stretch_samples(shifted, stretched, selected_category, "1", "1")                     # 3. Time stretch the samples
am.move_data(augmented_root, destination, stretched, selected_category, "1", "1")            # 4. Move the augmented samples to the destination folder


# %%
# reduce the samples to one second
am.reduce_samples(augmented_root)                                                           # 5. Reduce the samples to one second

# %%
# create a time histogram
am.analyse_duration(augmented_root)                                                         # 6. Create a time histogram of the original and augmented samples
