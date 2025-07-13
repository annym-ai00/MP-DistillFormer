import os

def get_class_labels_from_split(dataset_name="cub", root_dir='./dataset'):
    if dataset_name == 'cub':
        name = "CUB_200_2011"
    elif dataset_name == 'dog':
        name = "Stanford_Dogs"
    elif dataset_name == 'car':
        name = "Stanford_Cars"
    
    split_folder = os.path.join(root_dir, name)
    splits = ['train', 'valid', 'test']
    class_labels = []

    for split in splits:
        split_path = os.path.join(split_folder, split)
        if not os.path.isdir(split_path):
            continue

        for class_folder in os.listdir(split_path):
            full_class_path = os.path.join(split_path, class_folder)
            if os.path.isdir(full_class_path):
                if dataset_name == 'cub':
                    # Extract class name from folder (starts at 5th character)
                    class_name = class_folder[4:]
                elif dataset_name == 'dog':
                    # Extract class name from folder (starts at 11th character)
                    class_name = class_folder[10:]
                elif dataset_name == 'car':
                    # Extract class name from folder (use full character)
                    class_name = class_folder[:]
                
                class_labels.append(f"a photo of {class_name}")

    return class_labels
