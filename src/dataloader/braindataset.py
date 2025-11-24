TRAIN_DATASET_PATH = "/kaggle/working/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"

class BrainDataset(Dataset):
    'Generates data for PyTorch'
    def __init__(self, list_IDs, dim=(128, 128), n_channels=2, shuffle=True, dataset_path="./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"):
        'Initialization'
        self.dim = dim
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.dataset_path = dataset_path
        
        # Create expanded list of samples (each ID gets VOLUME_SLICES samples)
        self.samples = []
        for ID in list_IDs:
            for slice_idx in range(100):
                self.samples.append((ID, slice_idx))
        
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, idx):
        'Generate one sample of data'
        case_id, slice_idx = self.samples[idx]
        
        # Generate data
        X, y = self._load_sample(case_id, slice_idx)
        
        return X, y
    
    def _load_sample(self, case_id, slice_idx):
        'Loads and processes one sample'
        case_path = os.path.join(dataset_path, case_id)

        # Load FLAIR
        data_path = os.path.join(case_path, f'{case_id}_flair.nii')
        flair = nib.load(data_path).get_fdata()

        # Load T1CE
        data_path = os.path.join(case_path, f'{case_id}_t1ce.nii')
        t1ce = nib.load(data_path).get_fdata()

        # Load segmentation
        data_path = os.path.join(case_path, f'{case_id}_seg.nii')
        seg = nib.load(data_path).get_fdata()

        # Extract and resize the specific slice
        slice_pos = slice_idx + 22
        
        # Prepare input channels
        X = np.zeros((*self.dim, self.n_channels))
        X[:, :, 0] = cv2.resize(flair[:, :, slice_pos], self.dim)
        X[:, :, 1] = cv2.resize(t1ce[:, :, slice_pos], self.dim)
        
        # Prepare segmentation mask
        y_slice = seg[:, :, slice_pos]
        
        # Convert class 4 to class 3
        y_slice[y_slice == 4] = 3
        
        # Create one-hot encoding and resize
        y_one_hot = np.eye(4)[y_slice.astype(int)]  # Shape: (240, 240, 4)
        
        # Resize the one-hot mask
        y_resized = np.zeros((*self.dim, 4))
        for c in range(4):
            y_resized[:, :, c] = cv2.resize(y_one_hot[:, :, c], self.dim)
        
        # Normalize X
        X_max = np.max(X)
        if X_max > 0:
            X = X / X_max
        
        # Convert to PyTorch tensors and change dimension order to (C, H, W)
        X = torch.FloatTensor(X).permute(2, 0, 1)  # (2, IMG_SIZE, IMG_SIZE)
        y = torch.FloatTensor(y_resized).permute(2, 0, 1)  # (4, IMG_SIZE, IMG_SIZE)
        
        return X, y

# Alternative version that maintains batch-like structure similar to original
class BrainDatasetBatched(Dataset):
    'Generates data for PyTorch with batch-like structure'
    def __init__(self, list_IDs, dim=(128, 128), n_channels=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        
        if self.shuffle:
            np.random.shuffle(self.list_IDs)

    def __len__(self):
        'Denotes the number of cases'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        'Generate all slices for one case'
        case_id = self.list_IDs[idx]
        
        # Generate data for all slices of this case
        X, y = self._load_case(case_id)
        
        return X, y
    
    def _load_case(self, case_id):
        'Loads and processes all slices for one case'
        case_path = os.path.join(TRAIN_DATASET_PATH, case_id)

        # Load data
        flair = nib.load(os.path.join(case_path, f'{case_id}_flair.nii')).get_fdata()
        t1ce = nib.load(os.path.join(case_path, f'{case_id}_t1ce.nii')).get_fdata()
        seg = nib.load(os.path.join(case_path, f'{case_id}_seg.nii')).get_fdata()

        # Initialize arrays for all slices
        X = np.zeros((VOLUME_SLICES, *self.dim, self.n_channels))
        Y = np.zeros((VOLUME_SLICES, *self.dim, 4))

        # Process each slice
        for j in range(VOLUME_SLICES):
            slice_pos = j + VOLUME_START_AT
            
            X[j, :, :, 0] = cv2.resize(flair[:, :, slice_pos], self.dim)
            X[j, :, :, 1] = cv2.resize(t1ce[:, :, slice_pos], self.dim)

            y_slice = seg[:, :, slice_pos]
            y_slice[y_slice == 4] = 3
            
            # Create one-hot and resize
            y_one_hot = np.eye(4)[y_slice.astype(int)]
            for c in range(4):
                Y[j, :, :, c] = cv2.resize(y_one_hot[:, :, c], self.dim)

        # Normalize
        X_max = np.max(X)
        if X_max > 0:
            X = X / X_max
        
        # Convert to tensors: (VOLUME_SLICES, channels, height, width)
        X = torch.FloatTensor(X).permute(0, 3, 1, 2)
        Y = torch.FloatTensor(Y).permute(0, 3, 1, 2)
        
        return X, Y

# Usage examples:
# For individual slice processing (recommended for most cases):
train_dataset = BrainDataset(train_ids, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

valid_dataset = BrainDataset(val_ids, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

test_dataset = BrainDataset(test_ids, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)