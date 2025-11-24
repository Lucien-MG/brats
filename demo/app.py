import streamlit as st
import os
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, dropout=0.5):
        super(UNet, self).__init__()
        
        # Encoder (Contracting Path)
        # Block 1
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5 (Bottleneck)
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout)
        
        # Decoder (Expanding Path)
        # Block 6
        self.upconv6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 512 = 256 + 256 (concatenation)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Block 7
        self.upconv7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 256 = 128 + 128
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Block 8
        self.upconv8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 128 = 64 + 64
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Block 9
        self.upconv9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 64 = 32 + 32
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Output layer
        self.conv10 = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        # Block 1
        conv1 = F.relu(self.conv1_1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        # Block 2
        conv2 = F.relu(self.conv2_1(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)
        
        # Block 3
        conv3 = F.relu(self.conv3_1(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)
        
        # Block 4
        conv4 = F.relu(self.conv4_1(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)
        
        # Block 5 (Bottleneck)
        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))
        conv5 = self.dropout(conv5)
        
        # Decoder
        # Block 6
        up6 = self.upconv6(conv5)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = F.relu(self.conv6_1(merge6))
        conv6 = F.relu(self.conv6_2(conv6))
        
        # Block 7
        up7 = self.upconv7(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = F.relu(self.conv7_1(merge7))
        conv7 = F.relu(self.conv7_2(conv7))
        
        # Block 8
        up8 = self.upconv8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = F.relu(self.conv8_1(merge8))
        conv8 = F.relu(self.conv8_2(conv8))
        
        # Block 9
        up9 = self.upconv9(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = F.relu(self.conv9_1(merge9))
        conv9 = F.relu(self.conv9_2(conv9))
        
        # Output
        out = self.conv10(conv9)
        
        return out

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="BraTS MRI Slicer")

# Default paths (Try to match your environment)
DEFAULT_DATA_PATH = "./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
IMG_SIZE = 128
SLICE_START = 22  # Matches your dataset logic
SLICE_END = 122   # Matches your dataset logic (start + 100)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def load_model(model_path, device):
    """Loads the model architecture and weights."""
    model = UNet(in_channels=2, num_classes=4)
    model.to(device)
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Model weights loaded successfully.")
        except Exception as e:
            st.warning(f"Could not load weights from {model_path}. Using random initialization. Error: {e}")
    else:
        st.warning(f"Model file not found at {model_path}. Using random initialization for demo.")
    
    model.eval()
    return model

@st.cache_data
def load_case_volume(case_path, case_id):
    """
    Loads the full 3D volumes for a specific case.
    Returns: FLAIR, T1CE, SEG (original size)
    """
    flair_path = os.path.join(case_path, f'{case_id}_flair.nii')
    t1ce_path = os.path.join(case_path, f'{case_id}_t1ce.nii')
    seg_path = os.path.join(case_path, f'{case_id}_seg.nii')

    if not os.path.exists(flair_path):
        return None, None, None

    flair = nib.load(flair_path).get_fdata()
    t1ce = nib.load(t1ce_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    
    return flair, t1ce, seg

def preprocess_slice(flair_slice, t1ce_slice):
    """Resizes and normalizes a single slice for the model."""
    X = np.zeros((IMG_SIZE, IMG_SIZE, 2))
    X[:, :, 0] = cv2.resize(flair_slice, (IMG_SIZE, IMG_SIZE))
    X[:, :, 1] = cv2.resize(t1ce_slice, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    X_max = np.max(X)
    if X_max > 0:
        X = X / X_max
        
    # To Tensor (C, H, W)
    X_tensor = torch.FloatTensor(X).permute(2, 0, 1)
    return X_tensor

def predict_volume(model, flair_vol, t1ce_vol, device):
    """
    Runs inference on the specific slice range (22-122) defined in your dataset.
    Returns a 3D volume of predictions resized back to (128, 128).
    """
    preds = []
    
    # Only process the relevant slices
    with torch.no_grad():
        for i in range(SLICE_START, SLICE_END):
            # Prepare input
            inp = preprocess_slice(flair_vol[:,:,i], t1ce_vol[:,:,i])
            inp = inp.unsqueeze(0).to(device) # Add batch dim -> (1, 2, 128, 128)
            
            # Inference
            output = model(inp)
            
            # Get class with highest probability
            pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0] # (128, 128)
            preds.append(pred_mask)
            
    return np.array(preds) # Shape (100, 128, 128)

# ==========================================
# 4. MAIN APP UI
# ==========================================

def main():
    st.title("🧠 BraTS 2020 Tumor Segmentation Viewer")
    st.markdown("Visualize FLAIR MRI, Ground Truth Labels, and Model Predictions.")

    # --- Sidebar Controls ---
    st.sidebar.header("Configuration")
    
    # Dataset Path
    dataset_root = st.sidebar.text_input("Dataset Path", DEFAULT_DATA_PATH)
    model_path = st.sidebar.text_input("Model Path (.pth)", "weights/" + "unet_brats.pth")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"Running on: {device}")

    # Get list of available cases
    if os.path.exists(dataset_root):
        cases = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
        selected_case = st.sidebar.selectbox("Select Case ID", cases)
    else:
        st.error("Dataset path not found.")
        return

    # --- Load Data & Model ---
    if selected_case:
        case_path = os.path.join(dataset_root, selected_case)
        
        with st.spinner(f"Loading Case {selected_case}..."):
            flair_vol, t1ce_vol, seg_vol = load_case_volume(case_path, selected_case)

        if flair_vol is None:
            st.error("Could not load NIfTI files. Check directory structure.")
            return

        # Load Model
        model = load_model(model_path, device)

        # --- Run Prediction (Cached) ---
        # We create a specific cache key based on case_id so it re-runs only when case changes
        if 'current_case' not in st.session_state or st.session_state.current_case != selected_case:
            with st.spinner("Running inference on volume..."):
                prediction_vol = predict_volume(model, flair_vol, t1ce_vol, device)
                st.session_state.prediction_vol = prediction_vol
                st.session_state.current_case = selected_case
        
        pred_vol = st.session_state.prediction_vol

        # --- Visualization Controls ---
        st.markdown("---")
        # Slider logic: relative to the 100 slices we extracted
        # We map 0-99 (prediction indices) to 22-121 (original MRI indices)
        slice_idx = st.slider("Select Slice (Z-Axis)", 0, 99, 50)
        actual_slice_pos = slice_idx + SLICE_START

        # --- Prepare Images for Plotting ---
        
        # 1. Original FLAIR (Resized for visualization consistency)
        img_flair = cv2.resize(flair_vol[:, :, actual_slice_pos], (IMG_SIZE, IMG_SIZE))
        
        # 2. Ground Truth (Resized and remapped)
        img_seg = seg_vol[:, :, actual_slice_pos]
        img_seg[img_seg == 4] = 3 # Remap class 4 to 3
        img_seg_resized = cv2.resize(img_seg, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # 3. Prediction
        img_pred = pred_vol[slice_idx]

        # --- Plotting ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Custom colormap for masks (0=bg, 1=NCR, 2=ED, 3=ET)
        cmap_mask = mcolors.ListedColormap(['black', 'red', 'green', 'blue'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap_mask.N)

        # Plot MRI
        axes[0].imshow(img_flair, cmap='gray')
        axes[0].set_title(f"MRI (FLAIR)\nSlice {actual_slice_pos}")
        axes[0].axis('off')

        # Plot Ground Truth
        axes[1].imshow(img_seg_resized, cmap=cmap_mask, norm=norm)
        axes[1].set_title("Ground Truth\n(Red:1, Green:2, Blue:4->3)")
        axes[1].axis('off')

        # Plot Prediction
        axes[2].imshow(img_pred, cmap=cmap_mask, norm=norm)
        axes[2].set_title("Model Prediction")
        axes[2].axis('off')

        st.pyplot(fig)

        # Legend
        st.markdown("""
        **Legend:**
        - ⬛ **Black:** Background (0)
        - 🟥 **Red:** Necrotic/Core (1)
        - 🟩 **Green:** Edema (2)
        - 🟦 **Blue:** Enhancing Tumor (4 mapped to 3)
        """)

if __name__ == "__main__":
    main()