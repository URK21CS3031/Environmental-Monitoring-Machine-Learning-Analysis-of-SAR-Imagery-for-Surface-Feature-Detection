import streamlit as st
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

st.sidebar.markdown("""
    ## Satellite Image Segmentation 🛰️

    ### About the Model
    - **Architecture**: DeepLabV3 with ResNet50 Backbone
    - **Classes**: 6 semantic classes""")
with st.sidebar:
    col1,col2,col3 = st.columns(3)
    with col1:
        color = st.color_picker('Buildings', '#191970')
    with col2:
        color = st.color_picker('Water', '#87ceeb')
    with col3:
        color = st.color_picker('Road', '#cf3476')
    col4,col5,col6 = st.columns(3)
    with col4:
        color = st.color_picker('Land', '#a0522d')
    with col5:
        color = st.color_picker('Vegetation','#3cb371')
    with col6:
        color = st.color_picker('Unlabeled','#a9a9a9')

st.sidebar.markdown("""
    ### How to Use
    1. Upload a satellite image
    2. Wait for the segmentation result
    3. View color-coded mask and class distribution

    ### Model Performance
    - Trained on multi-source satellite imagery
    - Provides pixel-level semantic segmentation

    ### Limitations
    - Best results with 256x256 pixel images
    - Performance may vary with different image types
    """)

# Define color map for visualization
class_colors = {
    0: (25,25,112),   # Building - dark blue
    1: (160,82,45),  # Land - purple
    2: (207,52,118), # Road - light blue
    3: (60,179,113),  # Vegetation - yellow
    4: (135,206,235),  # Water - orange
    5: (169,169,169)  # Unlabeled - gray
}

class_names = {
    0: 'Building',
    1: 'Land',
    2: 'Road',
    3: 'Vegetation', 
    4: 'Water',
    5: 'Unlabeled'
}

def get_model(num_classes=6):
    """Load the pretrained model architecture"""
    model = deeplabv3_resnet50(pretrained=False)
    
    # Modify the classifier to match the number of classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    # Remove the auxiliary classifier
    model.aux_classifier = None
    
    return model

def prepare_image(image):
    """Prepare image for model prediction"""
    # Resize and normalize transformation
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Apply transformations
    transformed = transform(image=image_np)
    return transformed['image'].unsqueeze(0)  # Add batch dimension

def colorize_mask(mask):
    """Convert mask to color image"""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        color_mask[mask == class_idx] = color
    return color_mask

def main():
    st.title("Satellite Image Segmentation")
    st.write("Upload a satellite image to get semantic segmentation results")

    # Model loading
    @st.cache_resource
    def load_model():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model()
        
        # Load the state dict
        state_dict = torch.load('best_model.pth', map_location=device)
        
        # Remove any keys related to auxiliary classifier
        keys_to_remove = [k for k in state_dict.keys() if 'aux_classifier' in k]
        for k in keys_to_remove:
            del state_dict[k]
        
        # Load the cleaned state dict
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, device

    try:
        model, device = load_model()
    except FileNotFoundError:
        st.error("Model file 'best_model.pth' not found. Please ensure the model is saved correctly.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a satellite image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Prepare image for prediction
        input_tensor = prepare_image(image)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)['out']
            predictions = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Visualize prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Segmentation Mask")
            color_mask = colorize_mask(predictions)
            st.image(color_mask, use_container_width=True)

        with col2:
            st.write("### Class Distribution")
            # Calculate class distribution
            unique, counts = np.unique(predictions, return_counts=True)
            class_dist = dict(zip(unique, counts))

            # Create distribution chart
            class_labels = [class_names[k] for k in class_dist.keys()]
            class_percentages = [v / predictions.size * 100 for v in class_dist.values()]

            df = pd.DataFrame({
                'Class': class_labels,
                'Percentage': class_percentages
            })

            # # Optional: Add a more detailed breakdown
            # st.write("Breakdown:")
            # vegetation_percent = df[df['Class'] == 'Vegetation']['Percentage'].values[0] if 'Vegetation' in df['Class'].values else 0
            # land_percent = df[df['Class'] == 'Land']['Percentage'].values[0] if 'Land' in df['Class'].values else 0
            # st.write(f"- Vegetation: {vegetation_percent:.2f}%")
            # st.write(f"- Land: {land_percent:.2f}%")
            # Calculate potential urbanization percentage (sum of Vegetation and Land)
            urbanization_potential = df[df['Class'].isin(['Vegetation', 'Land'])]['Percentage'].sum()

            # Display the bar chart
            st.bar_chart(df.set_index('Class')['Percentage'])

        # Display urbanization potential
        st.write(f"### Potential Urbanization Area")
        st.success(f"The total area available for potential urbanization (Vegetation + Land) is: {urbanization_potential:.2f}%")
            # st.write("### Class Distribution")
            # # Calculate class distribution
            # unique, counts = np.unique(predictions, return_counts=True)
            # class_dist = dict(zip(unique, counts))
            
            # # Create distribution chart
            # class_labels = [class_names[k] for k in class_dist.keys()]
            # class_percentages = [v / predictions.size * 100 for v in class_dist.values()]
            
            # df = pd.DataFrame({
            #     'Class': class_labels,
            #     'Percentage': class_percentages
            # })
            # st.bar_chart(df.set_index('Class')['Percentage'])

        # Detailed class information
        st.write("### Detailed Class Breakdown")
        for class_id, percentage in zip(class_dist.keys(), class_percentages):
            st.info(f"{class_names[class_id]}: {percentage:.2f}%")

if __name__ == "__main__":
    main()