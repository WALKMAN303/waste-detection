import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Page config
st.set_page_config(
    page_title="Eco Guardian - AI Waste Classifier",
    page_icon="♻️",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return YOLO('model/best.pt')

model = load_model()

# Title and description
st.title("♻️ Eco Guardian - AI-Powered Waste Classification")
st.markdown("""
Upload an image to detect and classify waste into recyclable categories.
**Categories:** Plastic, Metal, Glass, Paper, Organic
""")

# Sidebar
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Model:** YOLOv8n  
**Classes:** 5 (Plastic, Metal, Glass, Paper, Organic)  
**mAP@50:** 20.3%  
**Best Performance:** Plastic (38.2%)
""")

# Recycling tips
recycling_tips = {
    'plastic': '♻️ **Plastic**: Rinse and place in recycling bin. Remove caps and labels if possible.',
    'metal': '♻️ **Metal**: Aluminum cans are 100% recyclable. Rinse before recycling.',
    'glass': '♻️ **Glass**: Remove caps and rinse. Can be recycled endlessly without quality loss.',
    'paper': '♻️ **Paper**: Keep dry and flatten boxes. Remove any plastic/tape.',
    'organic': '🌱 **Organic**: Perfect for composting. Reduces landfill waste by 30%.'
}

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    # Run detection
    with st.spinner('🔍 Detecting waste...'):
        results = model(np.array(image), conf=confidence)
        
    with col2:
        st.subheader("🎯 Detection Results")
        # Draw results
        annotated_img = results[0].plot()
        st.image(annotated_img, use_container_width=True)
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Detection Summary")
    
    detections = results[0].boxes
    
    if len(detections) > 0:
        # Count detections by class
        classes = detections.cls.cpu().numpy()
        confidences = detections.conf.cpu().numpy()
        
        class_counts = {}
        for cls, conf in zip(classes, confidences):
            name = model.names[int(cls)]
            if name not in class_counts:
                class_counts[name] = {'count': 0, 'avg_conf': []}
            class_counts[name]['count'] += 1
            class_counts[name]['avg_conf'].append(conf)
        
        # Display in columns
        cols = st.columns(len(class_counts))
        for i, (waste_type, data) in enumerate(class_counts.items()):
            with cols[i]:
                avg_confidence = np.mean(data['avg_conf']) * 100
                st.metric(
                    label=waste_type.capitalize(),
                    value=f"{data['count']} item(s)",
                    delta=f"{avg_confidence:.1f}% conf"
                )
        
        # Recycling recommendations
        st.markdown("---")
        st.subheader("💡 Recycling Recommendations")
        for waste_type in class_counts.keys():
            st.markdown(recycling_tips.get(waste_type.lower(), ""))
        
        # Environmental impact
        st.markdown("---")
        st.subheader("🌍 Environmental Impact")
        impact_values = {
            'plastic': 0.5,
            'metal': 1.2,
            'glass': 0.3,
            'paper': 0.8,
            'organic': 0.2
        }
        
        total_impact = sum(
            class_counts[waste].get('count', 0) * impact_values.get(waste.lower(), 0)
            for waste in class_counts.keys()
        )
        
        st.success(f"♻️ Recycling these items saves approximately **{total_impact:.1f} kg CO₂** emissions!")
        
    else:
        st.warning("⚠️ No waste detected in the image. Try adjusting the confidence threshold or upload a clearer image.")

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    1. **Upload an image** using the file uploader above
    2. **Adjust confidence threshold** in the sidebar if needed
    3. **View detection results** with bounding boxes
    4. **Check recycling recommendations** for detected items
    5. **See environmental impact** of proper recycling
    
    **Tips for best results:**
    - Use clear, well-lit images
    - Ensure waste items are visible
    - Avoid cluttered backgrounds
    - Plastic items work best (38% accuracy!)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using YOLOv8 | <a href='https://github.com/yourusername/eco-guardian'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)