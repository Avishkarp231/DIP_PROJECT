import streamlit as st
import os
from image_mosaic import GenerateMosaic, resize_and_save_images

def download_panorama(result_path):
    with open(result_path, "rb") as f:
        content = f.read()
    st.download_button(
        label="Click here to download Panorama",
        key="download_panorama_click",
        data=content,
        file_name="panorama.jpg",
        mime="image/jpeg",
    )

def clean(parent_folder):
    if os.path.exists(parent_folder):
        for file in os.listdir(parent_folder):
            
            if file.endswith('jpg') or file.endswith('.png'):
                os.remove(os.path.join(parent_folder,file))

def main():
    parent_folder = 'uploaded_images'
    clean(parent_folder)
    clean(os.path.join(parent_folder,'results'))
    st.title("Panorama Generator")

    st.sidebar.header("Upload Images")
    st.sidebar.text("Refresh the page for new input.")
    uploaded_files = st.sidebar.file_uploader("Choose at least 2 images", accept_multiple_files=True, type=["jpg", "png"])
    
    if not uploaded_files or len(uploaded_files) < 2:
        st.warning("Please upload at least 2 images.")
        return
    
    os.makedirs(parent_folder, exist_ok=True)

    img_name_list = []
    for i, file in enumerate(uploaded_files):
        img_path = os.path.join(parent_folder, f"image_{i + 1}.jpg")
        with open(img_path, "wb") as f:
            f.write(file.read())
        img_name_list.append(f"image_{i + 1}.jpg")

    # Resize and save uploaded images
    resize_and_save_images(parent_folder, img_name_list, (800, 800))

    st.success("Images uploaded successfully!")

    # Generate Panorama
    st.subheader("Generated Panorama")
    obj = GenerateMosaic(parent_folder=parent_folder, img_name_list=img_name_list)
    obj.mosaic()
    
    # Remove uploaded images from the screen
    
    # Dynamic result_path based on the number of uploaded files
    result_path = f'uploaded_images/results/panorama_{len(uploaded_files) - 1}.jpg'
    st.image(result_path, caption="Generated Panorama", use_column_width=True)
    
    if result_path:
        download_panorama(result_path)

    
    

if __name__ == "__main__":
    main()
