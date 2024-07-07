import os
from google.oauth2.service_account import Credentials
from vertexai.preview.generative_models import GenerativeModel, Image
import fitz  # PyMuPDF
import streamlit as st

def extraction(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # Load the first page
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")

    # Load the image for Vertex AI
    vertex_image = Image.from_bytes(img_data)

    # Create the model and generate content
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

    # Generate content
    response = generative_multimodal_model.generate_content(["Extract the complete text data from this PDF page:", vertex_image])

    if response.text:
        print("Extracted text:")
        print(response.text)
    else:
        print("No text was extracted from the image.")
    return response.text

def summary(response_original, response_changed):
    generative_model = GenerativeModel("gemini-1.5-flash-001")
    # Prepare the prompt
    prompt = f"""
    I have two lists of data extracted from an image:

    Original response:
    {response_original}

    Changed response:
    {response_changed}

    Please provide the differences between these two responses in a point-by-point manner. Focus on key changes, additions, or deletions.
    """

    # Generate the content
    generated_response = generative_model.generate_content(prompt)
    print(generated_response.text) 
    return generated_response.text  


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
    project = "drgenai"
    location = "us-central1"
    bucket_name = "classification-pdf"
    
    # Fix the typo and use os.path.join for cross-platform compatibility
    pdf_original_path = os.path.join("Pdf_classification", "pdf_docs_original")
    pdf_changed_path = os.path.join("Pdf_classification", "pdf_docs_changed")

    # Ensure directories exist
    os.makedirs(pdf_original_path, exist_ok=True)
    os.makedirs(pdf_changed_path, exist_ok=True)

    st.set_page_config(page_title="Classification of PDF")
    st.header("Classfication of Products")

    with st.sidebar:
        st.title("Menu:")
        st.subheader("Original PDF")
        pdf_original = st.file_uploader("Upload original PDF", type="pdf")
        
        st.subheader("Changed PDF")
        pdf_changed = st.file_uploader("Upload changed PDF", type="pdf")

        if st.button("Submit & Process"):
            if pdf_original and pdf_changed:
                with st.spinner("Processing..."):
                    try:
                        # Save original PDF
                        original_path = os.path.join(pdf_original_path, pdf_original.name)
                        with open(original_path, "wb") as f:
                            f.write(pdf_original.getbuffer())

                        # Save changed PDF
                        changed_path = os.path.join(pdf_changed_path, pdf_changed.name)
                        with open(changed_path, "wb") as f:
                            f.write(pdf_changed.getbuffer())

                        # Extract text from both PDFs
                        original_text = extraction(original_path)
                        changed_text = extraction(changed_path)

                        st.session_state['original_text'] = original_text
                        st.session_state['changed_text'] = changed_text

                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.error("Please upload both original and changed PDFs.")

    if st.button("Generate Summary"):
        if 'original_text' in st.session_state and 'changed_text' in st.session_state:
            output = summary(st.session_state['original_text'], st.session_state['changed_text'])
            st.write("Summary of differences:", output)
        else:
            st.warning("Please process the PDFs first.")