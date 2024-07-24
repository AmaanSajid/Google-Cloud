import os
from google.oauth2.service_account import Credentials
from vertexai.preview.generative_models import GenerativeModel, Image
import fitz  # PyMuPDF
import streamlit as st
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

import fitz  # PyMuPDF

def extract_sentences_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    sentences = []
    for page in document:
        text = page.get_text()
        sentences.extend(text.split('.'))  # Simple sentence splitting by period
    document.close()
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# #pdf extract
# def extract_sentences_from_pdf(pdf_path):
#     sentences = []
    
#     # Iterate through all files in the folder
#     for filename in os.listdir(pdf_path):
#         if filename.lower().endswith('.pdf'):
#             pdf_path = os.path.join(pdf_path, filename)
            
#             try:
#                 with open(pdf_path, 'rb') as file:
#                     reader = PyPDF2.PdfReader(file)
#                     text = ""
#                     for page in reader.pages:
#                         if page.extract_text() is not None:
#                             text += page.extract_text() + " "
                
#                 # Split the text into sentences
#                 pdf_sentences = [sentence.strip() for sentence in text.split('. ') if sentence.strip()]
#                 sentences.extend(pdf_sentences)
                
#                 print(f"Processed: {filename}")
#             except Exception as e:
#                 print(f"Error processing {filename}: {str(e)}")
#     print("got sentences")
#     return sentences

#text embeddings
def generate_text_embeddings(sentences) -> list:
  aiplatform.init(project=project,location=location)
  model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
  embeddings = model.get_embeddings(sentences)
  vectors = [embedding.values for embedding in embeddings]
  print("created vectors")
  return vectors


def generate_and_save_embeddings(pdf_folder, sentence_file_path, embed_file_path):
    def clean_text(text):
        cleaned_text = re.sub(r'\u2022', '', text)  # Remove bullet points
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Remove extra whitespaces and strip
        return cleaned_text

    all_sentences = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            sentences = extract_sentences_from_pdf(pdf_path)
            all_sentences.extend(sentences)

    if all_sentences:
        embeddings = generate_text_embeddings(all_sentences)

        with open(embed_file_path, 'w') as embed_file, open(sentence_file_path, 'w') as sentence_file:
            for sentence, embedding in zip(all_sentences, embeddings):
                cleaned_sentence = clean_text(sentence)
                id = str(uuid.uuid4())

                embed_item = {"id": id, "embedding": embedding}  # Convert numpy array to list
                sentence_item = {"id": id, "sentence": cleaned_sentence}

                json.dump(sentence_item, sentence_file)
                sentence_file.write('\n')
                json.dump(embed_item, embed_file)
                embed_file.write('\n')
                      
def upload_file(bucket_name,file_path):
    storage_client = storage.Client()
    bucket = storage_client.create_bucket(bucket_name,location=location)
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)
    print("uploaded in bucket")

def create_and_save_faiss_index(embed_file_path, index_file_path):
    # Load embeddings and IDs
    embeddings = []
    ids = []
    with open(embed_file_path, 'r') as embed_file:
        for line in embed_file:
            embed_item = json.loads(line)
            embeddings.append(embed_item['embedding'])
            ids.append(embed_item['id'])

    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')

    # Get the dimension of the embeddings
    d = embeddings_array.shape[1]

    # Create the index
    index = faiss.IndexFlatL2(d)
    
    # Add vectors to the index
    index.add(embeddings_array)

    # Save the index
    faiss.write_index(index, index_file_path)

    # Save the IDs separately (FAISS doesn't store them)
    with open(index_file_path + '.ids', 'w') as id_file:
        json.dump(ids, id_file)
    print(f"Index saved to {index_file_path}")
    print(f"IDs saved to {index_file_path}.ids")

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)
    print(f"File {source_file_path} uploaded to {destination_blob_name}.")

    # Upload both FAISS index and IDs file
    
def load_local_faiss_index(index_path, ids_path):
    # Load the FAISS index
    index = faiss.read_index(index_path)
    
    # Load the IDs
    with open(ids_path, 'r') as id_file:
        ids = json.load(id_file)
    
    return index, ids

def create_vector_index(bucket_name, index_name):
    resume_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name = index_name,
    contents_delta_uri = "gs://"+bucket_name,
    dimensions = 786,
    approximate_neighbors_count = 10,
    )
    lakeside_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name = index_name,
    public_endpoint_enabled = True
    )

    lakeside_index_endpoint.deploy_index(
    index = resume_index, deployed_index_id = index_name
    )

def generate_embedding(text):
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    embeddings = embedding_model.get_embeddings([text])
    if embeddings and embeddings[0].values:
        return np.array(embeddings[0].values, dtype=np.float32)
    else:
        raise ValueError("Failed to generate embedding")

def search_faiss_index(index, ids, query_vector, k=5):
    query_vector = np.array([query_vector]).astype('float32')
    distances, indices = index.search(query_vector, k)
    results = [
        {"id": ids[idx], "score": float(1 / (1 + dist))}
        for idx, dist in zip(indices[0], distances[0])
    ]
    return results

def fetch_content_by_id(id, content_file_path):
    with open(content_file_path, 'r') as file:
        for line in file:
            item = json.loads(line)
            if item['id'] == id:
                return item['sentence']
    return None

def answer_prompt(prompt, index, ids, content_file_path, model):
    # Generate embedding for the prompt
    query_embedding = generate_embedding(prompt)
    
    # Search for similar items
    similar_items = search_faiss_index(index, ids, query_embedding)
    
    # Fetch content for similar items
    context = ""
    for item in similar_items:
        content = fetch_content_by_id(item['id'], content_file_path)
        if content:
            context += content + " "
    
    # Generate answer using Gemini Pro
    full_prompt = f"Based on the following context from a PDF:\n\n{context}\n\nAnswer this question: {prompt}"
    response = model.generate_content(full_prompt)
    
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
    
    ##### variables
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
    project=""
    location="us-central1"
    pdf_path="pdf_docs"
    bucket_name = "resume-bucket-final"
    sentence_file_path = "resume_sentences.json"
    index_name="resume_index-final-try"
    embed_file_path = 'resume_embeddings.json'
    index_file_path = 'faiss_index'
    
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using Google Cloud")
    
    with st.sidebar:
        st.title("Menu:")
        st.subheader("Original PDF")
        pdf_original = st.file_uploader("Upload original PDF", type="pdf")
        
        st.subheader("Changed PDF")
        pdf_changed = st.file_uploader("Upload changed PDF", type="pdf")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                for uploaded_file in pdf_docs:
                    file_path = os.path.join(pdf_path, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                generate_and_save_embeddings(pdf_path,sentence_file_path,embed_file_path)
                upload_file(bucket_name,embed_file_path)
                # create_vector_index(bucket_name, index_name)
                create_and_save_faiss_index(embed_file_path, index_file_path)
                upload_to_gcs('resume-bucket-final', 'faiss_index', 'faiss_index')
                upload_to_gcs('resume-bucket-final', 'faiss_index.ids', 'faiss_index.ids')
                st.success("Done")
                
    
    user_question = st.text_input("Ask a Question from the Files")
    if st.button("Enter"):
        if user_question:
            vertexai.init(project=project, location=location)
            model = GenerativeModel("gemini-pro")
            # Load local FAISS index
            index_path = "faiss_index"
            ids_path = "faiss_index.ids"
            content_file_path = "resume_sentences.json"
            
            index, ids = load_local_faiss_index(index_path, ids_path)

            # Get user prompt
            user_prompt = user_question
            # Get answer
            answer = answer_prompt(user_prompt, index, ids, content_file_path, model)
            st.write("Reply: ",answer)
            print(answer)
        else:
            st.warning("Please enter the question.")
    
    
    
    
    