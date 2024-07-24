from langchain.agents import AgentType
from langchain_google_vertexai import ChatVertexAI
import nltk
from nltk.corpus import stopwords
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatVertexAI
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import os
from langchain.callbacks import get_openai_callback
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import google.auth
import google.auth.transport.requests
from google.cloud import aiplatform
import vertexai.preview.generative_models as generative_models
from langchain_google_vertexai import ChatVertexAI



safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# Initialize Vertex AI
project = "drgenai"
location = "us-central1"
aiplatform.init(project=project,location=location)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "drgenai-7daf412ca440.json"

# Create Gemini model
generative_model = GenerativeModel("gemini-1.5-pro-001")

# Create LangChain chat model
llm = ChatVertexAI(model_name="gemini-1.5-pro-001")

def execute_gcp_query(category):
    project_id = "drgenai"
    dataset_id = "Songs"  
    table_id = "Songs"  
    
    full_table_name = f"`{project_id}.{dataset_id}.{table_id}`"
    
    if category == "1":
        query = f"""
        SELECT
            `Track`, `Artist`, `Album`, `Release Date`, `ISRC`
        FROM
            {full_table_name}
        WHERE
            `Release Date` IS NOT NULL
        LIMIT 10;
        """
    elif category == "2":
        query = f"""
        SELECT
            `Track`, `Artist`, `Spotify Streams`
        FROM
            {full_table_name}
        WHERE
            `Spotify Streams` IS NOT NULL
        ORDER BY
            `Spotify Streams` DESC
        LIMIT 10;
        """
    else:
        query = f"""
        SELECT
            `Artist`, COUNT(`Track`) AS TrackCount, SUM(`Spotify Streams`) AS TotalStreams
        FROM
            {full_table_name}
        GROUP BY
            `Artist`
        ORDER BY
            TotalStreams DESC
        LIMIT 10;
        """
    
    credentials_path = "drgenai-541b904f4ee7.json"
    
    # Authenticate and create a BigQuery client
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = bigquery.Client(credentials=credentials, project=project_id)

    # Execute the query
    query_job = client.query(query)

    df = query_job.to_dataframe()
    
    # Print diagnostic information
    print(f"Retrieved {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nFirst few rows of the data:")
    print(df.head())
    return df

def generate_response(prompt):
    classifier_context = '''Based on the provided prompt, respond with a value pair from the following data {
        "What are the top 10 streamed songs": {
            "Instructions: Return the top 10 songs by Spotify streams. Include Track, Artist, and Spotify Streams in a tabular format."
        },
        "Which artists have the most streams": {
            "Instructions: Return the artists with the highest total Spotify streams. Include Artist, Total Streams, and Number of Tracks in a tabular format."
        },
        "Details of specific songs": {
            "Instructions: If the song title is provided, return the details including Track, Artist, Album, Release Date, and ISRC."
        }
    }. If the prompt matches more than 50 percent or is semantically similar to any of the key, return the corresponding value. If there is no match, return only integer 0.
    '''
    
    response = generative_model.generate_content([classifier_context, prompt],safety_settings=safety_settings)
    final_prompt = response.text

    if final_prompt:
        print("final prompt..is this responded with only value", final_prompt)
        return prompt, final_prompt
    else:
        print("prompt", prompt)
        return prompt, ""
    
def classifier_context(prompt):   
    classifier_context = ''' classify the query into anyone of the below labels. respond with just a label either 
    either 1, 2 or 3 and nothing else the output must always be a single number.
    don't include any other character apart from  1,2 or 3
    
    1 if the prompt is related to song details, artist information, or release dates
    2 if the prompt is related to Spotify streams or top songs
    3 if the prompt is related to artist analysis or none of the above 2 '''

    response = generative_model.generate_content([classifier_context, prompt],safety_settings=safety_settings)
    category = response.text
    return category


with st.sidebar:
    st.title('🤗💬 Spotify Stream Analysis Chat Bot') 
    st.markdown('''
    ### <span style="color:darkblue;"> About: </span>
    This app is an LLM-powered chatbot built using:
    - [Streamlit](<https://streamlit.io/>)
    - [Gemini 1.5 Pro](<https://platform.openai.com/docs/models/gemini-1-5-pro>)
    - [Langchain](<https://python.langchain.com/docs/get_started/introduction.html>)
    #### <span style="color:darkblue;"> Disclaimer: </span>
    <span style="color:maroon;"> ***No confidential information was exposed to any of above mentioned platforms, all data is present within the secure network***</span> ''', unsafe_allow_html=True)

file = "Drgenai.Songs"
if "messages" not in st.session_state:
    st.session_state["messages"] = {file: []}
print(st.session_state)

if st.session_state['messages'][file] == [] or st.sidebar.button("Clear history - {}".format(file[:20]), key=file):
    st.session_state["messages"][file] = [{"role": "assistant", "content": "Hi! I'm your Spotify Stream Analysis bot! How can I help you?"}]

for msg in st.session_state.messages[file]:
    content = msg["content"]
    instruction_index = content.lower().find("instructions:")
    if instruction_index != -1:
        content = content[:instruction_index]
    else:
        content = content.split("instructions:")[0]
    st.chat_message(msg["role"]).write(content.strip())

if prompt := st.chat_input(placeholder="Ask about Spotify data..."):
    category = classifier_context(prompt)
    combined_df = execute_gcp_query(category)
    prompt, final_prompt = generate_response(prompt)
    if final_prompt == "":
        final_prompt = prompt
    
    st.session_state.messages[file].append({"role": "user", "content": prompt + " Instructions:" + final_prompt})

    if st.session_state.messages[file][-1]["role"] != "assistant":
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pandas_df_agent = create_pandas_dataframe_agent(llm,combined_df,verbose=True,agent_executor_kwargs={"handle_parsing_errors": True},allow_dangerous_code=True)
                response = pandas_df_agent.run(st.session_state.messages[file], callbacks=[st_cb])
                print('This is the cost of the question', 'Cost information not available for Vertex AI')
                st.session_state.messages[file].append({"role": "assistant", "content": response})
                st.write(response)