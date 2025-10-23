import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import time

from youtube_transcript_api.proxies import WebshareProxyConfig

load_dotenv()

# Function to extract video ID from a YouTube URL (Helper Function)

def extract_video_id(url):
    """
    Extracts the YouTube video ID from any valid YouTube URL.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None

# function to get transcript of the video
def get_transcript(video_id, language):
    ytt_api = YouTubeTranscriptApi(
        proxy_config = WebshareProxyConfig(
            proxy_username = "zlstcdbe",
            proxy_password = "d3goqke55n3k",
        )
    )
    try:
        transcript = ytt_api.fetch(video_id, languages = [language])
        full_transcript = " ".join([i.text for i in transcript])
        time.sleep(10)
        return full_transcript
    except Exception as e:
        st.error(f"Error fetching video {e}")


# function to translate the transcript to english
    # initialize the genai model
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    temperature = 0.2
)

def translate_transcript(transcript):
    try:
        prompt = ChatPromptTemplate.from_template(f"""
        You are an expert translator with deep cultural and linguistic knowledge.
        I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving:
        - Full meaning and context (no omissions, no additions)
        - Tone and Style (formal/informal, emotional/neutral as in original).
        - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent.)
        - Speaker's voice (same perspective, no rewriting in third-person).
        Do no summarize or simplify. The translation should read naturally in the target language but stay as close as possible to the original intent.
        
        Transcript :
        {transcript}
        """)

        # Runnable Chain
        chain = prompt | llm

        # invoke chain
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e:
        st.error(f"Error fetching transcript {e}")


# function to get important topics
def get_important_topics(transcript):
    try:
        prompt = ChatPromptTemplate.from_template(f"""
                You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.
                
                Rules:
                - Summarize in exactly 5 major points.
                - Each point should represent a key topic or concept, not small details.
                - Keep wording concise and focussed on the technical content.
                - Do not phrase them as questions or opinions.
                - Output should be a numbered list.
                - Show only points that are discussed in the transcript.
                
                Here is the transcript:
                {transcript}    
                """)

        # Runnable Chain
        chain = prompt | llm

        # invoke chain
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e:
        st.error(f"Error fetching transcript {e}")


# function to get notes from the video
def generate_notes(transcript):
    try:
        prompt = ChatPromptTemplate.from_template(f"""
                You are an AI note-taker. Your task is to read the following YouTube video transcript and produce well-structured,
                concise notes.
                
                Requirements:
                - Present the output as **bulleted points**, grouped into clear sections.
                - Highlight key takeaways, important facts, and examples.
                - Use **short, clear sentences** (no long paragraphs).
                - If the transcript includes multiple themes, organize them under **sub-headings**
                - Do not add information that is not present in the transcript.
                
                Here is the transcript:
                {transcript}
                """)

        # Runnable Chain
        chain = prompt | llm

        # invoke chain
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e:
        st.error(f"Error fetching transcript {e}")



# Function to create chunks
def create_chunks(transcript):
    text_splitters = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    doc = text_splitters.create_documents([transcript])
    return doc

# function to creating embeddings of chunks and store it into vector space
def create_vector_store(docs):
    embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embedding)
    return vector_store

# RAG function
def rag_answer(question, vectorstore):
    results = vectorstore.similarity_search(question, k = 4) # will convert question to embeddings and then retrieve top 4 similar chunks from vector store
    context_text = "\n".join([i.page_content for i in results]) # combine the 4 chunks' important content (excluding meta data) to create one long paragraph to give it to llm

    prompt = ChatPromptTemplate.from_template("""
                You are a kind, and precise assistant.
                - Begin with a warm and respectful greeting (avoid repeating greetings every turn)
                - Understand the user's intent even with typos or grammatical mistakes.
                - Answer ONLY using the retrieved context.
                - If answer not in context, say:
                  "I couldn't find information in the database. Could you please rephrase or ask something else?"
                - Keep answers clear, concise, and friendly.
                
                Context:
                {context}
                
                Question;  
                {question}
                
                Answer:
                """)

    # chain
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question":question})

    return response.content



