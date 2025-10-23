# --- THIS MUST BE THE FIRST THING IN YOUR SCRIPT ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')  # because of chroma database and deploying on streamlit
# --- END OF FIX ---

import streamlit as st
from supporting_functions import (
    extract_video_id,
    get_transcript,
    translate_transcript,
    generate_notes,
    get_important_topics,
    create_chunks,
    create_vector_store,
    rag_answer
)

# --- Sidebar (Inputs) ---
with st.sidebar:
    st.title("VidSynth AI")
    st.markdown("---")
    st.markdown("Transform any YouTube video into key topics, a podcast, or a chatbot.")
    st.markdown("### Input Details")

    youtube_url = st.text_input("YouTube URL", placeholder = "https://www.youtube.com/watch?v=...")
    language = st.text_input("Video Language Code", placeholder = "e.g., en, hi, es, fr", value = "en")

    task_option = st.radio(
        "Choose what you want to generate:",
        ["Chat with Video", "Notes For You"]
    )

    submit_button = st.button("Start Processing")
    st.markdown("---")

    # The "New Chat" button has been removed from here.

# --- Main Page ---
st.title("YouTube Content Synthesizer")
st.markdown("Paste a video link and select a task from the sidebar.")


# --Processing Flow--
if submit_button:
    if youtube_url and language:
        video_id = extract_video_id(youtube_url)
        if video_id:
            with st.spinner("Step 1/3 : Fetching transcript...."):
                full_transcript = get_transcript(video_id, language)

                if language != "en":
                    with st.spinner("Step 1.5/3 : Translating transcript into English, this may take a few moments..."):
                        full_transcript = translate_transcript(full_transcript)


        if task_option == "Notes For You":
            with st.spinner("Step 2/3 : Extracting important Topics..."):
                important_topics = get_important_topics(full_transcript)
                st.subheader("Important Topics")
                st.write(important_topics)
                st.markdown("---")

            with st.spinner("Step 3/3 : Generating Notes for you..."):
                notes = generate_notes(full_transcript)
                st.subheader("Notes for you")
                st.write(notes)

            st.success("Summary and Notes Generated")

        if task_option == "Chat with Video":
            with st.spinner("Step 2/3 : Creating Chunks and Vector store...."):
                chunks = create_chunks(full_transcript)
                vectorstore = create_vector_store(chunks)
                st.session_state.vector_store = vectorstore #created a session named vector_store to store the embeddings
                # and not recall the function everytime when prompt is called, basically, to give memory, as streamlit in itself
                # cant store anything

            st.session_state.messages = [] # empty list to store entire chats as ('User' : message by user , 'Assistant' : message by assistant )
            st.success('Video is ready for chat....')

# chatbot session
if task_option == "Chat with Video" and "vector_store" in st.session_state:
    st.divider()
    st.subheader("Chat with Video")

    # Display the entire history of chats of user and assistant
    for message in st.session_state.get('messages', []):
        with st.chat_message(message['role']): # checking sbka role and the display their icon, then their message
            st.write(message['content'])

    # user input
    prompt = st.chat_input("Ask me anything about the video.")  # store user's input as prompt

    if prompt:
        st.session_state.messages.append({'role' : 'user', 'content': prompt})
        with st.chat_message('user'):   # to create icon for user in front of the text given by user to the chat model
            st.write(prompt)  # to display the prompt that user wrote

        with st.chat_message('assistant'):
            response = rag_answer(prompt, st.session_state.vector_store)
            st.write(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})



