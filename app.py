import streamlit as st
import os
import json
from pipeline import initialize_pipeline, incremental_update_document_store
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import Document

DOCUMENT_STORE_PATH = "document_store.json"

def load_document_store():
    """Loads the document store from a JSON file."""
    if os.path.exists(DOCUMENT_STORE_PATH):
        with open(DOCUMENT_STORE_PATH, "r") as f:
            data = json.load(f)
            docs = [Document.from_dict(d) for d in data]
            doc_store = InMemoryDocumentStore()
            doc_store.write_documents(docs)
            return doc_store
    return InMemoryDocumentStore()

def save_document_store(doc_store):
    """Saves the document store to a JSON file."""
    with open(DOCUMENT_STORE_PATH, "w") as f:
        json.dump([doc.to_dict() for doc in doc_store.filter_documents()], f)

def main():
    st.set_page_config(page_title="Corgi Chat", page_icon="2025_08_25_22_21_23_852_424851.webp")
    assistant_avatar_path = "2025_08_25_22_21_23_852_424851.webp"

    # Load document store and initialize pipeline on startup
    if "document_store" not in st.session_state:
        st.session_state.document_store = load_document_store()
        # if st.session_state.document_store.count_documents() > 0:
        #     st.session_state.rag_pipeline = initialize_pipeline(st.session_state.document_store)

    # Sidebar for document management
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)
        
        if st.button("Index Documents"):
            if uploaded_files:
                # We need to save the uploaded files to a temporary directory
                # because some of the processing functions expect file paths.
                temp_dir = "temp_pdf_files"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)

                # Placeholders for temporary progress indicators
                counter_placeholder = st.empty()
                progress_bar_placeholder = st.empty()
                file_path_placeholder = st.empty()

                for progress in incremental_update_document_store(file_paths, st.session_state.document_store):
                    processed = progress["processed"]
                    total = progress["total"]
                    current_file = progress["current_file"]
                    
                    if total > 0:
                        counter_placeholder.text(f"Processing {processed}/{total}")
                        progress_bar_placeholder.progress(processed / total)
                        file_path_placeholder.text(f"Checking: {current_file}")
                    else:
                        counter_placeholder.text(current_file)


                # Clear temporary indicators
                counter_placeholder.empty()
                progress_bar_placeholder.empty()
                file_path_placeholder.empty()
                
                # Save the updated document store and re-initialize the pipeline
                save_document_store(st.session_state.document_store)
                st.session_state.rag_pipeline = initialize_pipeline(st.session_state.document_store)
                
                st.success("Indexing complete!")
                st.session_state.indexed_dir = "Uploaded files" # Update indexed_dir to reflect file upload
            else:
                st.error("Please upload at least one PDF document.")

        if st.button("Clear Chat History"):
            st.session_state.messages = []

        # Display document count
        st.markdown("---")
        st.markdown(f"**{st.session_state.document_store.count_documents()} documents indexed**")
        if "indexed_dir" in st.session_state:
            st.info(st.session_state.indexed_dir)


    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=assistant_avatar_path):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    query = st.chat_input("Ask a question about your documents")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        if "rag_pipeline" not in st.session_state:
            st.warning("Please index your documents before asking a question.")
        else:
            with st.spinner("Generating answer..."):
                result = st.session_state.rag_pipeline.run(
                    data={
                        "text_embedder": {"text": query},
                        "bm25_retriever": {"query": query},
                        "ranker": {"query": query},
                        "prompt_builder": {"question": query}
                    },
                    include_outputs_from={"joiner", "generator"}
                )
            
            answer = result["generator"]["replies"][0] if "generator" in result and "replies" in result["generator"] else "No answer generated."
            references = ""
            if "joiner" in result and "documents" in result["joiner"]:
                source_codes = set(doc.meta.get("doc_code", os.path.basename(doc.meta["file_path"])) for doc in result["joiner"]["documents"])
                references = "\n\n**Reference Documents:**\n" + "\n".join(f"- {code}" for code in source_codes)
            
            response = f"{answer}{references}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant", avatar=assistant_avatar_path):
                st.markdown(response)

if __name__ == "__main__":
    main()