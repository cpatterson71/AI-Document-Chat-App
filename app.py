import streamlit as st
import os
import json
import boto3
import hashlib
from pipeline import initialize_pipeline, incremental_update_document_store
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import Document

def get_file_hash(file_path):
    """Calculates the SHA256 hash of a file."""
    if not os.path.exists(file_path):
        return None
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def download_from_s3(bucket_name, key, local_path):
    if not bucket_name:
        return False
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, key, local_path)
        return True
    except Exception as e:
        st.error(f"Error downloading {key} from S3 bucket {bucket_name}: {e}")
        return False

def upload_to_s3(bucket_name, key, local_path):
    if not bucket_name:
        return False
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket_name, key)
        return True
    except Exception as e:
        st.error(f"Error uploading {key} to S3 bucket {bucket_name}: {e}")
        return False

def load_document_store(s3_bucket_name, s3_document_store_name):
    """Loads the document store from a JSON file, prioritizing S3."""
    if s3_bucket_name:
        st.info(f"Attempting to load document store from S3 bucket: {s3_bucket_name}")
        if download_from_s3(s3_bucket_name, s3_document_store_name, s3_document_store_name):
            st.success("Document store loaded from S3.")
            with open(s3_document_store_name, "r") as f:
                data = json.load(f)
                docs = [Document.from_dict(d) for d in data]
                doc_store = InMemoryDocumentStore()
                doc_store.write_documents(docs)
                return doc_store
        else:
            st.warning("Could not download document store from S3. Falling back to local file or new store.")

    if os.path.exists(s3_document_store_name):
        st.info("Loading document store from local file.")
        with open(s3_document_store_name, "r") as f:
            data = json.load(f)
            docs = [Document.from_dict(d) for d in data]
            doc_store = InMemoryDocumentStore()
            doc_store.write_documents(docs)
            return doc_store
    st.info("Creating a new InMemoryDocumentStore.")
    return InMemoryDocumentStore()

def save_document_store(doc_store, s3_bucket_name, s3_document_store_name):
    """Saves the document store to a JSON file locally and to S3."""
    with open(s3_document_store_name, "w") as f:
        json.dump([doc.to_dict() for doc in doc_store.filter_documents()], f)
    
    if s3_bucket_name:
        st.info(f"Uploading document store to S3 bucket: {s3_bucket_name}")
        if upload_to_s3(s3_bucket_name, s3_document_store_name, s3_document_store_name):
            st.success("Document store uploaded to S3.")
        else:
            st.error("Failed to upload document store to S3.")

def main():
    st.set_page_config(page_title="Corgi Chat", page_icon="2025_08_25_22_21_23_852_424851.webp")
    assistant_avatar_path = "2025_08_25_22_21_23_852_424851.webp"

    # Sidebar for S3 configuration
    with st.sidebar:
        st.header("S3 Configuration")
        s3_bucket_name = st.text_input("S3 Bucket Name", value="ai-document-chat-document-store")
        s3_document_store_name = st.text_input("Document Store File Name", value="document_store.json")

    # Load document store and initialize pipeline on startup
    if "document_store" not in st.session_state:
        st.session_state.document_store = load_document_store(s3_bucket_name, s3_document_store_name)
        if st.session_state.document_store.count_documents() > 0:
            st.session_state.rag_pipeline = initialize_pipeline(st.session_state.document_store)

    # Sidebar for document management
    with st.sidebar:
        st.header("Document Source")
        source_option = st.radio("Choose document source:", ("Upload Files", "Local Folder"))

        if source_option == "Upload Files":
            uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)
            folder_path = None
        else:
            folder_path = st.text_input("Enter the path to your local folder:")
            uploaded_files = None

        
        if st.button("Index Documents"):
            file_paths = []
            temp_dir = None

            if uploaded_files:
                temp_dir = "temp_pdf_files"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
            
            elif folder_path and os.path.isdir(folder_path):
                file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            
            else:
                st.error("Please upload files or provide a valid folder path.")
                return

            if not file_paths:
                st.warning("No PDF files found to index.")
                return

            # Download existing document store from S3 for comparison
            s3_doc_store_path = "s3_document_store.json"
            download_from_s3(s3_bucket_name, s3_document_store_name, s3_doc_store_path)
            s3_hash = get_file_hash(s3_doc_store_path)
            
            # Get hash of local document store
            local_hash = get_file_hash(s3_document_store_name)

            if s3_hash == local_hash and s3_hash is not None:
                st.info("Document store is already up to date.")
            else:
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
                save_document_store(st.session_state.document_store, s3_bucket_name, s3_document_store_name)
                st.session_state.rag_pipeline = initialize_pipeline(st.session_state.document_store)
                
                st.success("Indexing complete!")
                st.session_state.indexed_dir = folder_path if folder_path else "Uploaded files"
            
            # Clean up temporary files after processing
            if temp_dir:
                for path in file_paths:
                    os.remove(path)
                os.rmdir(temp_dir)
            if os.path.exists(s3_doc_store_path):
                os.remove(s3_doc_store_path)

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
