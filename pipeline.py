import os
import re
import json
import hashlib
import io
import concurrent.futures
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.dataclasses import Document
from dotenv import load_dotenv
from haystack.core.component import component
from google.cloud import vision
from pdf2image import convert_from_path
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.rankers import SentenceTransformersSimilarityRanker

# Load environment variables from .env file
load_dotenv()

# --- Set your OpenAI API key ---
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable in your .env file.")

def process_single_pdf(file_path):
    """Processes a single PDF file, performs OCR, and returns a Document object."""
    try:
        print(f"Processing file: {file_path}")
        client = vision.ImageAnnotatorClient.from_service_account_file(r'C:\Users\carlp\OneDrive\Desktop\AI_Projects\AI_Document_Chat\Document_Store\haystack-ai-image-7d30c6a4401d.json')
        images = convert_from_path(file_path)
        full_text = ""
        for i, image in enumerate(images):
            print(f"  Processing page {i+1}/{len(images)} of {os.path.basename(file_path)}")
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                content = output.getvalue()

            image_for_vision = vision.Image(content=content)
            
            response = client.document_text_detection(image=image_for_vision, timeout=120)

            if response.error.message:
                raise Exception(
                    "{}\nFor more info on error messages, check: "
                    "https://cloud.google.com/apis/design/errors".format(response.error.message)
                )
            
            full_text += response.full_text_annotation.text + "\n"

        # Extract document code
        doc_code_match = re.search(r'LPI-[A-Z]{2,3}-\d+', full_text)
        doc_code = doc_code_match.group(0) if doc_code_match else os.path.basename(file_path)

        print(f"Finished processing file: {file_path}, Found doc code: {doc_code}")
        return Document(content=full_text, meta={"file_path": file_path, "doc_code": doc_code})
    except Exception as e:
        print(f"Could not process file {file_path}: {e}")
        return None

def get_file_metadata(file_path):
    """Gets metadata for a file (path, last modified time, and hash)."""
    return {
        "file_path": file_path,
        "last_modified": os.path.getmtime(file_path),
        "sha256": hashlib.sha256(open(file_path, "rb").read()).hexdigest()
    }

def initialize_pipeline(document_store):
    """Initializes and returns the RAG pipeline."""
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    embedding_retriever = InMemoryEmbeddingRetriever(document_store, top_k=10)
    bm25_retriever = InMemoryBM25Retriever(document_store, top_k=10)
    joiner = DocumentJoiner()
    ranker = SentenceTransformersSimilarityRanker()
    prompt_template = """
    You are a helpful assistant. Your primary task is to find and directly quote the part of the document that answers the user's question.

    Here is an example of how to answer:

    ---
    Example Context:
    The document LPI-SOP-123 describes the procedure for calibrating the pH meter. The calibration should be performed daily by a qualified analyst. The analyst must use certified buffer solutions of pH 4.0, 7.0, and 10.0. The results of the calibration must be recorded in the pH meter logbook.

    Example Question:
    How often should the pH meter be calibrated, and by whom?

    Example Answer:
    According to document LPI-SOP-123, "the pH meter should be calibrated daily by a qualified analyst."
    ---

    Now, answer the following question based on the context below.

    Context:
    {{ documents }}

    Question:
    {{ question }}

    Instructions:
    1.  Carefully read the question and the provided context.
    2.  Your main goal is to find the exact sentence or phrase in the context that answers the question.
    3.  Provide the answer as a direct quote. If the document name is available, mention it.
    4.  If the answer cannot be found in the context, respond with "I am sorry, but I cannot answer your question based on the provided documents."
    """
    prompt_builder = PromptBuilder(template=prompt_template)
    generator = OpenAIGenerator(model="gpt-4")

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("embedding_retriever", embedding_retriever)
    rag_pipeline.add_component("bm25_retriever", bm25_retriever)
    rag_pipeline.add_component("joiner", joiner)
    rag_pipeline.add_component("ranker", ranker)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", generator)

    rag_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    rag_pipeline.connect("embedding_retriever.documents", "joiner.documents")
    rag_pipeline.connect("bm25_retriever.documents", "joiner.documents")
    rag_pipeline.connect("joiner.documents", "ranker.documents")
    rag_pipeline.connect("ranker.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")
    
    return rag_pipeline

def incremental_update_document_store(pdf_files, document_store):
    """
    Compares the current index with the directory, updates the index incrementally,
    and yields progress updates.
    """
    
    # Get documents already in the store
    indexed_docs = {doc.meta["file_path"]: doc for doc in document_store.filter_documents()}
    
    docs_to_delete_ids = []
    files_to_process = []

    # Check for new and modified files
    for file_path in pdf_files:
        if file_path not in indexed_docs:
            files_to_process.append(file_path)
        else:
            # Use .get() for safe access to sha256
            if get_file_metadata(file_path)["sha256"] != indexed_docs[file_path].meta.get("sha256"):
                docs_to_delete_ids.append(indexed_docs[file_path].id)
                files_to_process.append(file_path)

    # Check for deleted files
    indexed_file_paths = set(indexed_docs.keys())
    disk_file_paths = set(pdf_files)
    deleted_files = indexed_file_paths - disk_file_paths
    for file_path in deleted_files:
        docs_to_delete_ids.append(indexed_docs[file_path].id)

    # Delete documents that need to be updated or were deleted
    if docs_to_delete_ids:
        document_store.delete_documents(docs_to_delete_ids)

    # Process new and modified files
    if not files_to_process:
        yield {"processed": 0, "total": 0, "current_file": "No changes detected."}
        return

    total_files = len(files_to_process)
    processed_files = 0
    
    documents_to_index = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_pdf, pdf_file): pdf_file for pdf_file in files_to_process}
        for future in concurrent.futures.as_completed(future_to_file):
            processed_files += 1
            file_path = future_to_file[future]
            doc = future.result()
            if doc:
                # Add sha256 hash to metadata for future comparisons
                doc.meta["sha256"] = get_file_metadata(file_path)["sha256"]
                documents_to_index.append(doc)
            
            yield {
                "processed": processed_files,
                "total": total_files,
                "current_file": os.path.basename(file_path)
            }

    if documents_to_index:
        doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        doc_splitter = DocumentSplitter(split_by="sentence", split_length=3, split_overlap=1)
        writer = DocumentWriter(document_store)

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("splitter", doc_splitter)
        indexing_pipeline.add_component("embedder", doc_embedder)
        indexing_pipeline.add_component("writer", writer)
        
        indexing_pipeline.connect("splitter.documents", "embedder.documents")
        indexing_pipeline.connect("embedder.documents", "writer.documents")
        
        indexing_pipeline.run({"splitter": {"documents": documents_to_index}})