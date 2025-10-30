# AI Document Query

This tool allows you to ask questions about a collection of PDF documents and get answers based on their content. It uses AI to understand your questions and find the most relevant information in the documents.

## Quick Start (Recommended)

The easiest way to use this tool is to run the included standalone executable.

1.  **Place the Executable:** Make sure the `QA_Document_Store.exe` file is in the main `AI_Projects` folder.

2.  **Set Your API Key:**
    *   In the same `AI_Projects` folder, create a new file named `.env`.
    *   Open the `.env` file in a text editor and add the following line, replacing `your_api_key_here` with your actual OpenAI API key:
        ```
        OPENAI_API_KEY=your_api_key_here
        ```

3.  **Run the Program:** Double-click `QA_Document_Store.exe` to start the program. It will open a command window and guide you through the process.

## How It Works

*   **Document Caching:** The first time you run the program, it will process all the PDF documents in the folder you specify. This can take some time. It will then save the processed data into a file named `document_store.json`.
*   **Fast Queries:** On subsequent runs, the program will load the data from `document_store.json`, which is much faster.
*   **Updating the Data:** If you add new documents or want to re-process the existing ones, simply answer "yes" when the program asks if you want to update the data. This will delete the old `document_store.json` and create a new one.

## For Developers (Running from Source)

If you want to run the program from the Python source code, follow these steps:

1.  **Install Dependencies:**
    ```bash
    pip install -r Haystack_AI_Document_Query/requirements.txt
    ```

2.  **Set Environment Variable:** Make sure your `OPENAI_API_KEY` is set as an environment variable or is in the `.env` file in the `Haystack_AI_Document_Query` directory.

3.  **Run the Script:**
    ```bash
    python Haystack_AI_Document_Query/QA_Document_Store.py
    ```
