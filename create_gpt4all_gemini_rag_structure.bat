@echo off
setlocal
echo ============================================================
echo GPT4ALL + GEMINI RAG PLATFORM - PROJECT STRUCTURE
echo ============================================================
echo.

if not exist "data" mkdir "data"
if not exist "data\pdfs" mkdir "data\pdfs"
if not exist "src" mkdir "src"
if not exist "tests" mkdir "tests"
if not exist "chroma_db" mkdir "chroma_db"
if not exist "logs" mkdir "logs"
if not exist "docs" mkdir "docs"
if not exist "config" mkdir "config"

if not exist "src\__init__.py" type nul > "src\__init__.py"
if not exist "src\document_loader.py" type nul > "src\document_loader.py"
if not exist "src\chunking.py" type nul > "src\chunking.py"
if not exist "src\embeddings.py" type nul > "src\embeddings.py"
if not exist "src\vector_store.py" type nul > "src\vector_store.py"
if not exist "src\retriever.py" type nul > "src\retriever.py"
if not exist "src\gpt4all_llm.py" type nul > "src\gpt4all_llm.py"
if not exist "src\gemini_llm.py" type nul > "src\gemini_llm.py"
if not exist "src\rag_pipeline.py" type nul > "src\rag_pipeline.py"

if not exist "tests\__init__.py" type nul > "tests\__init__.py"
if not exist "tests\test_document_loader.py" type nul > "tests\test_document_loader.py"
if not exist "tests\test_retriever.py" type nul > "tests\test_retriever.py"
if not exist "tests\test_rag_pipeline.py" type nul > "tests\test_rag_pipeline.py"

if not exist "config\__init__.py" type nul > "config\__init__.py"
if not exist "config\settings.py" type nul > "config\settings.py"

if not exist "app.py" type nul > "app.py"
if not exist "requirements.txt" type nul > "requirements.txt"
if not exist ".env.example" type nul > ".env.example"
if not exist ".gitignore" type nul > ".gitignore"
if not exist "README.md" type nul > "README.md"
if not exist "docs\architecture.md" type nul > "docs\architecture.md"

echo.
echo Project structure created successfully.
echo.
tree /F
echo.
echo ============================================================
echo NEXT STEP
echo ============================================================
echo Copy your PDF files into:
echo data\pdfs\
echo.
echo Then we will start coding document_loader.py
echo ============================================================
pause
endlocal
