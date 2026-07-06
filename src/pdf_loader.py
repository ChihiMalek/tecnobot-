from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Charger PDF
loader = PyPDFLoader("test.pdf")  # ton fichier PDF
docs = loader.load()

print("Pages :", len(docs))

# 2. Split texte
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print("Chunks :", len(chunks))

# 3. Test affichage
print("\n--- Exemple ---\n")
print(chunks[0].page_content)
