from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict
import json
import os
import io
import google.generativeai as genai
import numpy as np
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import faiss
from dotenv import load_dotenv
import datetime  # Adicionado import faltante


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carrega variáveis de ambiente
load_dotenv()

# Configura a API do Gemini - a chave deve estar no .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Chave da API Gemini não encontrada no arquivo .env")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o índice FAISS
dimension = 768  # Dimensão corrigida para o modelo de embedding
index = faiss.IndexFlatL2(dimension)
stored_documents = []  # Armazena metadados e conteúdo dos documentos

def extract_text(file: UploadFile) -> str:
    """Extrai texto de vários tipos de arquivo"""
    content = file.file.read()
    
    if file.filename.lower().endswith('.pdf'):
        # Extrai texto de PDF
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Extrai texto de imagens usando OCR
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image)
        return text
    
    else:
        raise HTTPException(status_code=400, detail="Tipo de arquivo não suportado")

async def generate_embeddings(text: str) -> List[float]:
    """Gera embeddings usando a API do Gemini"""
    try:
        # Gera embeddings usando o modelo de embedding do Gemini
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar embeddings: {str(e)}")

def store_in_vector_db(text: str, embedding: List[float], metadata: Dict):
    """Armazena documento e seu embedding no FAISS"""
    embedding_array = np.array([embedding]).astype('float32')
    index.add(embedding_array)
    
    stored_documents.append({
        "id": len(stored_documents),
        "content": text,
        "metadata": metadata
    })

async def analyze_with_gemini(text: str) -> dict:
    """Analisa texto usando o Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        prompt = f"""
        Analise o seguinte documento e extraia informações-chave.
        Foque em tópicos principais, pontos-chave e detalhes importantes.
        
        Documento:
        {text[:20000]}  # Limita o tamanho para evitar erros
        """
        
        # Chamada CORRETA - sem await na resposta
        response = model.generate_content(prompt)
        
        # Processa a resposta
        if not response.text:
            raise ValueError("Resposta vazia do Gemini")
            
        return {
            "analysis": response.text,
            "confidence": 0.95
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao analisar com Gemini: {str(e)}"
        )

async def search_similar_documents(query: str, k: int = 5) -> List[Dict]:
    """Busca documentos similares usando FAISS"""
    try:
        query_embedding = await generate_embeddings(query)
        query_array = np.array([query_embedding]).astype('float32')
        
        distances, indices = index.search(query_array, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(stored_documents):
                doc = stored_documents[idx]
                results.append({
                    "content": doc["content"][:500] + "...",
                    "metadata": doc["metadata"],
                    "similarity_score": float(1 / (1 + distances[0][i]))
                })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar documentos: {str(e)}")

@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        embedding = await generate_embeddings(text)
        
        metadata = {
            "filename": file.filename,
            "type": file.content_type,
            "size": file.size,
            "timestamp": str(datetime.datetime.now())
        }
        store_in_vector_db(text, embedding, metadata)
        
        analysis = await analyze_with_gemini(text)
        
        results = [
            {
                "content": text[:500] + "...",
                "confidence": 0.98,
                "type": "extracted_text"
            },
            {
                "content": analysis["analysis"],
                "confidence": analysis["confidence"],
                "type": "ai_analysis"
            }
        ]
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_documents(query: str, limit: int = 5):
    results = await search_similar_documents(query, limit)
    return {"results": results}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)