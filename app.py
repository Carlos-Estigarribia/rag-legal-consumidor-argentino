# ─────────────────────────────────────────────────────────────────────────────
# app.py — RAG con HuggingFace Inference API y Gradio
# Para HuggingFace Spaces: configurá el secreto HF_TOKEN en Settings.
# Uso local:  HF_TOKEN=tu_token python app.py
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace # No me funciona, los cambio por los siguientes
from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from typing import Optional, List

# ─── Configuración ────────────────────────────────────────────────────────────

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
#MODEL_ID = "google/gemma-2-2b-it" #"meta-llama/Llama-3.2-3B-Instruct"  # gratuito con token HF # ahora no la usamos

if not HF_TOKEN:
    raise ValueError("Configurá el secreto HF_TOKEN en el Space.")

# ─── Embeddings locales (corren en la CPU del Space) ──────────────────────────

modelo_embeddings = SentenceTransformerEmbeddings(
    model_name="intfloat/multilingual-e5-small"
)

# ─── ChromaDB en memoria (sin disco para Spaces) ──────────────────────────────

vectorstore = Chroma(
    collection_name="proyecto_rag_spaces",
    embedding_function=modelo_embeddings
)

# ─── Divisor de texto ─────────────────────────────────────────────────────────

divisor = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " "]
)

# ─── LLM via HuggingFace Serverless Inference ─────────────────────────────────

hf_client = InferenceClient(
    model="Qwen/Qwen2.5-72B-Instruct",
    token=HF_TOKEN
)

class HFInferenceLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "hf_inference"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        response = hf_client.chat_completion(            
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content

llm = HFInferenceLLM()

# ─── Pipeline RAG ─────────────────────────────────────────────────────────────

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def formatear_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

TEMPLATE = """Sos un asistente especializado en derecho del consumidor argentino.
Tu función es ayudar a abogados y estudios jurídicos a consultar legislación y normativa.
Respondé basándote ÚNICAMENTE en los documentos proporcionados.
Cuando sea posible, citá el artículo específico que fundamenta tu respuesta.
Si la respuesta no está en los documentos, decilo claramente: no la inventes.
Respondé siempre en español formal y jurídico.

Documentos:
{context}

Pregunta: {question}

Respuesta:"""

prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=["context", "question"]
)

pipeline_rag = (
    {"context": retriever | formatear_documentos, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ─── Funciones de la interfaz ─────────────────────────────────────────────────

def cargar_pdfs_interfaz(archivos):
    if not archivos:
        return "No se seleccionaron archivos."
    nuevas_paginas = []
    nombres = []
    for archivo in archivos:
        loader = PyPDFLoader(archivo.name)
        paginas = loader.load()
        nuevas_paginas.extend(paginas)
        nombres.append(Path(archivo.name).name)
    nuevos_fragmentos = divisor.split_documents(nuevas_paginas)
    vectorstore.add_documents(nuevos_fragmentos)
    return f"✓ Archivos: {', '.join(nombres)}\n✓ Fragmentos: {len(nuevos_fragmentos)}"

def responder_pregunta(pregunta, historial):
    if not pregunta.strip():
        return historial, ""
    respuesta = pipeline_rag.invoke(pregunta)
    fragmentos_fuente = retriever.invoke(pregunta)
    lineas_fuente = []
    for frag in fragmentos_fuente:
        fuente = Path(frag.metadata.get("source", "desconocida")).name
        pagina = frag.metadata.get("page", "?")
        lineas_fuente.append(f"• {fuente} (pág. {pagina})")
    historial = historial + [
        {"role": "user",      "content": pregunta},
        {"role": "assistant", "content": respuesta}
    ]
    return historial, "\n".join(lineas_fuente)

# ─── Interfaz Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(title="AsistenteLegal RAG — IFTS24", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AsistenteLegal — Derecho del Consumidor Argentino")
    gr.Markdown("**Consultá la Ley 24.240 y el CCyC en lenguaje natural**")

    with gr.Tab("📄 Cargar documentos"):
        upload_component = gr.File(
            label="Seleccioná tus PDFs",
            file_types=[".pdf"],
            file_count="multiple"
        )
        boton_cargar = gr.Button("Indexar documentos", variant="primary")
        estado_carga = gr.Textbox(label="Estado", interactive=False, lines=3)
        boton_cargar.click(
            fn=cargar_pdfs_interfaz,
            inputs=[upload_component],
            outputs=[estado_carga]
        )

    with gr.Tab("💬 Hacer preguntas"):
        chatbot_componente = gr.Chatbot(label="Conversación", height=400)
        with gr.Row():
            pregunta_componente = gr.Textbox(
                label="Tu pregunta",
                placeholder="¿Qué dice el documento sobre...?",
                scale=4
            )
            boton_preguntar = gr.Button("Preguntar", variant="primary", scale=1)
        fuentes_componente = gr.Textbox(
            label="Fragmentos consultados",
            interactive=False,
            lines=3
        )
        boton_preguntar.click(
            fn=responder_pregunta,
            inputs=[pregunta_componente, chatbot_componente],
            outputs=[chatbot_componente, fuentes_componente]
        )
        pregunta_componente.submit(
            fn=responder_pregunta,
            inputs=[pregunta_componente, chatbot_componente],
            outputs=[chatbot_componente, fuentes_componente]
        )

demo.launch()
