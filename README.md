# AsistenteLegal — RAG para Derecho del Consumidor Argentino

## ¿De qué se trata?

Sistema de consulta jurídica basado en RAG (Retrieval-Augmented Generation) 
orientado a abogados independientes y pequeños estudios jurídicos argentinos. 
Permite hacer preguntas en lenguaje natural sobre legislación y jurisprudencia 
de derecho del consumidor, obteniendo respuestas fundamentadas con cita del 
artículo o fallo correspondiente.

## ¿Para qué sirve?

Los profesionales del derecho pierden horas buscando manualmente en expedientes, 
legislación y jurisprudencia dispersa cada vez que necesitan fundamentar un caso. 
Este sistema indexa esa documentación y permite consultarla conversacionalmente, 
devolviendo tiempo para el trabajo que realmente requiere expertise jurídico.

## ¿Cómo funciona?

1. El usuario sube sus PDFs (leyes, fallos, doctrina)
2. El sistema los fragmenta y los indexa en una base vectorial (ChromaDB)
3. Ante cada pregunta, recupera los fragmentos más relevantes
4. Un LLM genera una respuesta basada únicamente en esos fragmentos
5. Se muestra la respuesta junto con las fuentes consultadas

## Documentos de trabajo

- Ley 24.240 de Defensa del Consumidor (texto completo)
- Código Civil y Comercial — Título III: Contratos de consumo

## Tecnologías utilizadas

- **Embeddings**: `intfloat/multilingual-e5-small` (local, sin API)
- **Base vectorial**: ChromaDB en memoria
- **LLM**: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference API
- **Pipeline RAG**: LangChain (LCEL)
- **Interfaz**: Gradio
- **Deploy**: HuggingFace Spaces

## Desarrollado por

Carlos Estigarribia  
Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial — IFTS24  
Buenos Aires, 2026

## ¿Por qué este proyecto?

La motivación surge del contacto directo con profesionales del derecho que 
expresaron frustración con la búsqueda manual en documentación dispersa. 
Este MVP es el primer paso hacia una herramienta orientada a PyMEs jurídicas 
que combine RAG con la API pública del SAIJ para acceso a jurisprudencia 
argentina actualizada — sin depender de servicios externos pagos y respetando 
la confidencialidad de los expedientes.

## Corpus documental incluido

Los PDFs para testear el sistema están disponibles en este repositorio:

- [`Ley Defensa al Consumidor.pdf`](./Ley%20Defensa%20al%20Consumidor.pdf) — Ley 24.240 completa
- [`CCYC - TITULO III - Contratos de Consumo.pdf`](./CCYC%20-%20TITULO%20III%20-%20Contratos%20de%20Consumo.pdf) — Arts. 1092–1122 CCyC

Descargalos y subilos desde la pestaña "Cargar documentos" del Space.

## Space en HuggingFace

🚀 [Probá el sistema en vivo](https://huggingface.co/spaces/CarlosEstigarribia/rag-legal-consumidor-argentino)

## Cómo usar el sistema

1. Abrí la pestaña **"Cargar documentos"**
2. Subí uno o más PDFs jurídicos
3. Hacé clic en **"Indexar documentos"**
4. Andá a la pestaña **"Hacer preguntas"**
5. Escribí tu consulta en lenguaje natural y presioná **"Preguntar"**
