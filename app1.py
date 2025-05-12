import os
import base64
import tempfile
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Dependency imports
import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract  # Added for OCR
from pdf2image import convert_from_path  # Added for PDF to image conversion
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langgraph.graph import StateGraph, END
from neo4j import GraphDatabase

# Environment variables (replace with your own)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBCQPHfPkbhumZ6H7P9DmxAUdlLepCEeuM"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "Ragpdf"
NEO4J_PASSWORD = "tamil@123"

# Initialize Gemini model
gemini_pro = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
gemini_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Neo4j connection setup
class Neo4jDatabase:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        self.driver.close()
        
    def create_constraints(self):
        with self.driver.session() as session:
            # Create constraints and indexes
            session.run("CREATE CONSTRAINT pdf_node_id IF NOT EXISTS FOR (p:PDF) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT page_node_id IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT chunk_node_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT image_node_id IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE")
            session.run("CREATE CONSTRAINT table_node_id IF NOT EXISTS FOR (t:Table) REQUIRE t.id IS UNIQUE")
            session.run("CREATE CONSTRAINT ocr_text_node_id IF NOT EXISTS FOR (o:OCRText) REQUIRE o.id IS UNIQUE")
            
    def add_pdf(self, pdf_id, filename, metadata=None):
        with self.driver.session() as session:
            if not metadata:
                metadata = {}
            session.run(
                "CREATE (p:PDF {id: $id, filename: $filename, metadata: $metadata})",
                id=pdf_id, filename=filename, metadata=metadata
            )
            
    def add_page(self, page_id, pdf_id, page_num, text):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (pdf:PDF {id: $pdf_id})
                CREATE (p:Page {id: $page_id, page_num: $page_num, text: $text})
                CREATE (pdf)-[:HAS_PAGE]->(p)
                """,
                pdf_id=pdf_id, page_id=page_id, page_num=page_num, text=text
            )
            
    def add_chunk(self, chunk_id, page_id, text, embedding):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:Page {id: $page_id})
                CREATE (c:Chunk {id: $chunk_id, text: $text, embedding: $embedding})
                CREATE (p)-[:HAS_CHUNK]->(c)
                """,
                page_id=page_id, chunk_id=chunk_id, text=text, embedding=embedding
            )
            
    def add_image(self, image_id, page_id, image_data, caption, embedding):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:Page {id: $page_id})
                CREATE (i:Image {id: $image_id, image_data: $image_data, caption: $caption, embedding: $embedding})
                CREATE (p)-[:HAS_IMAGE]->(i)
                """,
                page_id=page_id, image_id=image_id, image_data=image_data, caption=caption, embedding=embedding
            )
            
    def add_table(self, table_id, page_id, table_data, caption, embedding):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:Page {id: $page_id})
                CREATE (t:Table {id: $table_id, table_data: $table_data, caption: $caption, embedding: $embedding})
                CREATE (p)-[:HAS_TABLE]->(t)
                """,
                page_id=page_id, table_id=table_id, table_data=table_data, caption=caption, embedding=embedding
            )
            
    def add_ocr_text(self, ocr_id, page_id, image_id, ocr_text, embedding):
        """Add OCR text extracted from an image to Neo4j"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:Page {id: $page_id})
                MATCH (i:Image {id: $image_id})
                CREATE (o:OCRText {id: $ocr_id, text: $ocr_text, embedding: $embedding})
                CREATE (p)-[:HAS_OCR_TEXT]->(o)
                CREATE (i)-[:HAS_OCR_TEXT]->(o)
                """,
                page_id=page_id, image_id=image_id, ocr_id=ocr_id, ocr_text=ocr_text, embedding=embedding
            )

# OCR Utilities
class OCRProcessor:
    def __init__(self, language='eng'):
        """Initialize OCR processor with language setting"""
        self.language = language
        # Ensure pytesseract knows where Tesseract is installed
        # Uncomment and modify if needed:
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
        # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux
    
    def process_image(self, image_path):
        """Extract text from an image using OCR"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.language)
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def process_pdf_page(self, pdf_path, page_number):
        """Convert a PDF page to image and extract text using OCR"""
        try:
            # Convert PDF page to image
            images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)
            if not images:
                return ""
            
            # Save image temporarily
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
                images[0].save(temp_img.name, 'PNG')
                temp_img_path = temp_img.name
            
            # Extract text via OCR
            text = self.process_image(temp_img_path)
            
            # Clean up
            os.unlink(temp_img_path)
            
            return text
        except Exception as e:
            print(f"PDF OCR Error: {e}")
            return ""
    
    def needs_ocr(self, page_text):
        """Determine if a page needs OCR based on extracted text"""
        # If the page has very little text or contains placeholder text like '[?]',
        # it might be a scanned page that needs OCR
        if len(page_text.strip()) < 100 or '[?]' in page_text:
            return True
        return False

# Enhanced PDF Processor with OCR capabilities
class PDFProcessor:
    def __init__(self, neo4j_db, embeddings_model):
        self.neo4j_db = neo4j_db
        self.embeddings_model = embeddings_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.ocr_processor = OCRProcessor()  # Initialize OCR processor
        
    def process_pdf(self, pdf_path):
        # Generate a unique ID for the PDF
        pdf_id = f"pdf_{Path(pdf_path).stem}"
        filename = Path(pdf_path).name
        
        # Add PDF to Neo4j
        self.neo4j_db.add_pdf(pdf_id, filename)
        
        # Process the PDF document
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            # Process page
            page_id = f"{pdf_id}_page{page_num}"
            page_text = page.get_text()
            
            # Check if OCR is needed
            if self.ocr_processor.needs_ocr(page_text):
                ocr_text = self.ocr_processor.process_pdf_page(pdf_path, page_num)
                if ocr_text:
                    # If OCR extracted text, use it instead
                    page_text = ocr_text
            
            self.neo4j_db.add_page(page_id, pdf_id, page_num, page_text)
            
            # Process text chunks
            chunks = self.text_splitter.split_text(page_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{page_id}_chunk{i}"
                embedding = self.embeddings_model.embed_query(chunk)
                self.neo4j_db.add_chunk(chunk_id, page_id, chunk, embedding)
            
            # Process images
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                
                # Save image temporarily to process with OCR and Gemini Vision
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
                    temp_img.write(img_bytes)
                    temp_img_path = temp_img.name
                
                # Generate caption using Gemini Vision
                image_caption = self._generate_image_caption(temp_img_path)
                
                # Store the image in Neo4j
                image_id = f"{page_id}_image{img_idx}"
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                img_embedding = self.embeddings_model.embed_query(image_caption)
                self.neo4j_db.add_image(image_id, page_id, img_base64, image_caption, img_embedding)
                
                # Perform OCR on the image
                ocr_text = self.ocr_processor.process_image(temp_img_path)
                if ocr_text:
                    # Store OCR text in Neo4j
                    ocr_id = f"{image_id}_ocr"
                    ocr_embedding = self.embeddings_model.embed_query(ocr_text)
                    self.neo4j_db.add_ocr_text(ocr_id, page_id, image_id, ocr_text, ocr_embedding)
                
                # Clean up temp file
                os.unlink(temp_img_path)
            
            # Process tables
            tables = self._extract_tables_from_page(page)
            for table_idx, table_data in enumerate(tables):
                table_id = f"{page_id}_table{table_idx}"
                
                # Convert table to string representation
                table_str = self._table_to_string(table_data)
                
                # Generate caption
                table_caption = self._generate_table_caption(table_str)
                
                # Store in Neo4j
                table_embedding = self.embeddings_model.embed_query(table_caption + " " + table_str[:200])
                self.neo4j_db.add_table(table_id, page_id, table_str, table_caption, table_embedding)
    
    def _generate_image_caption(self, image_path):
        """Generate a caption for an image using Gemini Vision"""
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
            
        image = Image.open(image_path)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at describing images from documents. Provide a detailed caption."),
            ("human", "Describe what you see in this image from a document in detail")
        ])
        
        response = gemini_vision.invoke([
            {"type": "text", "text": "Describe what you see in this image from a document in detail"},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"}
        ])
        
        return response.content
    
    def _extract_tables_from_page(self, page):
        """Extract tables from a PDF page"""
        tables = []
        for table in page.find_tables():
            table_data = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                table_data.append(row_data)
            tables.append(table_data)
        return tables
    
    def _table_to_string(self, table_data):
        """Convert a table to a string representation"""
        table_str = ""
        for row in table_data:
            table_str += " | ".join(row) + "\n"
        return table_str
    
    def _generate_table_caption(self, table_str):
        """Generate a caption for a table using Gemini"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at understanding tables from documents. Provide a concise caption."),
            ("human", f"Provide a short descriptive caption for this table:\n{table_str}")
        ])
        
        chain = prompt | gemini_pro | StrOutputParser()
        return chain.invoke({})

# Enhanced RAG pipeline with OCR-aware querying
class RAGPipeline:
    def __init__(self, neo4j_uri, neo4j_username, neo4j_password, embeddings_model):
        self.embeddings = embeddings_model
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        
        # Initialize vector store for text chunks
        self.vector_store = Neo4jVector.from_existing_graph(
            embeddings=embeddings_model,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            index_name="text_embedding",
            node_label="Chunk",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        
        # Initialize vector store for OCR text
        self.ocr_vector_store = Neo4jVector.from_existing_graph(
            embeddings=embeddings_model,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            index_name="ocr_embedding",
            node_label="OCRText",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        
        # Set up retrievers
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        self.ocr_retriever = self.ocr_vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Set up LLM chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant that answers questions about documents.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise."""),
            ("human", "{question}"),
            ("ai", "I'll help answer that based on the document."),
            ("human", "Context: {context} \n\nQuestion: {question}")
        ])
        
        # Create the answer generation chain
        self.answer_chain = create_stuff_documents_chain(gemini_pro, self.prompt)
        
        # Create the retrieval chain
        self.rag_chain = create_retrieval_chain(self.retriever, self.answer_chain)
        
    def query(self, question):
        """Execute a query against the RAG pipeline"""
        result = self.rag_chain.invoke({"question": question})
        return result["answer"]
    
    def query_with_ocr(self, question):
        """Execute a query against both regular text and OCR text"""
        # Get regular text results
        regular_results = self.retriever.invoke(question)
        
        # Get OCR text results
        ocr_results = self.ocr_retriever.invoke(question)
        
        # Combine results - prefer regular text if available
        combined_results = regular_results
        
        # Add OCR results if they seem relevant
        if ocr_results:
            # Use Gemini to determine if OCR results should be included
            ocr_text = "\n".join([doc.page_content for doc in ocr_results])
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are evaluating if OCR text is relevant to a question. Reply with YES or NO only."),
                ("human", f"Question: {question}\n\nOCR Text: {ocr_text}\n\nIs this OCR text relevant to answering the question? Answer YES or NO only.")
            ])
            
            chain = prompt | gemini_pro | StrOutputParser()
            is_relevant = chain.invoke({}).strip().upper()
            
            if is_relevant == "YES":
                combined_results.extend(ocr_results)
        
        # Generate answer with combined results
        context_text = "\n\n".join([doc.page_content for doc in combined_results])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that answers questions about documents using both regular text and OCR-extracted text."),
            ("human", "{question}"),
            ("ai", "I'll help answer that based on the document content."),
            ("human", "Context: {context} \n\nQuestion: {question}")
        ])
        
        chain = prompt | gemini_pro | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context_text})
        
        return answer

# Enhanced LangGraph workflow with OCR awareness
class PDFQueryWorkflow:
    def __init__(self, rag_pipeline, neo4j_db):
        self.rag_pipeline = rag_pipeline
        self.neo4j_db = neo4j_db
        self.workflow = self._build_workflow()
        
    def _build_workflow(self):
        # Define the state
        class State:
            query: str
            query_type: str = None
            context: List[Dict] = None
            answer: str = None
            
        # Define workflow nodes
        def classify_query(state):
            """Classify the query type as text, image, table, ocr, or general"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Classify the query as one of: TEXT (for text content), IMAGE (for image content), TABLE (for table data), OCR (for text in images or scanned pages), or GENERAL (for overall document questions)."),
                ("human", "{query}")
            ])
            
            chain = prompt | gemini_pro | StrOutputParser()
            query_type = chain.invoke({"query": state.query}).strip().upper()
            
            # Ensure valid classification
            if query_type not in ["TEXT", "IMAGE", "TABLE", "OCR", "GENERAL"]:
                query_type = "GENERAL"
                
            return {"query_type": query_type}
            
        def retrieve_text_context(state):
            """Retrieve context for text queries"""
            # Use the vector store retriever (already set up in RAG pipeline)
            result = self.rag_pipeline.retriever.invoke(state.query)
            return {"context": result}
        
        def retrieve_image_context(state):
            """Retrieve image context based on query"""
            # Connect to Neo4j to retrieve relevant images
            with self.neo4j_db.driver.session() as session:
                result = session.run(
                    """
                    MATCH (i:Image)
                    WITH i, i.caption AS caption, i.image_data AS image_data, 
                         gds.similarity.cosine(
                           $query_embedding, 
                           i.embedding
                         ) AS similarity
                    ORDER BY similarity DESC
                    LIMIT 3
                    RETURN caption, image_data, similarity
                    """,
                    query_embedding=self.rag_pipeline.embeddings.embed_query(state.query)
                )
                
                context = []
                for record in result:
                    context.append({
                        "type": "image",
                        "caption": record["caption"],
                        "image_data": record["image_data"],
                        "similarity": record["similarity"]
                    })
                    
            return {"context": context}
            
        def retrieve_table_context(state):
            """Retrieve table context based on query"""
            with self.neo4j_db.driver.session() as session:
                result = session.run(
                    """
                    MATCH (t:Table)
                    WITH t, t.caption AS caption, t.table_data AS table_data,
                         gds.similarity.cosine(
                           $query_embedding, 
                           t.embedding
                         ) AS similarity
                    ORDER BY similarity DESC
                    LIMIT 3
                    RETURN caption, table_data, similarity
                    """,
                    query_embedding=self.rag_pipeline.embeddings.embed_query(state.query)
                )
                
                context = []
                for record in result:
                    context.append({
                        "type": "table",
                        "caption": record["caption"],
                        "table_data": record["table_data"],
                        "similarity": record["similarity"]
                    })
                    
            return {"context": context}
        
        def retrieve_ocr_context(state):
            """Retrieve OCR text context based on query"""
            with self.neo4j_db.driver.session() as session:
                result = session.run(
                    """
                    MATCH (o:OCRText)
                    WITH o, o.text AS text,
                         gds.similarity.cosine(
                           $query_embedding, 
                           o.embedding
                         ) AS similarity
                    ORDER BY similarity DESC
                    LIMIT 5
                    RETURN text, similarity
                    """,
                    query_embedding=self.rag_pipeline.embeddings.embed_query(state.query)
                )
                
                context = []
                for record in result:
                    context.append({
                        "type": "ocr",
                        "text": record["text"],
                        "similarity": record["similarity"]
                    })
                    
                # Also get associated images for context
                if context:
                    image_result = session.run(
                        """
                        MATCH (o:OCRText)<-[:HAS_OCR_TEXT]-(i:Image)
                        WHERE o.text IN $ocr_texts
                        RETURN i.caption AS caption, i.image_data AS image_data
                        LIMIT 3
                        """,
                        ocr_texts=[item["text"] for item in context]
                    )
                    
                    for record in image_result:
                        context.append({
                            "type": "image",
                            "caption": record["caption"],
                            "image_data": record["image_data"],
                            "similarity": 1.0  # Associated images are highly relevant
                        })
                    
            return {"context": context}
        
        def retrieve_general_context(state):
            """Retrieve mixed context (text, images, tables, OCR) for general queries"""
            # This combines text, image, table, and OCR retrievals
            text_context = retrieve_text_context(state).get("context", [])
            
            # Get a smaller number of images, tables, and OCR for general queries
            with self.neo4j_db.driver.session() as session:
                # Get images
                image_result = session.run(
                    """
                    MATCH (i:Image)
                    WITH i, i.caption AS caption, i.image_data AS image_data, 
                         gds.similarity.cosine(
                           $query_embedding, 
                           i.embedding
                         ) AS similarity
                    ORDER BY similarity DESC
                    LIMIT 1
                    RETURN caption, image_data, similarity
                    """,
                    query_embedding=self.rag_pipeline.embeddings.embed_query(state.query)
                )
                
                # Get tables
                table_result = session.run(
                    """
                    MATCH (t:Table)
                    WITH t, t.caption AS caption, t.table_data AS table_data,
                         gds.similarity.cosine(
                           $query_embedding, 
                           t.embedding
                         ) AS similarity
                    ORDER BY similarity DESC
                    LIMIT 1
                    RETURN caption, table_data, similarity
                    """,
                    query_embedding=self.rag_pipeline.embeddings.embed_query(state.query)
                )
                
                # Get OCR text
                ocr_result = session.run(
                    """
                    MATCH (o:OCRText)
                    WITH o, o.text AS text,
                         gds.similarity.cosine(
                           $query_embedding, 
                           o.embedding
                         ) AS similarity
                    ORDER BY similarity DESC
                    LIMIT 1
                    RETURN text, similarity
                    """,
                    query_embedding=self.rag_pipeline.embeddings.embed_query(state.query)
                )
                
                image_context = []
                for record in image_result:
                    image_context.append({
                        "type": "image",
                        "caption": record["caption"],
                        "image_data": record["image_data"],
                        "similarity": record["similarity"]
                    })
                
                table_context = []
                for record in table_result:
                    table_context.append({
                        "type": "table",
                        "caption": record["caption"],
                        "table_data": record["table_data"],
                        "similarity": record["similarity"]
                    })
                    
                ocr_context = []
                for record in ocr_result:
                    ocr_context.append({
                        "type": "ocr",
                        "text": record["text"],
                        "similarity": record["similarity"]
                    })
            
            # Combine all contexts
            combined_context = text_context + image_context + table_context + ocr_context
            
            # Sort by similarity if available
            combined_context.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            return {"context": combined_context[:5]}  # Limit to top 5 most relevant pieces
        
        def generate_answer(state):
            """Generate an answer based on query type and context"""
            if state.query_type == "IMAGE":
                # Format image context for LLM
                context_str = "\n\n".join([
                    f"Image Caption: {item['caption']}" for item in state.context
                ])
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an assistant that answers questions about document images."),
                    ("human", "{query}"),
                    ("ai", "I'll help answer that based on the images in the document."),
                    ("human", "Image Contexts:\n{context}\n\nQuestion: {query}")
                ])
                
                chain = prompt | gemini_pro | StrOutputParser()
                answer = chain.invoke({"query": state.query, "context": context_str})
                
            elif state.query_type == "TABLE":
                # Format table context for LLM
                context_str = "\n\n".join([
                    f"Table Caption: {item['caption']}\nTable Data:\n{item['table_data']}" 
                    for item in state.context
                ])
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an assistant that answers questions about tables in documents."),
                    ("human", "{query}"),
                    ("ai", "I'll help answer that based on the tables in the document."),
                    ("human", "Table Contexts:\n{context}\n\nQuestion: {query}")
                ])
                
                chain = prompt | gemini_pro | StrOutputParser()
                answer = chain.invoke({"query": state.query, "context": context_str})
                
            elif state.query_type == "OCR":
                # Format OCR context for LLM
                context_str = "\n\n".join([
                    f"OCR Text: {item['text']}" if item['type'] == 'ocr' else f"Related Image: {item['caption']}"
                    for item in state.context
                ])
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an assistant that answers questions about text extracted from images and scanned documents using OCR."),
                    ("human", "{query}"),
                    ("ai", "I'll help answer that based on the OCR-extracted text from the document."),
                    ("human", "OCR Contexts:\n{context}\n\nQuestion: {query}")
                ])
                
                chain = prompt | gemini_pro | StrOutputParser()
                answer = chain.invoke({"query": state.query, "context": context_str})
                        
        def generate_answer(state):
            """Generate an answer based on query type and context"""
            if state.query_type == "IMAGE":
                # Format image context for LLM
                context_str = "\n\n".join([
                    f"Image Caption: {item['caption']}" for item in state.context
                ])
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an assistant that answers questions about document images."),
                    ("human", "{query}"),
                    ("ai", "I'll help answer that based on the images in the document."),
                    ("human", "Image Contexts:\n{context}\n\nQuestion: {query}")
                ])
                
                chain = prompt | gemini_pro | StrOutputParser()
                answer = chain.invoke({"query": state.query, "context": context_str})
                
            elif state.query_type == "TABLE":
                # Format table context for LLM
                context_str = "\n\n".join([
                    f"Table Caption: {item['caption']}\nTable Data:\n{item['table_data']}" 
                    for item in state.context
                ])
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an assistant that answers questions about tables in documents."),
                    ("human", "{query}"),
                    ("ai", "I'll help answer that based on the tables in the document."),
                    ("human", "Table Contexts:\n{context}\n\nQuestion: {query}")
                ])
                
                chain = prompt | gemini_pro | StrOutputParser()
                answer = chain.invoke({"query": state.query, "context": context_str})
                
            elif state.query_type == "GENERAL":
                # Format mixed context for LLM
                context_parts = []
                
                for item in state.context:
                    if item.get("type") == "image":
                        context_parts.append(f"Image Caption: {item['caption']}")
                    elif item.get("type") == "table":
                        context_parts.append(f"Table Caption: {item['caption']}\nTable Data:\n{item['table_data']}")
                    else:
                        # Text document
                        context_parts.append(f"Document Text: {item.page_content}")
                
                context_str = "\n\n".join(context_parts)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an assistant that answers questions about documents including text, images, and tables."),
                    ("human", "{query}"),
                    ("ai", "I'll help answer that based on the document content."),
                    ("human", "Context:\n{context}\n\nQuestion: {query}")
                ])
                
                chain = prompt | gemini_pro | StrOutputParser()
                answer = chain.invoke({"query": state.query, "context": context_str})
                
            else:  # TEXT type - use the standard RAG pipeline
                answer = self.rag_pipeline.query(state.query)
                
            return {"answer": answer}
        
        # Define the graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("classify_query", classify_query)
        workflow.add_node("retrieve_text_context", retrieve_text_context)
        workflow.add_node("retrieve_image_context", retrieve_image_context)
        workflow.add_node("retrieve_table_context", retrieve_table_context)
        workflow.add_node("retrieve_general_context", retrieve_general_context)
        workflow.add_node("generate_answer", generate_answer)
        
        # Add edges
        workflow.set_entry_point("classify_query")
        
        workflow.add_conditional_edges(
            "classify_query",
            lambda state: state.query_type,
            {
                "TEXT": "retrieve_text_context",
                "IMAGE": "retrieve_image_context",
                "TABLE": "retrieve_table_context",
                "GENERAL": "retrieve_general_context"
            }
        )
        
        workflow.add_edge("retrieve_text_context", "generate_answer")
        workflow.add_edge("retrieve_image_context", "generate_answer")
        workflow.add_edge("retrieve_table_context", "generate_answer")
        workflow.add_edge("retrieve_general_context", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def process_query(self, query):
        """Process a query through the workflow"""
        result = self.workflow.invoke({"query": query})
        return result.answer

# Main application
class PDFQueryApp:
    def _init_(self):
        # Initialize Neo4j
        self.neo4j_db = Neo4jDatabase(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        self.neo4j_db.create_constraints()
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(self.neo4j_db, embeddings)
        
        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, embeddings)
        
        # Initialize workflow
        self.workflow = PDFQueryWorkflow(self.rag_pipeline, self.neo4j_db)
        
    def upload_pdf(self, pdf_path):
        """Upload and process a PDF file"""
        self.pdf_processor.process_pdf(pdf_path)
        return f"PDF {Path(pdf_path).name} processed successfully."
        
    def query_pdf(self, query):
        """Query the processed PDF data"""
        return self.workflow.process_query(query)

@app.route('/query', methods=['POST'])
def query_pdf():
    global pdf_file_path, conversation_history, pdf_history
    if pdf_file_path is None:
        return jsonify({'error': 'Please upload a PDF first'}), 400

    data = request.get_json()
    user_query = data.get('query')
    include_beyond_pdf = data.get('includeBeyondPDF', False)

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        with open(pdf_file_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        # For both cases, use a prompt that first focuses on PDF content
        if include_beyond_pdf:
            enhanced_prompt = f"""
            Answer the following question about the PDF document:
            
            Question: {user_query}
            
            Important instructions:
            1. Your response MUST be structured in TWO clearly separated sections:
               
               PDF CONTENT:
               - Include ALL relevant information explicitly found in the PDF
               - If no information is found in the PDF about this query, state this clearly
               
               ADDITIONAL RELEVANT INFORMATION:

               
            
            2. Both sections are REQUIRED and must be clearly labeled with the headings above
            3. Format your answer with proper spacing between sections
            """
        else:
            enhanced_prompt = f"""
            Answer the following question using ONLY information found in the PDF document:
            
            Question: {user_query}
            
            Important instructions:
            1. Only use information explicitly stated in the provided PDF.

            """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type='application/pdf',
                ),
                enhanced_prompt,
            ],
        )

        formatted_response = format_as_points(response.text)
        raw_response = response.text
        
        # Store the question and answer in both conversation histories
        qa_entry = {
            "question": user_query,
            "answer": raw_response,
            "includedBeyondPDF": include_beyond_pdf,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        conversation_history.append(qa_entry)
        
        # Add to PDF-specific history
        pdf_name = os.path.basename(pdf_file_path)
        if pdf_name not in pdf_history:
            pdf_history[pdf_name] = []
        pdf_history[pdf_name].append(qa_entry)

        # Save complete conversation history to files
        save_conversation_history()

        return jsonify({'response': formatted_response}), 200

    except FileNotFoundError:
        return jsonify({'error': f"File not found at {pdf_file_path}"}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/download/<format_type>')
def download_output(format_type):
    """Download the entire conversation history in the specified format"""
    if format_type == 'txt':
        output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'conversation.txt')
        if not os.path.exists(output_file_path):
            return jsonify({'error': 'No conversation history found. Generate a response first.'}), 404
        return send_from_directory(directory=app.config['UPLOAD_FOLDER'], path='conversation.txt', as_attachment=True)
    elif format_type == 'json':
        output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'conversation.json')
        if not os.path.exists(output_file_path):
            return jsonify({'error': 'No JSON conversation history found. Generate a response first.'}), 404
        return send_from_directory(directory=app.config['UPLOAD_FOLDER'], path='conversation.json', as_attachment=True)
    else:
        return jsonify({'error': 'Invalid format type'}), 400

# Example usage
if _name_ == "_main_":
    app = PDFQueryApp()