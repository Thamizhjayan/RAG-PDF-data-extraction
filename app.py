import sys
from google import genai
from google.genai import types
import pathlib
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import re
import json
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBCQPHfPkbhumZ6H7P9DmxAUdlLepCEeuM"  # Replace with your actual API key

# Initialize Google Generative AI client
client = genai.Client()

UPLOAD_FOLDER = 'uploads'  # Directory to store uploaded PDFs
TEMPLATE_FOLDER = 'templates'  # Directory for HTML templates
STATIC_FOLDER = 'static'  # Directory for static files (CSS, JS)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATE_FOLDER'] = TEMPLATE_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

pdf_file_path = None  # Store the path of the uploaded PDF
conversation_history = []  # Store the history of questions and answers
pdf_history = {}  # Store conversations by PDF filename

# Function to format response as clean points without numbers or symbols
def format_as_points(text):
    # Remove the "ANS:" prefix if present
    text = re.sub(r'^ANS:\s*', '', text.strip())
    
    # Create HTML structure
    formatted_html = '<div class="clean-list">'
    
    # Check if the response indicates information not found in PDF
    if "information not found in the pdf" in text.lower() or "not mentioned in the pdf" in text.lower():
        formatted_html += f'<p class="not-found-message">{text}</p>'
        formatted_html += '</div>'
        return formatted_html
    
    # Split text into sections (PDF info vs Beyond PDF info)
    if "BEYOND PDF:" in text:
        parts = text.split("BEYOND PDF:", 1)
        pdf_content = parts[0].strip()
        beyond_content = parts[1].strip()
        
        # Process PDF content
        formatted_html += process_content_section(pdf_content, "pdf-content")
        
        # Add a divider with improved styling for the section title
        formatted_html += '<div class="section-divider"><h2 class="section-title">Additional Relevant Information</h2></div>'
        
        # Process Beyond PDF content
        formatted_html += process_content_section(beyond_content, "beyond-content")
    else:
        # Just regular content
        formatted_html += process_content_section(text, "pdf-content")
    
    formatted_html += '</div>'
    return formatted_html

def process_content_section(text, section_class):
    """Process a section of content into clean points with improved title formatting"""
    section_html = f'<div class="{section_class}">'
    
    # Split text into paragraphs first to better identify section headers
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Check if this paragraph is a section title/header
        # Look for patterns like "Section Title:" or "SECTION TITLE" or "1. SECTION TITLE"
        if re.match(r'^(?:\d+\.\s*)?[A-Z][A-Z\s\d\:]+\s*(?:\:|\-)?$', paragraph) or len(paragraph) < 60 and paragraph.isupper():
            # This is likely a title - format as blue and bold
            clean_title = re.sub(r'^\s*\d+\.\s*', '', paragraph).strip()
            clean_title = re.sub(r'[\:\-]$', '', clean_title).strip()
            section_html += f'<h2 class="content-title">{clean_title}</h2>'
            continue
        
        # Check if this is a figure title
        if re.match(r'^Figure \d+:', paragraph):
            section_html += f'<h3 class="figure-title">{paragraph}</h3>'
            continue
            
        # Process regular content paragraphs
        # Split into sentences for more granular processing
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        # Group sentences into 2-3 line points
        current_point = ""
        line_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Calculate approximate number of lines this sentence would take
            estimated_lines = max(1, len(sentence) // 80)  # Assuming ~80 chars per line
            
            # If adding this sentence would exceed 3 lines, start a new point
            if line_count > 0 and line_count + estimated_lines > 3:
                # Remove any numbering, bullet points, etc.
                clean_point = re.sub(r'^\s*\d+\.\s*', '', current_point).strip()
                clean_point = re.sub(r'^\s*[•\-\*\?\★]\s*', '', clean_point).strip()
                clean_point = re.sub(r'^\s*Q[:.]\s*', '', clean_point).strip()
                
                # Process bold text
                clean_point = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', clean_point)
                
                section_html += f'<div class="point-item">{clean_point}</div>'
                current_point = sentence
                line_count = estimated_lines
            else:
                # Add to current point
                if current_point:
                    current_point += " " + sentence
                else:
                    current_point = sentence
                line_count += estimated_lines
        
        # Add any remaining content as a final point
        if current_point:
            # Remove any numbering, bullet points, etc.
            clean_point = re.sub(r'^\s*\d+\.\s*', '', current_point).strip()
            clean_point = re.sub(r'^\s*[•\-\*\?\★]\s*', '', clean_point).strip()
            clean_point = re.sub(r'^\s*Q[:.]\s*', '', clean_point).strip()
            
            # Process bold text
            clean_point = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', clean_point)
            
            section_html += f'<div class="point-item">{clean_point}</div>'
    
    section_html += '</div>'
    return section_html

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/show_modified')
def show_modified():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global pdf_file_path
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        pdf_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_file_path)
        # Store new conversation in pdf_history instead of clearing
        if filename not in pdf_history:
            pdf_history[filename] = []
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
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
               - Provide additional relevant information not found in the PDF
               - This section should complement and expand on the PDF content
               - Make sure this information relates directly to the query and the PDF's topic
               
            
            2. Both sections are REQUIRED and must be clearly labeled with the headings above
            3. Format your answer with proper spacing between sections
            """
        else:
            enhanced_prompt = f"""
            Answer the following question using ONLY information found in the PDF document:
            
            Question: {user_query}
            
            Important instructions:
            1. Only use information explicitly stated in the provided PDF.
            2. If the information is not in the PDF, clearly state: "This information is not found in the PDF."
            3. Do not make up or infer information not present in the document.
            4. Format your answer with clear section titles in ALL CAPS.
            5. Use proper spacing between different sections of content.
            6. If the question is partially addressed in the PDF, provide the available information and note what is missing.
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
    
def save_conversation_history():
    """Save the conversation history to both TXT and JSON files"""
    global conversation_history, pdf_file_path, pdf_history
    
    if not conversation_history:
        return
    
    # Create a text version of all conversations
    text_content = "PDF QA Session History\n"
    text_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add each PDF's conversations to the text file
    for pdf_name, qa_history in pdf_history.items():
        text_content += f"=== PDF: {pdf_name} ===\n\n"
        
        for i, qa in enumerate(qa_history, 1):
            beyond_info = "[With beyond-PDF info]" if qa.get('includedBeyondPDF') else "[PDF-only info]"
            text_content += f"Q{i}: {qa['question']} {beyond_info}\n"
            text_content += f"A{i}: {qa['answer']}\n"
            text_content += f"Time: {qa['timestamp']}\n\n"
        
        text_content += "=" * 50 + "\n\n"
    
    # Save as text file
    output_txt_path = os.path.join(app.config['UPLOAD_FOLDER'], 'conversation.txt')
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(text_content)
    
    # Save as JSON file with complete history
    json_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pdf_conversations": pdf_history
    }
    output_json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'conversation.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
        
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

# Add route to serve CSS file for styling the output
@app.route('/static/styles.css')
def serve_css():
    # Create CSS file if it doesn't exist
    css_file_path = os.path.join(app.config['STATIC_FOLDER'], 'styles.css')
    if not os.path.exists(css_file_path):
        with open(css_file_path, 'w', encoding='utf-8') as f:
            f.write("""
/* Main container styling */
.clean-list {
    font-family: 'Arial', sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    width: 100%;
    box-sizing: border-box;
    margin: 0 auto;
}

/* Section title styling */
.content-title {
    color: #0078d4;
    font-weight: bold;
    font-size: 1.4em;
    margin-top: 20px;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #ddd;
    word-wrap: break-word;
}

/* Section divider styling */
.section-divider {
    margin: 25px 0;
    text-align: center;
    position: relative;
    width: 100%;
}

/* Section title styling for "Additional Relevant Information" */
.section-title {
    color: #0078d4;
    font-weight: bold;
    font-size: 1em;
    background-color: #f9f9f9;
    display: inline-block;
    padding: 0 15px;
    position: relative;
    max-width: 90%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.section-divider:before {
    content: "";
    position: absolute;
    top: 50%;
    left: 0;
    width: 100%;
    height: 1px;
    background-color: #ddd;
    z-index: -1;
}

/* Figure title styling */
.figure-title {
    color: #444;
    font-style: italic;
    margin: 15px 0 10px 0;
    word-wrap: break-word;
}

/* Point item styling */
.point-item {
    background-color: #fff;
    padding: 12px 15px;
    margin-bottom: 10px;
    border-left: 3px solid #0078d4;
    border-radius: 0 4px 4px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    word-wrap: break-word;
}

/* Not found message styling */
.not-found-message {
    color: #721c24;
    background-color: #f8d7da;
    padding: 15px;
    border-radius: 4px;
    border-left: 5px solid #f5c6cb;
    word-wrap: break-word;
}

/* PDF content specific styling */
.pdf-content {
    margin-bottom: 20px;
    width: 100%;
}

/* Beyond PDF content specific styling */
.beyond-content {
    background-color: #f0f7ff;
    padding: 15px;
    border-radius: 6px;
    width: 100%;
    box-sizing: border-box;
}

/* Media Queries for Responsiveness */
@media (max-width: 992px) {
    .clean-list {
        padding: 15px;
    }
    
    .content-title {
        font-size: 1.3em;
    }
    
    .section-title {
        max-width: 85%;
    }
}

@media (max-width: 768px) {
    .clean-list {
        padding: 12px;
    }
    
    .content-title {
        font-size: 1.2em;
        margin-top: 15px;
    }
    
    .point-item {
        padding: 10px 12px;
    }
    
    .section-divider {
        margin: 20px 0;
    }
    
    .section-title {
        font-size: 0.9em;
        max-width: 80%;
    }
    
    .beyond-content {
        padding: 12px;
    }
}

@media (max-width: 480px) {
    .clean-list {
        padding: 10px;
        border-radius: 6px;
    }
    
    .content-title {
        font-size: 1.1em;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    
    .point-item {
        padding: 8px 10px;
        margin-bottom: 8px;
    }
    
    .section-divider {
        margin: 15px 0;
    }
    
    .section-title {
        font-size: 0.85em;
        padding: 0 10px;
        max-width: 75%;
    }
    
    .figure-title {
        margin: 10px 0 8px 0;
    }
    
    .not-found-message {
        padding: 10px;
    }
    
    .beyond-content {
        padding: 10px;
        border-radius: 4px;
    }
}
            """)
    return send_from_directory(directory=app.config['STATIC_FOLDER'], path='styles.css')

if __name__ == '__main__':
    app.run(debug=True, port=5000)