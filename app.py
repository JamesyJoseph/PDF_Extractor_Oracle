from flask import Flask, render_template, request, redirect, url_for, flash
import os
import fitz  
from sentence_transformers import SentenceTransformer
import numpy as np
import oracledb
import re
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'source'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


connection = oracledb.connect(
    user="pdf_data", password="pdf", dsn="localhost/FREEPDB1"
)
cursor = connection.cursor()


model = SentenceTransformer('gtr-t5-base')

def extract_text(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def extract_structured_data(text):
    data = {
        'bhk': re.search(r'(\d+ ?BHK)', text, re.IGNORECASE),
        'area': re.search(r'(\d{3,5}) ?sq\.? ?ft', text, re.IGNORECASE),
        'price': re.search(r'₹ ?[\d,]+(?: ?Lakh| ?Crore)?', text),
        'floor': re.search(r'Floor:? ?(\d+)', text, re.IGNORECASE),
    }
    return {k: (m.group(1) if m else "") for k, m in data.items()}

def compute_embedding(text):
    vector = model.encode(text, normalize_embeddings=True).tolist()
    return vector + [0.0] * (1024 - len(vector)) 


@app.route('/', methods=['GET', 'POST'])
def home():
    query = ''
    results = []
    
    previews = []

    if request.method == 'POST':
        try:
            if 'pdf_files' in request.files:
                files = request.files.getlist('pdf_files')
                for file in files:
                    if file and file.filename.endswith('.pdf'):
                        filename = file.filename
                        path = os.path.join(UPLOAD_FOLDER, filename)
                        file.save(path)
                        text = extract_text(path)
                        vec = compute_embedding(text)
                        structured = extract_structured_data(text)
                        
                        cursor.setinputsizes(None, None, oracledb.DB_TYPE_VECTOR, None, None, None, None)

                        cursor.execute("""
                                INSERT INTO pdf_data ( filename, text, embedding, bhk, area, price, floor) 
                                VALUES (:1, :2, :3, :4, :5, :6, :7)
                        """, [
                            filename, text, vec,
                            structured['bhk'], structured['area'],
                            structured['price'], structured['floor']
                        ])
                        connection.commit()

                flash("PDF(s) uploaded and processed successfully.")

            elif 'query' in request.form:
                query = request.form['query']
                query_vec = compute_embedding(query)

                cursor.setinputsizes( oracledb.DB_TYPE_VECTOR)

                cursor.execute("""
                    SELECT id, filename, text, bhk, area, price, floor,
                        VECTOR_DISTANCE(embedding, :1) AS score
                    FROM pdf_data
                    ORDER BY score
                    FETCH FIRST 10 ROWS ONLY
                """, [query_vec])

                results = []
                for row in cursor.fetchall():
                    doc_id, filename, text, bhk, area, price, floor, score = row

                    snippet = ''
                    text_str = text.read() if hasattr(text, "read") else str(text)
                    text_lower = text_str.lower()

                    query_lower = query.lower()

                    match_pos = text_lower.find(query_lower)
                    if match_pos != -1:
                        start = max(0, match_pos - 50)
                        end = min(len(text_str), match_pos + len(query) + 50)
                        snippet_raw = text_str[start:end]
                        snippet = snippet_raw.replace(text_str[match_pos:match_pos+len(query)],
                                                    f"<mark>{text_str[match_pos:match_pos+len(query)]}</mark>")
                    else:
                        snippet = text_str[:200] + "..."


                    results.append({
                        'filename': filename,
                        'score': f"{1 / (1 + score):.3f}",
                        'snippet': snippet,
                        'bhk': bhk, 'area': area,
                        'price': price, 'floor': floor,
                        'link': url_for('uploaded_file', filename=filename)
                    })

                results = sorted(results, key=lambda x: float(x['score']), reverse=True)

            cursor.execute("SELECT filename, text FROM pdf_data")
            for row in cursor.fetchall():
                filename, text = row
                text_str = text.read() if hasattr(text, "read") else str(text)
                previews.append({
                    'filename': filename,
                    'snippet': text_str[:300] + "..." if text_str else "",
                    'link': url_for('uploaded_file', filename=filename)
                })



        except Exception as e:
            flash(f"An error occurred: {e}")

    return render_template('home.html', query=query, results=results, previews=previews)

@app.route('/source/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=os.path.join(UPLOAD_FOLDER, filename)))

if __name__ == '__main__':
    app.run(debug=True)
