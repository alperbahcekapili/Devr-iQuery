import json
import re
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

import requests
from src.vector.indexer import ReRanker
from src.vector.indexer import FineGrainedReRanker
from src.vector.bm25 import clean_text, create_index
from src.data.reader import parse_data
from src.vector.indexer import ReRanker
import pyterrier as pt
import pandas as pd
import copy
import time

print("Starting UI")
index_folder = "./index"

print("Parsing data")
data = parse_data()
docs_df = pd.DataFrame(data["documents"].values())

queries_d = copy.deepcopy(data["topics"])
for k, v in data["qrels"].items():
    queries_d[k].update(data["qrels"][k])
queries = list(map(lambda x: f"{x['number']}<>{x['description']}", queries_d.values()))

# Check if index already exists
import os
if not pt.started():
    pt.init()


index_creation_duration = 0.0
faiss_retrieval_duration = 0.0
bm25_retrieval_duration = 0.0
bm25_index_creation_duration = 0.0

if not os.path.exists(index_folder):
    print("Creating new index")
    start_time = time.time()
    indexref = create_index(docs_df, index_folder=index_folder)
    bm25_index_creation_duration = time.time() - start_time
else:
    print("Loading existing index")
    indexref = pt.IndexRef.of(index_folder)


root = tk.Tk()
selected_predefined_query = ""
is_predefined_query = False
predefined_query_number = ""
# Set the default font for all widgets
default_font = tkfont.nametofont("TkDefaultFont")
default_font.configure(family="Arial", size=14, weight="normal")


root.title("Three Column UI Example")

# Header
header = tk.Label(root, text="Devr-i Query Retriever", font=("Arial", 18, "bold"), pady=20)
header.grid(row=0, column=0, columnspan=3, sticky="ew")

# Configure grid weights for responsive columnsintfloat/e5-mistral-7b-instruct

# Column 1
col1 = tk.Frame(root, bg="#e0e0e0", padx=10, pady=10)
col1.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

# Dropdown list (first row)
dropdown_label = tk.Label(col1, text="Embedding Model")
dropdown_label.grid(row=0, column=1, sticky="w", padx=5)
dropdown_var = tk.StringVar()
dropdown = ttk.Combobox(col1, textvariable=dropdown_var, values=[
    "all-MiniLM-L6-v2", 
    "BERT", 
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B", 
    "Qwen/Qwen3-Embedding-8B"
])

dropdown.grid(row=0, column=0, sticky="ew", pady=2)

# Numeric input (second row)
num_var1 = tk.DoubleVar()
num_entry1 = ttk.Entry(col1, textvariable=num_var1)
num_entry1.grid(row=1, column=0, sticky="ew", pady=2)
num_entry1_label = tk.Label(col1, text="BM25 Retrieval Limit:")
num_entry1_label.grid(row=1, column=1, sticky="w", padx=5)

# Numeric input (third row)
num_var2 = tk.DoubleVar()
num_entry2 = ttk.Entry(col1, textvariable=num_var2)
num_entry2.grid(row=2, column=0, sticky="ew", pady=2)
num_entry2_label = tk.Label(col1, text="Total #of Requested Docs: ")
num_entry2_label.grid(row=2, column=1, sticky="w", padx=5)

# Free text area (fourth row)
text_label = tk.Label(col1, text="Query Input")
text_label.grid(row=3, column=0, sticky="w", pady=(10, 0))
text_area = tk.Text(col1, height=15, width=40, wrap="word")
text_area.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)

# Checkbox (fifth row)
use_checks_var = tk.BooleanVar(value=False)
use_checks = ttk.Checkbutton(col1, text="Using Predefined Queries", variable=use_checks_var)
use_checks.grid(row=5, column=0, columnspan=2, sticky="w", pady=5)

use_finegrained_var = tk.BooleanVar(value=False)
use_finegrained = ttk.Checkbutton(col1, text="Use Fine-grained Reranking", variable=use_finegrained_var)
use_finegrained.grid(row=6, column=0, columnspan=2, sticky="w", pady=5)

# Buttons (sixth row)
def open_window1():
    win = tk.Toplevel(root)
    win.title("Predefined Queries")
    
    # Create a scrollable canvas
    canvas = tk.Canvas(win)
    scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Pack the canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    scrollbar.pack(side="right", fill="y")
    
    # Create clickable labels for each query in boxes
    for query in queries:
        # Create a frame for each query with border and margin
        query_frame = tk.Frame(scrollable_frame, relief="solid", borderwidth=1, padx=10, pady=5)
        query_frame.pack(fill="x", pady=5, padx=5)
        
        lbl = tk.Label(query_frame, text=query, cursor="hand2", wraplength=400, justify="left")
        lbl.pack(fill="x", pady=5)
        
        # Bind click event to set the selected query
        lbl.bind("<Button-1>", lambda e, q=query: set_selected_query(q))
    
    def set_selected_query(query):
        global selected_predefined_query, is_predefined_query, predefined_query_number
        selected_predefined_query = query
        is_predefined_query = True
        predefined_query_number = query.split("<>")[0]
        text_area.delete("1.0", tk.END)
        text_area.insert("1.0", query.split("<>")[1])
        win.destroy()

        

def open_window3():
    process_query()

button_frame = tk.Frame(col1)
button_frame.grid(row=7, column=0, columnspan=2, pady=10, sticky="ew")
btn1 = ttk.Button(button_frame, text="Predefined Queries", command=open_window1)

btn3 = ttk.Button(button_frame, text="Run Query", command=open_window3)
btn1.pack(side="left", expand=True, fill="x", padx=2)

btn3.pack(side="left", expand=True, fill="x", padx=2)

# Query and Indexing Stats header (last row)
stats_header = tk.Label(col1, text="Query and Indexing Stats", font=("Arial", 12, "bold"), pady=10)
stats_header.grid(row=8, column=0, columnspan=2, sticky="w")

# Stats value (below header)
stats_value = tk.StringVar(value="""

Faiss Index Creation Duration: 
Faiss Retrieval Duration:
BM25 Retrieval Duration:
BM25 Index Creation Duration: 
Query Metrics: Precision etc. 

""")
stats_label = tk.Label(col1, textvariable=stats_value, bg="#e0e0e0", anchor="w", justify="left")
stats_label.grid(row=9, column=0, columnspan=2, sticky="ew")

# Add new text box for query results
llm_response_text = tk.Text(col1, height=5, width=40, wrap="word")
llm_response_text.grid(row=10, column=0, columnspan=2, sticky="ew", pady=10)
llm_response_text.insert("1.0", "LLM answers results will appear here...")
llm_response_text.config(state="disabled")  # Make it read-only

def update_llm_response(text):
    global llm_response_text
    
    
    # Set up OpenAI API key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return
    
    # System and user prompts
    system_prompt = """
    You are a helpful assistant that can answer questions and help with tasks.
    You are given a query and a list of relevant documents.
    You need to answer the query based on the documents.
    You need to return the answer in a concise and informative manner.
    You need to return the answer in a markdown format.
    You need to return the answer in a concise and informative manner.
"""  # Replace with actual system prompt
    user_query = text  # Using the text parameter passed to the function
    

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {
        'Content-Type': 'application/json'
    }


    data = {
      "contents": [
          {
              "role": "user",  # This is your actual user query
              "parts": [
                  {
                      "text": f"System Prompt: {system_prompt} \n User Query: {user_query}"
                  }
              ]
          }
      ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    response_json = response.json()
    generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
    text = generated_text
    llm_response_text.config(state="normal")
    llm_response_text.delete("1.0", tk.END)
    llm_response_text.insert("1.0", text)
    llm_response_text.config(state="disabled")

def show_full_content(title, content):
    win = tk.Toplevel(root)
    win.title(title)
    text = tk.Text(win, wrap="word", width=80, height=30)
    text.insert("1.0", content)
    text.config(state="disabled")
    text.pack(expand=True, fill="both", padx=10, pady=10)

def display_list_with_status(parent, header_text, items, correctness):
    header = tk.Label(parent, text=header_text, font=("Arial", 12, "bold"), pady=10, bg=parent["bg"])
    header.pack(anchor="w", fill="x")
    # Create a canvas with scrollbar
    canvas = tk.Canvas(parent)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Pack the canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Add items to the scrollable frame
    for i, (item, correct) in enumerate(zip(items, correctness)):
        color = "#b6fcb6" if correct else "#ffb3b3"  # green or red
        print(item)
        short_text = item[:200] + ("..." if len(item) > 200 else "")
        lbl = tk.Label(scrollable_frame, text=short_text, bg=color, anchor="w", padx=5, pady=2, cursor="hand2", wraplength=400, justify="left")
        lbl.pack(fill="x", pady=1)
        lbl.bind("<Button-1>", lambda e, t=header_text, c=item: show_full_content(t, c))

def clear_list(parent, header_text):
    # Find and destroy all widgets in the parent frame
    for widget in parent.winfo_children():
        if isinstance(widget, (tk.Canvas, tk.Label, ttk.Scrollbar)):
            widget.destroy()

# Example data for columns 2 and 3
col2_items = ["""Doc1:""", "Doc2: Methods", "Doc3: Results"]
col2_correctness = [True, False, True]

col3_items = ["Query1: What is AI?", "Query2: Define ML", "Query3: NLP tasks"]
col3_correctness = [True, True, True]

# Column 2
col2 = tk.Frame(root, bg="#c0c0c0", padx=10, pady=10)
col2.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
display_list_with_status(col2, "Retrieved Documents", col2_items, col2_correctness)

# Column 3
col3 = tk.Frame(root, bg="#a0a0a0", padx=10, pady=10)
col3.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
display_list_with_status(col3, "Ground Truth Documents", col3_items, col3_correctness)

# ...existing code...

def process_query():
    
    to_rep_chars = ["(", ")", ".", ",", "!", "?", "-"]

    # Get query from text area
    query = text_area.get("1.0", "end-1c")
    # Clean query to prevent Java parsing errors
    query = clean_text(query) 
    # Remove newlines and extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    # Remove special characters that could cause parsing issues
    query = re.sub(r'[^\w\s.,?!-]', '', query)
    for char in to_rep_chars:
        query = query.replace(char, "")

    bm25_limit = int(num_var1.get())
    total_docs = int(num_var2.get())
    use_checks = use_checks_var.get()
    use_finegrained = use_finegrained_var.get()  # Add this line
    selected_model = dropdown_var.get()

    bm25 = pt.terrier.BatchRetrieve(indexref, wmodel="BM25", num_results=bm25_limit)
    start_time = time.time()
    results = bm25.search(query)
    bm25_retrieval_duration = time.time() - start_time

    # Process query using ReRanker
    docs_to_rerank = pd.merge(results, docs_df, on="docno", how="left")
    docs_to_rerank = docs_to_rerank[["docno", "text"]]
    if use_finegrained and selected_model.startswith('Qwen'):
        # Use fine-grained reranking
        reranked_d, reranked_indices, faiss_retrieval_duration, index_creation_duration = FineGrainedReRanker.rerank(
            docs_to_rerank["text"].tolist(), 
            query, 
            model=selected_model, 
            k=int(num_var2.get())
        )
    else:
        # Use regular reranking
        reranked_d, reranked_indices, faiss_retrieval_duration, index_creation_duration = ReRanker.rerank(
            docs_to_rerank["text"].tolist(), 
            query, 
            model=selected_model, 
            k=int(num_var2.get())
        )

    initial_order = docs_to_rerank["docno"].tolist()
    reranked_order = docs_to_rerank.iloc[reranked_indices]["docno"].tolist()

    # Filter docs_to_rerank to only include rows where docno is in reranked_order
    filtered_docs = docs_to_rerank[docs_to_rerank['docno'].isin(reranked_order)]
    # Get text column as list
    text_list = filtered_docs['text'].tolist()
    # Set correctness based on checkbox value
    if use_checks and is_predefined_query:
      # means user selected a predefined query thus we need to check whether these relevant texts are present in the qrels entry  
      
      query_qrels = queries_d[predefined_query_number]
      col2_correctness = [False] * len(text_list)
      filtered_docs_docnos = filtered_docs["docno"].tolist()
      for docno in filtered_docs_docnos:
        if docno in query_qrels and query_qrels[docno] == "1":
          col2_correctness[filtered_docs_docnos.index(docno)] = True
      
    else:
      col2_correctness = [True] * len(text_list)

    print("Retrieved Documents: ", text_list)

    if is_predefined_query and use_checks:
      col3_items = []
      col3_correctness = []
      for docno in query_qrels:
          if query_qrels[docno] == "1":
              try:
                col3_items.append(f"Doc{docno}: {docs_df.loc[docs_df['docno'] == docno, 'text'].values[0]}")
                col3_correctness.append(True)
              except Exception as e:
                  # means the docno is not in the docs_df
                  print(e)
                  continue
              print("GT hit")

      clear_list(col3, "Ground Truth Documents")
      display_list_with_status(col3, "Ground Truth Documents", col3_items, col3_correctness)


    clear_list(col2, "Retrieved Documents")
    display_list_with_status(col2, "Retrieved Documents", text_list, col2_correctness)

    to_send_query = f"""
    Query: {query}
    Retrieved Documents: {text_list}
    """
    update_llm_response(to_send_query)



    # also update the ground truth documents list
   
    

    # Update stats
    stats_value.set(f"""
Faiss Index Creation Duration: {index_creation_duration:.5f} seconds
Faiss Retrieval Duration: {faiss_retrieval_duration:.5f} seconds
BM25 Retrieval Duration: {bm25_retrieval_duration:.5f} seconds
BM25 Index Creation Duration: {bm25_index_creation_duration:.5f} seconds
    """)
    
root.mainloop()