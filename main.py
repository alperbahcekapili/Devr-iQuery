import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from src.vector.indexer import ReRanker
from src.vector.bm25 import create_index
from src.data.reader import parse_data
from src.vector.indexer import ReRanker
import pyterrier as pt
import pandas as pd


print("Starting UI")
index_folder = "./index"

print("Parsing data")
data = parse_data()
docs_df = pd.DataFrame(data["documents"].values())

# Check if index already exists
import os
if not pt.started():
    pt.init()
    
if not os.path.exists(index_folder):
    print("Creating new index")
    indexref = create_index(docs_df, index_folder=index_folder)
else:
    print("Loading existing index")
    indexref = pt.IndexRef.of(index_folder)

root = tk.Tk()


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
dropdown = ttk.Combobox(col1, textvariable=dropdown_var, values=["BERT", "all-MiniLM-L6-v2"])
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
use_checks = ttk.Checkbutton(col1, text="Use Checks", variable=use_checks_var)
use_checks.grid(row=5, column=0, columnspan=2, sticky="w", pady=5)

# Buttons (sixth row)
def open_window1():
    win = tk.Toplevel(root)
    win.title("Window 1")
def open_window2():
    win = tk.Toplevel(root)
    win.title("Window 2")
def open_window3():
    process_query()

button_frame = tk.Frame(col1)
button_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky="ew")
btn1 = ttk.Button(button_frame, text="Predefined Queries", command=open_window1)
btn2 = ttk.Button(button_frame, text="Documents", command=open_window2)
btn3 = ttk.Button(button_frame, text="Run Query", command=open_window3)
btn1.pack(side="left", expand=True, fill="x", padx=2)
btn2.pack(side="left", expand=True, fill="x", padx=2)
btn3.pack(side="left", expand=True, fill="x", padx=2)

# Query and Indexing Stats header (last row)
stats_header = tk.Label(col1, text="Query and Indexing Stats", font=("Arial", 12, "bold"), pady=10)
stats_header.grid(row=7, column=0, columnspan=2, sticky="w")

# Stats value (below header)
stats_value = tk.StringVar(value="""

Faiss Index Creation Duration: 
Faiss Retrieval Duration:
BM25 Retrieval Duration:
BM25 Index Creation Duration: 
Query Metrics: Precision etc. 

""")
stats_label = tk.Label(col1, textvariable=stats_value, bg="#e0e0e0", anchor="w", justify="left")
stats_label.grid(row=8, column=0, columnspan=2, sticky="ew")

# ...existing code...

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
    for i, (item, correct) in enumerate(zip(items, correctness)):
        color = "#b6fcb6" if correct else "#ffb3b3"  # green or red
        print(item)
        short_text = item[:200] + ("..." if len(item) > 200 else "")
        lbl = tk.Label(parent, text=short_text, bg=color, anchor="w", padx=5, pady=2, cursor="hand2", wraplength=400, justify="left")
        lbl.pack(fill="x", pady=1)
        lbl.bind("<Button-1>", lambda e, t=header_text, c=item: show_full_content(t, c))

def clear_list(parent, header_text):
    # Find and destroy the header label
    for widget in parent.winfo_children():
        if isinstance(widget, tk.Label) and widget.cget("text") == header_text:
            widget.destroy()
            break
    
    # Find and destroy all list item labels
    for widget in parent.winfo_children():
        if isinstance(widget, tk.Label) and widget.cget("cursor") == "hand2":
            widget.destroy()

# Example data for columns 2 and 3
col2_items = ["""Doc1: Introduction
              
              Command 'python' not found, did you mean:
  command 'python3' from deb python3
  command 'python' from deb python-is-python3
(devriquery) alpfischer@alpfischer-ubuntu:~/Devr-iQuery$ python3 src/ui/main.py 
Traceback (most recent call last):
  File "/home/alpfischer/Devr-iQuery/src/ui/main.py", line 1, in <module>
    import tkinter as tk
ModuleNotFoundError: No module named 'tkinter'
(devriquery) alpfischer@alpfischer-ubuntu:~/Devr-iQuery$ sudo apt update
sudo apt install python3-tk
[sudo] password for alpfischer: 
Get:1 file:/var/cuda-repo-ubuntu2204-12-6-local  InRelease [1.572 B]
Get:1 file:/var/cuda-repo-ubuntu2204-12-6-local  InRelease [1.572 B]
Hit:2 http://tr.archive.ubuntu.com/ubuntu jammy InRelease
Hit:3 https://packages.microsoft.com/repos/code stable InRelease                                                                                                                                                   
Hit:4 https://dl.google.com/linux/chrome/deb stable InRelease                                                                                                                                                      
Get:5 http://tr.archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]                                                                                                                                         
Hit:6 http://tr.archive.ubuntu.com/ubuntu jammy-backports InRelease                                                                                                                                                
Get:7 https://cli.github.com/packages stable InRelease [3.917 B]                                                
Get:8 http://security.ubuntu.com/ubuntu jammy-security InRelease [129 kB]                              
Hit:9 https://download.docker.com/linux/ubuntu jammy InRelease                 
Get:10 http://tr.archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [2.582 kB]
Err:7 https://cli.github.com/packages stable InRelease                                     
  The following signatures were invalid: EXPKEYSIG 23F3D4EA75716059 GitHub CLI <opensource+cli@github.com>
Get:11 http://tr.archive.ubuntu.com/ubuntu jammy-updates/main i386 Packages [809 kB]                   
Hit:12 https://ppa.launchpadcontent.net/obsproject/obs-studio/ubuntu jammy InRelease                             
Ign:13 https://ppa.launchpadcontent.net/otto-kesselgulasch/gimp/ubuntu jammy InRelease
Err:14 https://ppa.launchpadcontent.net/otto-kesselgulasch/gimp/ubuntu jammy Release
  404  Not Found [IP: 185.125.190.80 443]
Get:15 http://security.ubuntu.com/ubuntu jammy-security/main amd64 DEP-11 Metadata [54,5 kB]
Get:16 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 DEP-11 Metadata [208 B]
Get:17 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 DEP-11 Metadata [125 kB]
Get:18 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 DEP-11 Metadata [208 B]
Reading package lists... Done                                          
W: An error occurred during the signature verification. The repository is not updated and the previous index files will be used. GPG error: https://cli.github.com/packages stable InRelease: The following signatures were invalid: EXPKEYSIG 23F3D4EA75716059 GitHub CLI <opensource+cli@github.com>
E: The repository 'https://ppa.launchpadcontent.net/otto-kesselgulasch/gimp/ubuntu jammy Release' does not have a Release file.
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details.
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following package was automatically installed and is no longer required:
  nvidia-firmware-535-535.183.01
Use 'sudo apt autoremove' to remove it.
The following additional packages will be installed:
  blt tk8.6-blt2.5
Suggested packages:
  blt-demo tix python3-tk-dbg
The following NEW packages will be installed:
  blt python3-tk tk8.6-blt2.5
0 upgraded, 3 newly installed, 0 to remove and 6 not upgraded.
Need to get 757 kB of archives.
After this operation, 2.920 kB of additional disk space will be used.
Do you want to continue? [Y/n] y
Get:1 http://tr.archive.ubuntu.com/ubuntu jammy/main amd64 tk8.6-blt2.5 amd64 2.5.3+dfsg-4.1build2 [643 kB]
Get:2 http://tr.archive.ubuntu.com/ubuntu jammy/main amd64 blt amd64 2.5.3+dfsg-4.1build2 [4.838 B]
Get:3 http://tr.archive.ubuntu.com/ubuntu jammy-updates/main amd64 python3-tk amd64 3.10.8-1~22.04 [110 kB]
Fetched 757 kB in 0s (2.263 kB/s)    
Selecting previously unselected package tk8.6-blt2.5.
(Reading database ... 270059 files and directories currently installed.)
Preparing to unpack .../tk8.6-blt2.5_2.5.3+dfsg-4.1build2_amd64.deb ...
Unpacking tk8.6-blt2.5 (2.5.3+dfsg-4.1build2) ...
Selecting previously unselected package blt.
Preparing to unpack .../blt_2.5.3+dfsg-4.1build2_amd64.deb ...
Unpacking blt (2.5.3+dfsg-4.1build2) ...
Selecting previously unselected package python3-tk:amd64.
Preparing to unpack .../python3-tk_3.10.8-1~22.04_amd64.deb ...
Unpacking python3-tk:amd64 (3.10.8-1~22.04) ...
Setting up tk8.6-blt2.5 (2.5.3+dfsg-4.1build2) ...
Setting up blt (2.5.3+dfsg-4.1build2) ...
Setting up python3-tk:amd64 (3.10.8-1~22.04) ...
Processing triggers for libc-bin (2.35-0ubuntu3.9) ...
              
              
              
              """, "Doc2: Methods", "Doc3: Results"]
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
    # Get query from text area
    query = text_area.get("1.0", "end-1c")

    # Get parameters from UI
    bm25_limit = int(num_var1.get())
    total_docs = int(num_var2.get())
    use_checks = use_checks_var.get()  # Get checkbox value

    bm25 = pt.terrier.BatchRetrieve(indexref, wmodel="BM25", num_results=bm25_limit)
    results = bm25.search(query)

    # Process query using ReRanker
    docs_to_rerank = pd.merge(results, docs_df, on="docno", how="left")
    docs_to_rerank = docs_to_rerank[["docno", "text"]]
    reranked_d, reranked_indices = ReRanker.rerank(docs_to_rerank["text"].tolist(), query, model=dropdown_var.get(), k=int(num_var2.get()))

    initial_order = docs_to_rerank["docno"].tolist()
    reranked_order = docs_to_rerank.iloc[reranked_indices]["docno"].tolist()

    # Filter docs_to_rerank to only include rows where docno is in reranked_order
    filtered_docs = docs_to_rerank[docs_to_rerank['docno'].isin(reranked_order)]
    # Get text column as list
    text_list = filtered_docs['text'].tolist()
    # Set correctness based on checkbox value
    if use_checks:
      # means user selected a predefined query thus we need to check whether these relevant texts are present in the qrels entry  
      col2_correctness = [use_checks] * len(text_list) if use_checks else [True] * len(reranked_d)
    else:
      col2_correctness = [True] * len(text_list)

    print("Retrieved Documents: ", text_list)
    clear_list(col2, "Retrieved Documents")
    display_list_with_status(col2, "Retrieved Documents", text_list, col2_correctness)

    # Update stats
    stats_value.set(f"""
Faiss Index Creation Duration: {0.0} seconds
Faiss Retrieval Duration: {0.0} seconds
BM25 Retrieval Duration: {0.0} seconds
BM25 Index Creation Duration: {0.0} seconds
Query Metrics: Precision etc. 
    """)
    
    """# Update results in column 2
    for i, idx in enumerate(indices):
        if i < len(col2_items):
            col2_items[i] = col2_items[idx]
            col2_correctness[i] = True  # You might want to implement proper correctness checking
    
    # Refresh the display
    display_list_with_status(col2, "Retrieved Documents", col2_items, col2_correctness)
"""
root.mainloop()