import os
import json
import re
import tqdm

def parse_documents(folder_path):
    # Pattern to match <DOC>...</DOC>
    doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)

    # Pattern to match <TAG>...</TAG>
    tag_pattern = re.compile(r"<(.*?)>(.*?)</\1>", re.DOTALL)

    result = {}

    for filename in tqdm.tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all <DOC>...</DOC> blocks
            docs = doc_pattern.findall(content)

            for doc in docs:
                doc_dict = {}
                # Find all tags inside this <DOC> block
                for tag_match in tag_pattern.finditer(doc):
                    tag = tag_match.group(1).strip()
                    value = tag_match.group(2).strip()
                    doc_dict[tag] = value
                
                result[doc_dict["DOCNO"]] = doc_dict
                # result.append(doc_dict)

    return result


def parse_topics(file_path):
    """
        returns dict where keys are topic ids 
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match <top>...</DOtoptopC>
    top_pattern = re.compile(r"<top>(.*?)</top>", re.DOTALL)

    # Find all <top>...</top> blocks
    docs = top_pattern.findall(content)

    result = {}

    for doc in docs:
        entries = doc.split(">")
        topic = {}
        eindex = -1 # skips first entry because tag itself is first entry
        for e in entries:
            if "<" in e:
                e = e[:e.index("<")]

            e = e.strip()
            if eindex == 0:
                e = e.replace("Number:","").strip()
                topic["number"] = e
            if eindex == 1:
                topic["title"] = e
            if eindex == 2:
                e = e.replace("Description:","").strip()
                topic["description"] = e
            if eindex == 3:
                e = e.replace("Narrative:", "").strip()
                topic["narrative"] = e
            eindex += 1
    
        result[topic["number"]] = topic
    return result
        


def parse_qrel_judgements(file):
    """
    reads topic rel judgements where ids are topic ids
    """
    judgements = {}
    with open(file, "r") as f:
        lines = f.readlines()
        for l in lines:
            if l.strip() == "":
                continue
            entries = l.split()
            [topic_id, _, doc_id, rel] = entries
            if topic_id not in judgements:
                judgements[topic_id] = {}
            judgements[topic_id][doc_id]  = rel

    
    return judgements



def parse_data():
    """
    Loads ocuments, topics, qrel_judgements all into a single json from local paths
    """

    documents_root = "data/ft/all"
    topic_paths = [
        "data/query-relJudgments/q-topics-org-SET1.txt",
        "data/query-relJudgments/q-topics-org-SET2.txt",
        "data/query-relJudgments/q-topics-org-SET3.txt"
    ]
    qrel_paths = [
        "data/query-relJudgments/qrel_301-350_complete.txt",
        "data/query-relJudgments/qrels.trec7.adhoc_350-400.txt",
        "data/query-relJudgments/qrels.trec8.adhoc.parts1-5_400-450"
    ]

    documents = parse_documents(documents_root)
    topics_ar = [parse_topics(topic_path) for topic_path in topic_paths]
    qrels_ar = [parse_qrel_judgements(qrel_path) for qrel_path in qrel_paths]
    
    topics_ar[0].update(topics_ar[1])
    topics_ar[0].update(topics_ar[2])
    

    qrels_ar[0].update(qrels_ar[1])
    qrels_ar[0].update(qrels_ar[2])

    return {
        "documents" : documents,
        "topics" : topics_ar[0],
        "qrels" : qrels_ar[0]
    }


