
from allennlp.predictors.predictor import Predictor
from collections import defaultdict
import networkx as nx
import numpy as np
import random
import csv
import os
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import spacy
import tqdm
from nlp.data.object_detector_tags import TAGS

# load nlp model from spacy
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")


def refine_kg_to_include_only_relevant_objects(kg):
    """
    prune KG to include only objects that can be detected by vision module

    :param kg:  KG to be refined
    :return: Pruned KG
    """
    object2meanings = {}

    subgraph_nodes = []
    mapping = {}

    for object in TAGS:
        object2meanings[object] = []
        meanings = []
        for node in kg.nodes:
            if object in node.lower():
                subgraph_nodes.append(object)
                subgraph_nodes.extend([child for child in kg.successors(node)])
                mapping[node] = object
                meanings.append([child for child in kg.successors(node)])
        object2meanings[object].append(meanings)

    H = nx.relabel_nodes(kg, mapping)
    SG = H.subgraph(subgraph_nodes)
    return SG


def find_train_test_split(web_texts, test_set_artist_titles):
    """
    divide train and test split based on references to images contained in tesxts

    :param web_texts: list of texts from web search
    :param test_set_artist_titles: artists and titles of paintings in test set
    :return: train and test texts
    """
    train_texts = []
    test_texts = {}
    for text in web_texts:
        is_test = False
        for key in test_set_artist_titles.keys():
            if key not in test_texts:
                test_texts[key] = []
            painting = test_set_artist_titles[key]
            artist = painting['artist']
            titles = painting['titles']
            for name in artist:
                for title in titles:
                    if name.lower() in text.lower() and title.lower() in text.lower():
                        is_test = True
                        test_texts[key].append(text)
                        break
                    if is_test:
                        break
                if is_test:
                    break
            if is_test:
                break
        if not is_test:
            train_texts.append(text)
    return train_texts, test_texts

def extract_kg_triples(verb_dict, words):
    """
    Take verb dict from SRL module and extract head, verb, tails arrangement

    :param verb_dict: dictionary with 'verbs' and 'tags' from SRL module
    :param words: original text
    :return: head, verb, tail triplets
    """

    head = []
    tails = {}
    verb = verb_dict['verb']
    tags = verb_dict['tags']

    for i in range(len(words)):
        if tags[i] == 'B-ARG0' or tags[i] == 'I-ARG0':
            head.append(words[i])
        elif tags[i] == 'B-ARG1' or tags[i] == 'I-ARG1':
            head.append(words[i])
        elif 'B-ARG' in tags[i] or 'I-ARG' in tags[i]:
            index = tags[i].split('ARG')[-1]
            try:
                index = int(index)
            except:
                continue
            try:
                tails[index].append(words[i])
            except:
                tails[index] = [words[i]]
    return (head, verb, tails)


def remove_stop_words(text):
    # function to remove stop words from texts
    sent = []
    for word in text.split(' '):
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            sent.append(word)
    return ' '.join(sent)


def find_noun_chunks(words):
    # function to find noun chunks in texts
    doc = nlp(words)

    if not list(doc.noun_chunks):
        return [words]
    chunks = []
    for chunk in doc.noun_chunks:
        sent = remove_stop_words(chunk.text)
        #         sent = chunk.text
        chunks.append(sent)
    return chunks


def add_im(h, t, kg, weight):
    # add edge to KG
    kg.add_node(h, bipartite=0)
    kg.add_node(t, bipartite=1)
    kg.add_edge(h, t, weight=weight)


def add_edge_weights_to_dict(ht_dict, sent):
    """
    assigns weights based on number of times h-t relation has been observed in text.

    :param ht_dict: head-tails dict.
    :param sent: sentence
    :return: None
    """
    srl = predictor.predict(
        sentence=sent
    )

    verbs = srl['verbs']
    words = srl['words']

    for verb_dict in verbs:
        head, verb, tails = extract_kg_triples(verb_dict, words)
        head = ' '.join(head)
        heads = find_noun_chunks(head)
        for h in heads:
            if h == '':
                continue

            for key in tails.keys():
                tail = ' '.join(tails[key])
                tail = find_noun_chunks(tail)
                for t in tail:
                    if t == '':
                        continue
                    ht_dict[(h, t)] += 1

def create_edge_weight_dict(train_texts):
    """
    Create dictionary containing edge weights

    :param train_texts: texts used to create KG
    :return: heads-tails dictionary
    """
    ht_dict = defaultdict(int)
    for i in tqdm.tqdm(range(len(train_texts))):
            sents = sent_tokenize(train_texts[i])
            if len(sents)>500:
              sents = random.sample(sents, 500)
            for j in range(len(sents)):
              sent = sents[j].strip()[:512]
              add_edge_weights_to_dict(ht_dict, sent)
    return ht_dict


def create_kg_from_dict(ht_dict, edge_threshold=1, is_remove_stop_words=True):
    """
    Create a KG from a h-t dictionary

    :param ht_dict: head-tail dict
    :param edge_threshold: weight threshold at which to prune edges
    :param is_remove_stop_words: boolean defining whether to remove stopwords
    :return: Knowledge graph
    """
    ent_labels = ['PERSON', 'ORG', 'GPE', 'LOC']
    kg = nx.DiGraph()

    # add pairs and weights to graph
    for (h, t) in ht_dict:
        weight = ht_dict[(h, t)]
        h = ''.join(ch for ch in h if ch.isalnum() or ch is ' ')
        t = ''.join(ch for ch in t if ch.isalnum() or ch is ' ')
        if weight >= edge_threshold and '  ' not in h and '  ' not in t:

            if is_remove_stop_words:
                h = remove_stop_words(h)
                t = remove_stop_words(t)

            in_ent_labels = False

            h_doc = nlp(h)
            for ent in h_doc.ents:
                if ent.label_ in ent_labels:
                    in_ent_labels = True
                    break

            if in_ent_labels:
                continue

            t_doc = nlp(t)
            for ent in t_doc.ents:
                if ent.label_ in ent_labels:
                    in_ent_labels = True
                    break

            if in_ent_labels:
                continue

            add_im(h, t, kg, weight)
    return kg


def display_kg(kg):
    """
    Display the KG visually

    :param kg: Knowledge graph
    :return: None
    """

    pos = nx.spring_layout(kg, k=3 * 1 / np.sqrt(len(kg.nodes())), iterations=20)
    plt.figure(figsize=(20, 20))
    nx.draw(kg, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, arrowsize=0.001, node_color='seagreen', alpha=0.9,
            labels={node: node for node in kg.nodes()}, font_size=20)
    plt.axis('off')


def save_admatrix_nodelist(kg, dst_dir):
    """
    Save the adjacency matrix (AM) and nodelist (NL) of the KG

    :param kg: knowledge graph
    :param dst_dir: directory to save AM and NL
    :return: None
    """
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    admatix = nx.adjacency_matrix(kg).todense()
    np.savetxt(os.path.join(dst_dir, "KGAM_SRL.csv"), np.array(admatix, dtype=np.int), delimiter=",")

    with open(os.path.join(dst_dir, 'KGNL_SRL.csv'), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(list(kg.nodes))