# using grpc on :54321 listen for proto messages and print them
import grpc
import re
import json
import nltk
import string
from tqdm.notebook import tqdm
from num2words import num2words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from umap import UMAP
from typing import List, Union
from sklearn.preprocessing import MinMaxScaler
from torch_geometric import utils as util_g
import os
import sys
import spacy

project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_directory)

import resume_pb2 as pb
import resume_pb2_grpc as pb_grpc

class Resume(pb_grpc.ResumeServiceServicer):
    def GetResumes(self, request, context):
        print(request)
        return pb.ResumeResponse()

def get_resumes():
    res_data = []
    options = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    channel = grpc.insecure_channel('127.0.0.1:5007', options = options)
    stub = pb_grpc.ResumeServiceStub(channel)
    # Create a ResumeRequest message
    request = pb.ResumeRequest(
        ID = "1"
    )

    # Send the ResumeRequest to the server and receive the response with context
    response = stub.GetResumes(request=request, )

    # Print the received response
    print("Received Resumes from Server:")
    for resume in response.resumes:
        for i in resume.Experience:
            res_.append(i)
    print(res_data)        


    res_ = get_resumes()
    def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    text = text.replace('-', ' ')
    table=str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return text.translate(table)

def create_corpus(text):
    corpus=[]
    for sent in text.split(' '):
        words=[word.lower() for word in word_tokenize(sent) if ((word.isalpha()==True) & (word.lower() not in stop_ru))]
        corpus.extend(words)
    return corpus




lemmatizer = spacy.load("ru_core_news_lg")
stop_en = set(stopwords.words('english'))
stop_ru = set(stopwords.words('russian') + ['ооо', 'зао', ' ао ',' ао', 'ип', 'оао', ' сз ', ' сз'])




def get_vocab(res_data):
    vocab = []
    for exps in tqdm(res_data):
        vocab_loc_loc = []
        for data_loc in exps:
            vocab_loc = []
            clean = [remove_emoji(remove_URL(i.replace('\xa0', ' ').replace('\n', ' '))) for i in data_loc.split('ඞ')]
            try:
                clean[0] = clean[0].replace('по настоящее время','9999').replace('currently','9999')
                splited = clean[0].split(' ')
                if len(splited) == 6:
                    clean[0] = num2words(splited[4][4:], lang='ru') + ' ' + splited[-1]
                elif len(splited) == 8:
                    clean[0] = num2words(splited[4][4:], lang='ru') + ' ' + splited[5] + ' ' + num2words(splited[6], lang='ru')+ ' ' + splited[7]
                elif len(splited) == 5:
                    clean[0] = num2words(splited[3][4:], lang='ru') + ' ' + splited[-1]
            except:
                print(clean[0].split(' '), 'bad')
                
            clean[0] = ''
            clean[3] = clean[3].split(', ')[0]
            clean[3] = ''
            clean[2] = clean[2].replace("зам. ", "заместитель ")
            clean[2] = clean[2].replace("нач. ", "начальник ")
            clean[2] = clean[2].replace("рук. ", "начальник ")
            clean[2] = clean[2].replace("ген. ", "начальник ")
            clean[2] = clean[2].replace("гендиректор", "начальник ")
            clean[2] = clean[2].replace("руководитель", "начальник ")
            clean[2] = clean[2].replace("директор", "начальник ")
            clean[2] = clean[2].replace("старший", "начальник ")
            clean[2] = clean[2].replace("главный", "начальник ")
            
            no_punkt = [remove_punct(i) for i in clean]
            infos = []
            for part in no_punkt:
                infos.extend([i.lemma_ if len(i.text) > 2 else '' for i in lemmatizer(part)])
            for info in infos:
                corp_ = create_corpus(info)
                if len(corp_) > 0:
                    vocab_loc.append(corp_)
            vocab_loc_loc.append(vocab_loc)
        vocab.append(vocab_loc_loc)



    res_ = []
    for i in range(len(vocab)):
        resume_batch = []
        for exp in vocab[i]:
            str_ = ''
            for j, exp_field in enumerate(exp):
                str_+=' '.join(exp_field) + ' '
    #         str_+='\n'
    #         for drv in vocab_driver[i]:
    #             str_+=' '.join([i for i in drv])+' '
    #         for lng in vocab_lang[i]:
    #             str_+=' '.join([i for i in lng])+' '
    #         str_+=vocab_salary[i]
    #         for skill in vocab_skills[i]:
    #             str_+=' '.join([i for i in skill[:3]])+'\n'
    #         for edu in vocab_edus[i]:
    #             for edu_field in edu:
    #                 str_+=' '.join([i for i in edu_field])
    #             str_+='\n'
            resume_batch.append(str_.strip())
        res_.append(resume_batch)

    data_flat = [item for sublist in res_ for item in sublist]
    data_flat_new = []
    for i in data_flat:
        cleared = i.replace(' тч ', '').replace(' тч', '').replace(' тд','').replace(' тд ','').replace(' вет ','ветеринарный')
        data_flat_new.append(' '.join([i if len(i) > 2 else '' for i in cleared.split()]))
    
    data_batch_new = []
    for batch in res_:
        loc = []
        for i in batch:
            cleared = i.replace(' тч ', '').replace(' тч', '').replace(' тд','').replace(' тд ','').replace(' вет ','ветеринарный')
            loc.append(' '.join([i if len(i) > 2 else '' for i in cleared.split()]))
        data_batch_new.append(loc)
    return get_vecs(data_batch_new, data_flat_new)




def load_model(path_to_model):
    return SentenceTransformer(path_to_model)


def train_model(model, res_data, save_path, model_name, batch_size = 2, epochs = 1, lr = 3e-5):
    from sentence_transformers import SentenceTransformer, LoggingHandler
    from sentence_transformers import models, util, datasets, evaluation, losses
    from torch.utils.data import DataLoader

    train_dataset = datasets.DenoisingAutoEncoderDataset(data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)
        model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': lr},
        show_progress_bar=True
    )
    model.save(save_path)

def get_network_pred(edges, full_network, model_path):
    embeddings = model_bert.encode(strings_flat, normalize_embeddings = True, show_progress_bar = False)
    emb_umaped = umap_model.fit_transform(embeddings)
    cluster_labels = hdbscan_model.fit_predict(emb_umaped)
    labels_unique = np.unique(cluster_labels)
    adaj_matr = np.zeros((len(strings_flat), len(strings_flat)))
    x = []
    y = []
    glob_id_to_loc = dict.fromkeys(list(range(len(strings_flat))), -1)
    glob_id = 0
    true_indx = 0
    for batches in strings_:
        if len(batches) > 1:
            for rab in range(len(batches) - 1):            
                adaj_matr[glob_id, glob_id+1] = 1
                adaj_matr[glob_id+1, glob_id] = 1
                glob_id_to_loc[glob_id] = true_indx
                x.append(embeddings[glob_id])
                y.append(embeddings[glob_id + 1])
                glob_id+=1
                true_indx+=1
            glob_id_to_loc[glob_id] = true_indx
            if (glob_id + 1) < len(strings_flat):
                x.append(embeddings[glob_id])
                y.append(embeddings[glob_id + 1])
    #     else:
    #         true_indx+=1
    #         print(batches)
    #         print(strings_flat[glob_id])
        glob_id+=1
    loc_id_to_glob = dict((v,k) for k,v in glob_id_to_loc.items())
    edge_matrix = torch.tensor(np.triu(adaj_matr)).nonzero().contiguous()

import numpy as np


def visualize_topics2(topic_model,
                 topics: List[int] = None,
                 top_n_topics: int = None,
                 custom_labels: Union[bool, str] = False,
                 title: str = "<b>Intertopic Distance Map</b>",
                 width: int = 650,
                 height: int = 650):
    """ Visualize topics, their sizes, and their corresponding words

    This visualization is highly inspired by LDAvis, a great visualization
    technique typically reserved for LDA.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics
        custom_labels: If bool, whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:

    To visualize the topics simply run:

    ```python
    topic_model.visualize_topics()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/viz.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [topic_model.topic_sizes_[topic] for topic in topic_list]
    if isinstance(custom_labels, str):
        words = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topic_list]
        words = ["_".join([label[0] for label in labels[:4]]) for labels in words]
        words = [label if len(label) < 30 else label[:27] + "..." for label in words]
    elif custom_labels and topic_model.custom_labels_ is not None:
        words = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topic_list]
    else:
        words = [" | ".join([word[0] for word in topic_model.get_topic(topic)[:5]]) for topic in topic_list]

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])

    if topic_model.topic_embeddings_ is not None:
        embeddings = topic_model.topic_embeddings_[indices]
        embeddings = UMAP(n_neighbors=2, n_components=2, metric='cosine', random_state=42).fit_transform(embeddings)
    else:
        embeddings = topic_model.c_tf_idf_.toarray()[indices]
        embeddings = MinMaxScaler().fit_transform(embeddings)
        embeddings = UMAP(n_neighbors=2, n_components=2, metric='hellinger', random_state=42).fit_transform(embeddings)

    # Visualize with plotly
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                       "Topic": topic_list, "Words": words, "Size": frequencies})
    return df


def get_vecs(data_batch, data_flat):
    from umap import UMAP
    from sklearn.cluster import AgglomerativeClustering
    from bertopic.representation import MaximalMarginalRelevance
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic import BERTopic
    res_data = data_flat
    umap_model = UMAP(n_neighbors=15, n_components=128, min_dist=0.00001, metric='cosine') #128
    #25 ok
    hdbscan_model = AgglomerativeClustering(n_clusters=25, compute_distances = True, metric = 'cosine', linkage='complete')
    
    ctfidf_model = ClassTfidfTransformer()#bm25_weighting=True)
    vectorizer_model = CountVectorizer(ngram_range=(1, 3))
    representation_model = MaximalMarginalRelevance(diversity=0.1)

    topic_model = BERTopic(embedding_model=load_model('cointegrated/rubert-tiny2'), 
                        umap_model=umap_model,
                        hdbscan_model = hdbscan_model,
                        ctfidf_model = ctfidf_model,
                        vectorizer_model = vectorizer_model,
                        representation_model = representation_model, 
                        language = 'russian', verbose = True)

    topics, probs = topic_model.fit_transform(data_flat)
    topic_model.visualize_topics = visualize_topics2
    # two_d_data_ = topic_model.visualize_topics(topic_model)

    embeddings = model.encode(data_flat, normalize_embeddings = True, show_progress_bar = False)
    emb_umaped = umap_model.fit_transform(embeddings)
    cluster_labels = hdbscan_model.fit_predict(emb_umaped)
    labels_unique = np.unique(cluster_labels)

    adaj_matr = np.zeros((len(strings_flat), len(strings_flat)))
    glob_id_to_loc = dict.fromkeys(list(range(len(strings_flat))), -1)



    glob_id = 0
    true_indx = 0
    for batches in strings_:
        if len(batches) > 1:
            for rab in range(len(batches) - 1):            
                adaj_matr[glob_id, glob_id+1] = 1
                adaj_matr[glob_id+1, glob_id] = 1
                glob_id_to_loc[glob_id] = true_indx
                glob_id+=1
                true_indx+=1
            glob_id_to_loc[glob_id] = true_indx
    #     else:
    #         true_indx+=1
    #         print(batches)
    #         print(strings_flat[glob_id])
        glob_id+=1
    loc_id_to_glob = dict((v,k) for k,v in glob_id_to_loc.items())

    edge_matrix = torch.tensor(np.triu(adaj_matr)).nonzero().contiguous()
    edge_matrix_np = edge_matrix.numpy()

    link_dict = {}
    full_info = {}
    # embeds_berted = {}


    full_edges = []
    for label in tqdm(labels_unique):
        glob_ids = np.where(cluster_labels == label)[0]
        edge_loc = []
        info_loc = []
        target_clusters = set()#[]#!
        for glob_id in glob_ids:
            loc_id_for_edge = glob_id_to_loc[glob_id]
            info_loc.extend(docs_info.loc[glob_id]['Representation'])
            edges_ids = np.where(edge_matrix_np[:, 0] == loc_id_for_edge)[0]
            edges = edge_matrix_np[edges_ids, :]
    #         print(edges)
            for edge in edges:
                try:
                    target_glob_id = loc_id_to_glob[edge[1]]
                    if target_glob_id == -1:
                        print('no connection')
                    else:
                        target_clusters.add(cluster_labels[target_glob_id]) #!
                except:
                    print(edge[1])
                    pass
            edge_loc.extend(edges)
    #     print(info_loc)
    #     for glob_id in glob_ids:
        full_info[label] = Counter(info_loc).most_common(1)[0][0]
        full_edges.append(edge_loc)
        link_dict[label] = list(target_clusters)

    keys=sorted(link_dict.keys())
    size=len(keys)

    M = [[0]*size for i in range(size)]

    for a,b in [(keys.index(a), keys.index(b)) for a, row in link_dict.items() for b in row]:
        M[a][b] = 0 if (a==b) else 1
    M_triu = np.triu(M)
    edge_matrix111 = torch.tensor(M_triu).nonzero().t().contiguous()
    edge_matrix111 = util_g.sort_edge_index(edge_matrix111)
    edge_matrix_np2 = edge_matrix111.numpy()
    ids_ = list(set(set(edge_matrix_np2[0, :]) | set(edge_matrix_np2[1, :])))
    data_ = edge_matrix_np2.T
    d_ = {
        'nodes':[
            {
                "id": str(i),
                "text": ' '.join(full_info[i].split(' ')[:2])
            } for i in ids_
        ],
        'links':[
            {
                "source": str(i[0]),
                "target": str(i[1]),
                "value": 1
            } for i in data_
        ],
    }
    return d_

if __name__ == '__main__':    
    with open('../../frontend/data/data_cluster_2500_all_128_with_name1.json', 'w') as f:
        json.dump(get_vocab(get_resumes()))
    