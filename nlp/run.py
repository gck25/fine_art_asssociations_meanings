import os
import json
import networkx as nx

from nlp.data.gold_standard import GOLD_STANDARD
from nlp.data.test_set_artists_titles import TEST_SET_ARTSISTS_TITLES
from nlp.kg_utils import find_train_test_split,\
                        create_edge_weight_dict,\
                        create_kg_from_dict,\
                        save_admatrix_nodelist,\
                        refine_kg_to_include_only_relevant_objects

from nlp.metrics import calc_soft_hard_metrics, calc_semantic_metrics


# args
web_text_path = '../data/vanitas_meaning_text.json'
edge_threshold = 2
dst_dir = 'kg_srl'
kg_file = 'vanitas_meaning_KG_SRL_train.gpickle'
semantic_threshold = 0.7


def main():

    # load web texts
    with open(web_text_path) as f:
        web_texts = json.load(f)

    # train-test split
    train_texts, test_texts = find_train_test_split(web_texts, TEST_SET_ARTSISTS_TITLES)

    # create head-tails dictionary from train texts
    ht_dict = create_edge_weight_dict(train_texts)

    # create knowledge graph from head-tails dict
    kg = create_kg_from_dict(ht_dict, edge_threshold=edge_threshold)

    if not os.path.exists(dst_dir):
        os.mkdir((dst_dir))

    # refine KG to include only objects detected by object detector
    kg = refine_kg_to_include_only_relevant_objects(kg)

    # save KG, adjacency matrix and nodelist
    dst_path_kg = os.path.join(dst_dir, kg_file)
    nx.write_gpickle(kg, dst_path_kg)
    save_admatrix_nodelist(kg, dst_dir)

    # calculate hard and soft metrics
    hard_metrics, soft_metrics = calc_soft_hard_metrics(kg, GOLD_STANDARD)

    # calculate semantic metrics
    sem_p, sem_r, sem_f1 = calc_semantic_metrics(kg, GOLD_STANDARD, semantic_threshold)

    print("########### NLP Component ###########")
    print("########### Hard metrics ###########")
    print("Precision:", hard_metrics[0])
    print("Recall:", hard_metrics[1])
    print("F1:", hard_metrics[2])
    print("")
    print("########### Soft metrics ###########")
    print("Precision:", soft_metrics[0])
    print("Recall:", soft_metrics[1])
    print("F1:", soft_metrics[2])
    print("")
    print("########### Semantic metrics ###########")
    print("Precision:", sem_p)
    print("Recall:", sem_r)
    print("F1:", sem_f1)


if __name__ == "__main__":
    main()
