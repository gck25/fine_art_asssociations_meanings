from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model_embed = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')


def calc_precision(TP, FP):
    return TP/(TP+FP + 1e-15)


def calc_recall(TP, FN):
    return TP/(TP + FN + 1e-15)


def calc_f1(p, r):
    return 2*p*r/(p + r + 1e-15)


def cos_sim(e1, e2):
    e2_r = e2.reshape(1, -1)
    return cosine_similarity(e1, e2_r)


def calc_semantic_helper(s, S, threshold):
    e_s = model_embed.encode(s).reshape(1, -1)
    e_S = model_embed.encode(S)

    rejected = []

    TP = 0
    FP = 0

    under_threshold_for_all = True

    for i in range(len(e_S)):

        if cos_sim(e_s, e_S[i])[0][0] >= threshold:
            TP += 1
            under_threshold_for_all = False
        else:
            rejected.append(S[i])

    if under_threshold_for_all:
        FP += 1

    return set(rejected), TP, FP

def calc_semantic_metrics(SG, gold_standard, semantic_threshold):
    FP = 0
    FN = 0
    TP = 0

    for key in SG:
        rejected_all = None
        for pred in SG[key]:
            if key in gold_standard['generic']:
                rejected, TP_key, FP_key, sim_sent = calc_semantic_helper(pred, gold_standard['generic'][key], semantic_threshold)
                TP += TP_key
                FP += FP_key

                if not rejected_all:
                    rejected_all = rejected
                else:
                    rejected_all = rejected_all.intersection(rejected)
                FP += len(SG[key])

        if rejected_all:
            FN += len(rejected_all)

    for key in gold_standard['generic']:
        if key not in SG:
            FN += len(gold_standard['generic'][key])

    p = calc_precision(TP, FP)
    r = calc_recall(TP, FN)
    f1 = calc_f1(p, r)

    return p, r, f1


def calc_soft_hard_metrics(SG, gold_standard):
    hard_true_positives = 0
    hard_false_positives = 0

    soft_true_positives = 0
    soft_false_positives = 0

    hard_false_negatives = 0
    soft_false_negatives = 0

    for key in SG:
        if key not in gold_standard['generic']:
            soft_false_positives += len(SG[key])
            hard_false_positives += len(SG[key])
            continue

        for pred in SG[key]:
            is_label = False
            for label in gold_standard['generic'][key]:
                if pred.lower() == label.lower():
                    hard_true_positives += 1
                    soft_true_positives += 1
                    islabel = True
                    break
                elif pred.lower() in label.lower():
                    soft_true_positives += 1
                    hard_false_negatives += 1
                    is_label = True
                    break

            if not is_label:
                soft_false_positives += 1
                hard_false_positives += 1

    for key in gold_standard['generic'].keys():
        if key not in SG:
            hard_false_negatives += len(gold_standard['generic'][key])
            soft_false_negatives
            continue

        for label in gold_standard['generic'][key]:
            is_pred = False
            for pred in SG[key]:
                if pred.lower() == label.lower():
                    is_pred = True
                    break
                elif pred.lower() in label.lower():
                    is_pred = True
                    break
            if not is_pred:
                soft_false_negatives += 1
                hard_false_negatives += 1

    soft_precision = calc_precision(soft_true_positives, soft_false_positives)
    soft_recall = calc_recall(soft_true_positives, soft_false_negatives)
    soft_f1 = calc_f1(soft_precision, soft_recall)

    hard_precision = calc_precision(hard_true_positives, hard_false_positives)
    hard_recall = calc_precision(hard_true_positives, hard_false_negatives)
    hard_f1 = calc_f1(hard_precision, hard_recall)

    return (soft_precision, soft_recall, soft_f1), (hard_precision, hard_recall, hard_f1)

