import json
from sklearn.metrics import f1_score
import re

def emoverse_sentiment(mosei_sentiment_json):
    mosei_sentiment_json = "/home/usr/liao/swift-mllm/output/emoverse-4b-mosei/infer_result/20241204-212600.jsonl"
    pos_cnt = 0
    neg_cnt = 0

    pos_rcnt = 0
    neg_rcnt = 0

    with open(mosei_sentiment_json, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            if record["label"] == "Positive.":
                pos_cnt+=1
            elif record["label"] == "Negative.":
                neg_cnt+=1
            if record["response"] == record["label"]:

                if record["response"] == "Positive.":
                    pos_rcnt+=1
                elif record["response"] == "Negative.":
                    neg_rcnt+=1

    print(pos_rcnt/pos_cnt,neg_rcnt/neg_cnt) 
    print((pos_rcnt+neg_rcnt)/(pos_cnt+neg_cnt))

def emoverse_emotion(emotion_json):
    e_cnt = 0
    e_r = 0

    # emotion_list = ["Anger.","Sadness.","Surprise.","Disgust.","Joy.","Neutral.","Fear."]
    emotion_list = ["Happy.","Sadness.","Surprise.","Disgust.","Anger.","Fear.","Neutral."]
    emotion_correct = []
    emotion_pred = []
    with open(emotion_json, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            emotion_pred.append(emotion_list.index(record["response"]))
            emotion_correct.append(emotion_list.index(record["label"]))

            e_cnt+=1
            if record["response"] == record["label"]:
                e_r+=1

    print(e_r/e_cnt) # 情感识别准确率66.7%
    print(f1_score(emotion_correct,emotion_pred,average="weighted"))

def emoverse_ecf2(ecf2_json):

    pair_cnt = 0
    corr_cnt = 0
    pred_cnt = 0
    matched_pairs = []
    all_pairs = []

    categories = ["disgust", "anger", "joy", "sadness", "surprise", "fear"]
    category_counts = {cat: 0 for cat in categories}
    category_corr_counts = {cat: 0 for cat in categories}
    category_pred_counts = {cat: 0 for cat in categories}
    with open(ecf2_json, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            labels = record["label"].split(";")
            if labels[0] == "":
                continue
            pair_cnt += len(labels)
            all_pairs.extend(labels) # 记录全部的pair

            for label in labels:
                category = label.split(",")[0].split("_")[1]
                if category in category_counts:
                    category_counts[category] += 1

            preds = record["response"].split(";")
            pred_cnt += len(preds) # 记录预测的总数

            for pred in preds:
                if pred in labels:
                    matched_pairs.append(pred) # 记录正确的pair
                    corr_cnt += 1

                    category = pred.split(",")[0].split("_")[1]
                    if category in category_corr_counts:
                        category_corr_counts[category] += 1

                category = pred.split(",")[0].split("_")[1]
                if category in category_pred_counts:
                    category_pred_counts[category] += 1
    
    
    category_metrics = {}
                
    for category in categories:
        tp = category_corr_counts[category] # 预测对的数
        fp = category_pred_counts[category] # 该类别预测总数
        fn = category_counts[category] # 该类别的注释总数

        precision = tp / fp
        recall = tp / fn
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        category_metrics[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    total_support = sum(category_counts.values())
    weighted_f1 = sum(category_metrics[category]["f1"] * category_counts[category] / total_support for category in categories)
    
    p = corr_cnt/pred_cnt
    r = corr_cnt/pair_cnt
    f1 = 2*p*r/(p+r)

    print("F1: {:.4f}".format(f1)) # 0.6995
    print("WF1: {:.4f}".format(weighted_f1)) # 0.6985

def other_sentiment(sentiment_json):

    s_cnt = 0
    s_r = 0

    sentiment_list = ["positive", "negative"]
    label_list = ["Positive.", "Negative."]
    sentiment_correct = []
    sentiment_pred = []
    
    with open(sentiment_json, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            
            pred_sentiment_index = -1
            for i, sentiment in enumerate(sentiment_list):
                if sentiment in record["response"]:
                    pred_sentiment_index = i
                    break
            
            if pred_sentiment_index == -1:
                pred_sentiment_index = -1
            
            sentiment_pred.append(pred_sentiment_index)
            sentiment_correct.append(label_list.index(record["label"]))

            s_cnt += 1
            if label_list[pred_sentiment_index] == record["label"]:
                s_r += 1

    print(f"{s_r/s_cnt:.2%}")

def other_emotion():
    mosei_emotion_json = "/home/usr/liao/swift-mllm/output/emoverse-4b-mosei/infer_result/20241205-000127.jsonl"

    e_cnt = 0
    e_r = 0

    emotion_list = ["anger", "sadness", "surprise", "disgust", "happy", "neutral", "fear"]
    label_list = ["Anger.", "Sadness.", "Surprise.", "Disgust.", "Happy.", "Neutral.", "Fear."]
    emotion_correct = []
    emotion_pred = []
    
    with open(mosei_emotion_json, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            
            # 根据response中是否包含emotion_list中的情感词汇来确定预测的情感
            pred_emotion_index = -1
            for i, emotion in enumerate(emotion_list):
                if emotion in record["response"]:
                    pred_emotion_index = i
                    break
            
            # 如果没有找到对应的情感，则默认为Neutral.
            if pred_emotion_index == -1:
                pred_emotion_index = emotion_list.index("neutral")
            
            emotion_pred.append(pred_emotion_index)
            emotion_correct.append(label_list.index(record["label"]))

            e_cnt += 1
            if label_list[pred_emotion_index] == record["label"]:
                e_r += 1

    print(f"情感识别准确率: {e_r/e_cnt:.2%}")
    print(f"F1 Score (Weighted): {f1_score(emotion_correct, emotion_pred, average='weighted'):.4f}")

def find_numbers_in_string(input_string):
    matches = re.findall(r'\d+', input_string)
    numbers = [int(match) for match in matches]
    return numbers

def other_ecf2(ecf2_json):

    pair_cnt = 0
    corr_cnt = 0
    pred_cnt = 0
    matched_pairs = []
    all_pairs = []
    
    categories = ["disgust", "anger", "joy", "sadness", "surprise", "fear"]
    category_counts = {cat: 0 for cat in categories}
    category_corr_counts = {cat: 0 for cat in categories}
    category_pred_counts = {cat: 0 for cat in categories}
    with open(ecf2_json, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            labels = record["label"].split(";")
            if labels[0] == "":
                continue
            pair_cnt += len(labels)
            all_pairs.extend(labels) # 记录全部的pair

            for label in labels:
                category = label.split(",")[0].split("_")[1]
                if category in category_counts:
                    category_counts[category] += 1


            num_labels = [int(label.split(",")[1]) for label in labels]

            preds = find_numbers_in_string(record["response"])

            for pred in preds:
                if pred in num_labels:
                    matched_pairs.append(pred) # 记录正确的pair
                    corr_cnt += 1

                    # 获取num_labels的索引，然后对应到labels获取类别
                    right = labels[num_labels.index(pred)]
                    category = right.split(",")[0].split("_")[1]
                    if category in category_corr_counts:
                        category_corr_counts[category] += 1

                category = labels[0].split(",")[0].split("_")[1] # 这是它预测的索引
                if category in category_pred_counts:
                    category_pred_counts[category] += 1
                pred_cnt+=1
    
    
    category_metrics = {}
                
    for category in categories:
        tp = category_corr_counts[category] # 预测对的数
        fp = category_pred_counts[category] # 该类别预测总数
        fn = category_counts[category] # 该类别的注释总数

        precision = tp / fp
        recall = tp / fn
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        category_metrics[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    total_support = sum(category_counts.values())
    weighted_f1 = sum(category_metrics[category]["f1"] * category_counts[category] / total_support for category in categories)
    
    p = corr_cnt/pred_cnt
    r = corr_cnt/pair_cnt
    f1 = 2*p*r/(p+r)

    print("F1: {:.4f}".format(f1))
    print("WF1: {:.4f}".format(weighted_f1))

def compute_pos(mosei_sentiment_json,pos_neg_json):

    with open(pos_neg_json, 'r', encoding='utf-8') as file:
        pos_neg_data = json.load(file)

    pos_neg_images = [data["images"][0] for data in pos_neg_data] # 路径

    pos_cnt = 0
    neg_cnt = 0

    pos_rcnt = 0
    neg_rcnt = 0


    with open(mosei_sentiment_json, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            if record["images"][0] in pos_neg_images:

                if record["label"] == "Positive.":
                    pos_cnt+=1
                elif record["label"] == "Negative.":
                    neg_cnt+=1
                if record["response"] == record["label"]:

                    if record["response"] == "Positive.":
                        pos_rcnt+=1
                    elif record["response"] == "Negative.":
                        neg_rcnt+=1

    print(pos_rcnt/pos_cnt,neg_rcnt/neg_cnt) 
    print((pos_rcnt+neg_rcnt)/(pos_cnt+neg_cnt)) # 目前最佳结果为83%

def compute_other_pos(mosei_sentiment_json,pos_neg_json):

    with open(pos_neg_json, 'r', encoding='utf-8') as file:
        pos_neg_data = json.load(file)

    s_cnt = 0
    s_r = 083.87

    pos_neg_images = [data["images"][0] for data in pos_neg_data] # 路径
    sentiment_correct = []
    sentiment_pred = []

    sentiment_list = ["positive", "negative"]
    label_list = ["Positive.", "Negative."]

    with open(mosei_sentiment_json, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            if record["images"][0] in pos_neg_images:
                pred_sentiment_index = -1
                for i, sentiment in enumerate(sentiment_list):
                    if sentiment in record["response"]:
                        pred_sentiment_index = i
                        break
                
                if pred_sentiment_index == -1:
                    pred_sentiment_index = -1
                
                sentiment_pred.append(pred_sentiment_index)
                sentiment_correct.append(label_list.index(record["label"]))

                s_cnt += 1
                if label_list[pred_sentiment_index] == record["label"]:
                    s_r += 1

    print(f"{s_r/s_cnt:.2%}")