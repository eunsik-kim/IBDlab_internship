import torch
import datetime, os, logging 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

def get_logger(arg):
    formatter = logging.Formatter('||%(asctime)s||%(levelname)s||%(message)s||')
    log_dir = os.path.join(arg.data_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, datetime.datetime.now().strftime("%m-%d_%H-%M") + '.log'), mode = 'a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger

# 정답을 맞추는데 기여한 top-n 순서
def top_n_importance(model, surveys, word_index, question_type, emb_id, n=10):
    weights = torch.stack([torch.matmul(model.layers[0].weight[:, 768*i: 768*(i+1)], surveys[768*i: 768*(i+1)]) for i in range(427)], dim=0).cpu()
    top_n_indices = np.argsort(np.array(weights), axis=0)[-n:,:][::-1]
    questions_list = np.array(emb_id['question'][:427])
    top_n_questiontypes = np.take_along_axis(question_type, top_n_indices[:, word_index], axis=0)
    top_n_questions = np.take_along_axis(questions_list, top_n_indices[:, word_index], axis=0)
    top_n_weights =np.take_along_axis(np.array(weights)[:, word_index], top_n_indices[:, word_index], axis=0)
    logdf = pd.DataFrame({'qtype': top_n_questiontypes, 'question': top_n_questions, 'weight':top_n_weights})
    return logdf

# 정답을 틀리는데 기여한 top-n 순서
def drop_top_n_questions(model, surveys, question_type, emb_id, n=30):
    weights = {word:0 for word in ['normal', 'think', 'plan', 'try']}
    top_n_list = []
    for word_index, word in enumerate(weights.keys()):
        weights[word] = torch.stack([torch.matmul(model.layers[0].weight[:, 768*i: 768*(i+1)], surveys[word][768*i: 768*(i+1)]).data for i in range(427)], dim=0).cpu()
        weights[word] = torch.softmax(weights[word], dim = -1)
        weights[word] = weights[word] / (weights[word][:, word_index].unsqueeze(1)+ 1)
        weights[word] = torch.cat((weights[word][:, :word_index], weights[word][:, word_index + 1:]), dim=1)
        values = np.max(np.array(weights[word]), axis=1)
        top_n_indices = np.argsort(values)[-n:][::-1]
        top_n_list.extend([(value, indices) for value, indices in zip(np.take_along_axis(values, top_n_indices, axis = 0), top_n_indices)])
    sorted_n_list = sorted(top_n_list, reverse=True, key= lambda x:x[0])
    unique_n_index = [t[1] for i, t in enumerate(sorted_n_list) if t[1] not in [x[1] for x in sorted_n_list[:i]]][:n]
    unique_n_weight = [t[0] for i, t in enumerate(sorted_n_list) if t[1] not in [x[1] for x in sorted_n_list[:i]]][:n]

    questions_list = np.array(emb_id['question'][:427])
    top_n_questiontypes = np.take_along_axis(question_type, np.array(unique_n_index), axis=0)
    top_n_questions = np.take_along_axis(questions_list, np.array(unique_n_index), axis=0)
    top_n_weights =np.array(unique_n_weight)
    dropdf = pd.DataFrame({'qtype': top_n_questiontypes, 'question': top_n_questions, 'weight':top_n_weights})
    return dropdf, unique_n_index

def plot_heatmap(label, preds):
    cm = confusion_matrix(label, preds)
    unique_list = np.unique(np.array([label, preds]))
    accuracy = sum([cm[i, i] for i in range(len(unique_list))]) / sum(sum(cm))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', xticklabels=unique_list, yticklabels=unique_list)
    plt.title(f"Accuracy: {accuracy:.2f}")
    plt.xlabel('Predict')
    plt.ylabel('label')
    plt.title(f'Confusion Matrix\n Accuracy : {accuracy*100:.2f}%')
    return plt