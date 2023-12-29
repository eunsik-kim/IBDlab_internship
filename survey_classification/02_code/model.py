import torch
import torch.nn as nn

# sentence bert with classifier head
class SurveyClassifier(torch.nn.Module):
    def __init__(self, sentencebert, hidden_dims = [427 * 768,  3], avg = True, max = False, cls = False, ast = False):
        super().__init__()
        self.avg, self.max, self.cls, self.ast = avg, max, cls, ast
        self.bert = sentencebert
        self.pool = torch.nn.Linear(768, 1)
        self.flatten = torch.nn.Flatten()
        self.layers = torch.nn.Sequential()
        depth = len(hidden_dims)-1
        for idx in range(depth-1):
            self.layers.extend([torch.nn.Linear(hidden_dims[idx], hidden_dims[idx+1]),
                                torch.nn.ReLU(),
                                torch.nn.Dropout()])
            torch.nn.init.kaiming_uniform_(self.layers[-3].weight, mode='fan_in', nonlinearity='relu')
        self.layers.append(torch.nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        torch.nn.init.kaiming_uniform_(self.layers[-1].weight, mode='fan_in', nonlinearity='relu')
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    
    def forward(self, drop_indexes, **input):
        batch_size, _, _ = input['input_ids'].shape
        surveys = []
        for j in range(batch_size):
            out = self.bert(input['input_ids'][j])
            if j in drop_indexes:
                for k, v in out.items():
                    out[k] = torch.zeros_like(v)
            if not self.cls:
                if self.ast:
                    atten_mask = input['answer_mask'][j]
                else:
                    atten_mask = input['attention_mask'][j]
                x = out['last_hidden_state']
                if self.avg:
                    pooled_vector = self.mean_pooling(x, atten_mask)
                else:
                    pooled_vector = self.max_pooling(x, atten_mask)
            else:
                pooled_vector = out['pooler_output']
            surveys.append(pooled_vector)
        surveys = torch.stack(surveys, dim=0)
        surveys = self.flatten(surveys)

        if not self.training:  
            return self.layers(surveys), surveys
        return self.layers(surveys)

# additional prompt embedding layer for prompt tuning
# https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py
class PROMPTEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = False,
                initialize_from_input: bool = False,
                tokenizer_input_ids= None):
        super(PROMPTEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab,
                                                                               initialize_from_input,
                                                                               tokenizer_input_ids))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = False,
                             initialize_from_input: bool = False,
                             tokenizer_input_ids= None):
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        if initialize_from_input:
            length = tokenizer_input_ids.shape[0]
            if length < n_tokens:
                    tokenizer_input_ids = tokenizer_input_ids.repeat(n_tokens // length +1)
            return self.wte.weight[tokenizer_input_ids[:n_tokens]].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        input_embedding = self.wte(tokens)
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)

# bert model that expands the embedding layer by n_tokens  
class prompt_model(nn.Module):
    def __init__(self, bert_model, tokenizer_input_ids, n_tokens = 20, initialize_from_vocab = False, initialize_from_input= True):
        super().__init__()
        self.bert_model = bert_model
        self.freeze_bert()
        self.n_tokens = n_tokens
        self.bert_model.set_input_embeddings(PROMPTEmbedding(bert_model.get_input_embeddings(), n_tokens=self.n_tokens, 
                                                             initialize_from_vocab=initialize_from_vocab, 
                                                             initialize_from_input=initialize_from_input,
                                                             tokenizer_input_ids = tokenizer_input_ids))
    
    def forward(self, **input):
        input['attention_mask'] = torch.concat([torch.ones(input['input_ids'].shape[0], self.n_tokens).to('cuda'), input['attention_mask']], dim =1)
        input['token_type_ids'] = torch.zeros((input['input_ids'].shape[0], input['input_ids'].shape[1] + self.n_tokens), dtype=torch.long).to('cuda')
        input['position_ids'] = torch.arange(512).expand((1, -1))[:, :input['input_ids'].shape[1]+ self.n_tokens].to('cuda')
        return self.bert_model(**input) 
    
    def freeze_bert(self,):
        for param in self.bert_model.parameters():
            param.requires_grad = False

# SurveyClassifier having diffrent forward implementation
class prompt_SurveyClassifier(torch.nn.Module):
    def __init__(self, sentencebert, hidden_dims = [427 * 768,  3], avg = True, max = False, cls = False, ast = False):
        super().__init__()
        self.avg, self.max, self.cls, self.ast = avg, max, cls, ast
        self.bert = sentencebert
        self.pool = torch.nn.Linear(768, 1)
        self.flatten = torch.nn.Flatten()
        self.layers = torch.nn.Sequential()
        depth = len(hidden_dims)-1
        for idx in range(depth-1):
            self.layers.extend([torch.nn.Linear(hidden_dims[idx], hidden_dims[idx+1]),
                                torch.nn.ReLU(),
                                torch.nn.Dropout()])
            torch.nn.init.kaiming_uniform_(self.layers[-3].weight, mode='fan_in', nonlinearity='relu')
        self.layers.append(torch.nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        torch.nn.init.kaiming_uniform_(self.layers[-1].weight, mode='fan_in', nonlinearity='relu')
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    
    def forward(self, drop_indexes, **input):
        batch_size, _, _ = input['input_ids'].shape
        surveys = []
        for j in range(batch_size):
            input_j = {k:v[j] for k, v in input.items()}
            out = self.bert(**input_j)
            atten_mask = torch.concat([torch.ones(input_j['input_ids'].shape[0], self.bert.n_tokens).to('cuda'), input_j['attention_mask']], dim =1)
            x = out['last_hidden_state']
            pooled_vector = self.mean_pooling(x, atten_mask)
            surveys.append(pooled_vector)
        surveys = torch.stack(surveys, dim=0)
        surveys = self.flatten(surveys)

        if not self.training:  
            return self.layers(surveys), surveys
        return self.layers(surveys)
