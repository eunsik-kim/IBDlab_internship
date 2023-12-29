import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import datetime, time, copy, argparse, os
import huggingface_hub
import wandb 

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator

from dataset import surveyQA, collate_fn, load_KOR_Dataset, load_ENG_Dataset
from model import prompt_SurveyClassifier, prompt_model
from utils import get_logger, drop_top_n_questions, plot_heatmap, top_n_importance
    
def main(arg):
    logger = get_logger(arg)
    # load_dataset
    tokenizer = AutoTokenizer.from_pretrained(arg.model_name)
    if arg.data_type =='eng':
        emb_id, question_type = load_ENG_Dataset(sparse_mode=arg.sparse_mode)
    else:
        emb_id, question_type = load_KOR_Dataset(sparse_mode=arg.sparse_mode)
    
    testing_mode = False # debugging
    if testing_mode:
        fraction = 1/11  # 11분의 1
        emb_id = emb_id.iloc[:int(len(emb_id) * fraction)]
    
    ID_list = emb_id['ID'].unique().tolist()
    label_list = emb_id['label'].tolist()[::427]
    X_train, X_test, _, _ = train_test_split(ID_list, label_list, test_size=0.2, shuffle=True, stratify=label_list, random_state=156)

    train_df = emb_id[emb_id['ID'].isin(X_train)].reset_index(drop=True)
    test_df = emb_id[emb_id['ID'].isin(X_test)].reset_index(drop=True)
    label_counts = {k: torch.tensor(v, dtype = torch.float32) for k, v in test_df['label'].value_counts().items()}
    train_dataset = surveyQA(train_df, tokenizer, recon=arg.recon_mode, ast_mode = arg.ast_mode, sparse_mode = arg.sparse_mode) # recon=False, ast_mode = False, sparse_mode = False
    test_dataset = surveyQA(test_df, tokenizer, recon=arg.recon_mode, ast_mode = arg.ast_mode, sparse_mode = arg.sparse_mode)
    training_dataloader = DataLoader(train_dataset, batch_size=arg.batch, shuffle=True, collate_fn= collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn= collate_fn)

    # model
    bert_model = AutoModel.from_pretrained(
                arg.model_name,
                output_attentions = False,
                output_hidden_states = False, 
                add_pooling_layer = False)
    bert_hidden_dim = 768
    total_query_num = 427
    label_nums = 4
    label_word = ['normal', 'think', 'plan', 'try']
    tokenizer_input_ids = tokenizer('normal, suicide think, suicide plan, suicide try', return_tensors='pt')['input_ids'][0]
    prompt_tuning_model = prompt_model(bert_model, tokenizer_input_ids, arg.n_tokens, initialize_from_input = True)
    model = prompt_SurveyClassifier(prompt_tuning_model, hidden_dims=[total_query_num * bert_hidden_dim, label_nums], avg = True) # avg = True, max = False, cls = False, ast = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Trainable Parameters: {total_trainable_params}/{total_params}({total_trainable_params/total_params*100:.2f}%)")
    best_accuracy, best_epoch, best_model_state = 0, 0, False
    total_step = len(training_dataloader)
    logging_step = int(total_step * 0.1)
    saving_step = 10

    start_epoch = 1
    num_epochs = 30
    accelerator = Accelerator(mixed_precision = 'fp16', gradient_accumulation_steps=4)
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(model, optimizer, training_dataloader, scheduler)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = accelerator.device

    # arg.skip_mode:
    drop_df = pd.DataFrame()
    drop_indexes = []

    logger.info('학습시작')
    for epoch in range(start_epoch, num_epochs+1):
        model.train() 
        total_loss = 0
        stime = time.time()
        epochtime = 0
        for step, (input, label) in enumerate(training_dataloader, start = 1):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                output = model(drop_indexes, **input)  
                loss = criterion(output, label.squeeze(1))  
                accelerator.backward(loss)
                optimizer.step()
                total_loss += loss.item()
                if step % logging_step == 0:
                    ctime = time.time() - stime
                    stime = time.time()
                    epochtime += ctime
                    logger.info(f"[{step}/{total_step}], Loss: {loss}, Time: {ctime:.2f}")
        wandb.log({'train_epoch_loss': total_loss / len(training_dataloader)}, step=epoch)            
        logger.info(f"Epoch {epoch}, AVG Train Loss: {total_loss / len(training_dataloader)}, Total time cost: {epochtime:.2f}")
        scheduler.step()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        labels, preds = [], []
        with torch.no_grad():
            model.eval()
            surveys = {word: torch.zeros(total_query_num * bert_hidden_dim).to(device) for word in label_word}
            for input, label in test_dataloader:
                for k, v in input.items():
                    input[k] = v.to(device)
                output, survey = model(drop_indexes, **input)
                for idx, l in enumerate(label.reshape(-1)):
                    surveys[label_word[l]] += survey[idx]
                label = label.to(device)  
                loss = criterion(output, label.squeeze(1))
                pred = torch.argmax(output, axis = 1)
                correct = (pred == label.reshape(-1)).sum().item()
                sample = len(label)
                total_correct += correct
                total_samples += sample
                total_loss += loss.item()
                labels.extend(label.reshape(-1).detach().cpu().tolist())
                preds.extend(pred.reshape(-1).detach().cpu().tolist())
            epoch_accuracy = total_correct / total_samples*100
            logger.info(f"Epoch {epoch}, Test Accuracy : {epoch_accuracy:.2f}%, Test loss : {total_loss}")
            wandb.log({'Test_Accuracy': epoch_accuracy, 'Test_loss' : total_loss}, step=epoch)

            plt = plot_heatmap(labels, preds)
            wandb_image = wandb.Image(plt)
            wandb.log({"Heatmap": wandb_image}, step=epoch)
            
            if epoch % 3 == 0:
                for word in label_word:
                    word_index = label_word.index(word)
                    surveys[word] /= label_counts[word_index].to(device)
                    if word_index in label_counts.keys():
                        df_questions = top_n_importance(model, surveys[word], word_index, question_type, emb_id, n=10)
                        table = wandb.Table(dataframe=df_questions)
                        wandb.log({word: table}, step=epoch)

            if arg.skip_mode and epoch % arg.drop_epoch == 0:
                new_drop_df, unique_n_index = drop_top_n_questions(model, surveys, question_type, emb_id, n=arg.n_drop)
                drop_indexes.extend(unique_n_index)
                logger.info(f"{len(drop_indexes)} questions are dropped from dataset")
                drop_df = pd.concat([drop_df, new_drop_df], axis=0)
                table = wandb.Table(dataframe=drop_df)
                wandb.log({'drop queries': table}, step=epoch)

            if epoch_accuracy > best_accuracy:
                best_accuracy= epoch_accuracy
                best_epoch= epoch
                best_model_state = copy.deepcopy(model.state_dict())  
                best_model_state = {key: value.cpu() for key, value in best_model_state.items()} 
        
        if epoch % saving_step == 0:
            if not os.path.exists(arg.data_dir):
                os.makedirs(arg.data_dir, exist_ok=True)
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%m-%d_%H-%M")
            model_name = f"{epoch}epoch_{formatted_time}"
            save_dir = os.path.join(arg.data_dir, model_name)
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({'model': unwrapped_model.state_dict(), 
                        'optimizer': optimizer.optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'best_epoch': best_epoch,
                        'best_acc' : best_accuracy}, save_dir + '.pt')
            if best_model_state:
                torch.save(best_model_state, os.path.join(os.path.join(arg.data_dir), f'{best_epoch}epoch_{int(best_accuracy)}acc_Bestmodel.pt'))
            logger.info(save_dir + '.pt' + " 저장 완료")
    logger.info("학습 종료")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default="")
    parser.add_argument('--seed', default=100, type=int, help='seed for initializing training.')
                        
    parser.add_argument("--data_dir", type=str, default='../03_model/mentalbertast_ft_sparse')
    parser.add_argument("--data_type", type=str, default='eng')
    parser.add_argument('--model_name', type=str, default="mental/mental-bert-base-uncased")

    parser.add_argument('--batch', default=2, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0001, type=float, metavar='N', help='learning rate')
    parser.add_argument('--local_rank', type=int)
    
    # test
    parser.add_argument("--sparse_mode", action='store_true')
    parser.add_argument("--recon_mode", action='store_true')
    parser.add_argument("--avg_mode", action='store_true')
    parser.add_argument("--max_mode", action='store_true')
    parser.add_argument("--ast_mode", action='store_true')
    parser.add_argument("--cls_mode", action='store_true')
    
    parser.add_argument("--skip_mode", action='store_true')
    parser.add_argument("--drop_epoch", type=int, default = 5)
    parser.add_argument("--n_drop", type=int, default=30)
    parser.add_argument("--n_tokens", type=int, default=20)

    arg = parser.parse_args()    
    wandb.init(project= 'survey_classification')
    wandb.config.update(arg)
    wandb.run.name = arg.data_dir.split('/')[-1]
    if not arg.token:
        ValueError('You need to input arg \'--token <your_token>\'. Get your token in (https://huggingface.co/settings/tokens)')
    torch.manual_seed(arg.seed)    
    huggingface_hub.login(arg.token) # to use mental bert
    main(arg)

