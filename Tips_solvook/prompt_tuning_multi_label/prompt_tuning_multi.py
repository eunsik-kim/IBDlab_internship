import os, argparse
import torch
import pandas as pd
import huggingface_hub
import transformers
import wandb
import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PromptEmbedding, PromptTuningConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from utils import get_dataset, get_test_dataset, preprocess_function, collate_fn, pred_parse

def main(arg, device):
    # load dataset
    dataset = get_dataset(arg)
    raw_test_dataset = get_test_dataset(arg)

    # load skill_list, method_list 
    all_sheet = pd.read_excel(arg.excel_file, sheet_name= None)
    handout_type = all_sheet['5. handout_type(81 회의 이후 변경)']
    handout_type.columns = handout_type.iloc[0]
    handout_type = handout_type[1:]
    handout_type.reset_index(drop= True, inplace=True)
    skill_dict, method_dict = {}, {}
    
    for idx in range(len(handout_type)):
        row = handout_type.iloc[idx]
        skill_dict[row['skill #']] = row['skill (2depth)']
        method_dict[row['method #']] = row['method (2depth) 영어']

    skill_list = list(skill_dict[key] for key in sorted(skill_dict.keys()))
    method_list = list(method_dict[key] for key in sorted(method_dict.keys()))
    
    # preprocess dataset
    pretrained_model_name_or_path = arg.pretrained_model_name # "meta-llama/Llama-2-7b-chat-hf" https://huggingface.co/blog/llama2
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 
        
    processed_datasets = dataset.map(partial(preprocess_function, skill_dict = skill_dict, method_dict = method_dict, tokenizer = tokenizer),
                            batched=True,
                            num_proc=1,
                            remove_columns=dataset["train"].column_names,
                            keep_in_memory = True,
                            desc="Tokenizing on dataset",
                            )
    test_dataset = raw_test_dataset.map(partial(preprocess_function, skill_dict = skill_dict, method_dict = method_dict, tokenizer = tokenizer, is_testset = True),
                            batched=True,
                            num_proc=1,
                            remove_columns=dataset["train"].column_names,
                            keep_in_memory = True,
                            desc="Tokenizing on Test dataset",
                            )

    train_dataloader = DataLoader(processed_datasets['train'], shuffle=True, collate_fn=collate_fn, batch_size=arg.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(processed_datasets['eval'], shuffle=False, collate_fn=collate_fn, batch_size=arg.batch_size, pin_memory=True)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
                            pretrained_model_name_or_path = pretrained_model_name_or_path,
                            return_dict=True,
                            low_cpu_mem_usage=True,)

    config = PromptTuningConfig(task_type="CAUSAL_LM", 
                                prompt_tuning_init="TEXT",
                                num_virtual_tokens=arg.virtual_tokens,
                                prompt_tuning_init_text="[1.vocabulary, 2.grammar, 3.expression, 4.content, 5.context], [1. 내용 해석하기 (영-한 변환), 2. 어휘 쓰기 및 찾기, 3. 문장 쓰기, 4. 밑줄 친 부분 고쳐쓰기, 5. 선택지 내 요소들이 모두 맞는 것 찾기, 6. 내용과의 일치 여부 판단하기, 7. 요지 찾기, 8. 유추하기, 9. 주제문 찾기, 10. 순서 배열하기, 11. (글의 흐름에 맞게) 문장 배치하기, 12. 연결어 찾기, 13. 정오 여부 판단하기, 14. 잘못된 것 고치기, 15. 유사 여부 판단하기, 16. 관련 없는 문장 찾기]", # prompt 를 초기화 할때 넣는 임의의 text
                                tokenizer_name_or_path=pretrained_model_name_or_path,)
    model = get_peft_model(model, config)

    # train
    start_epoch = 1
    optimizer = torch.optim.Adam(model.parameters()) # Adafactor weight decay 1e−5, β2 decay 0.8
    model = model.to(device)

    smlb = MultiLabelBinarizer(classes=skill_list)
    skill_tar_bin = smlb.fit_transform(test_dataset['test']['skill'])
    mmlb = MultiLabelBinarizer(classes=method_list)
    method_tar_bin = mmlb.fit_transform(test_dataset['test']['method'])

    for epoch in range(start_epoch, arg.epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()

        model.eval()
        eval_loss = 0
        for batch in tqdm(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
            eval_loss += loss.detach().float()
            
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        train_epoch_loss = total_loss / len(train_dataloader)
        
        # verify output
        example_input= tokenizer(
            f"[INST] <<SYS>>\n영어문제가 주어진다. 지시사항에 답하여라. \n<<SYS>>\n\n 다음의 질문에 해당되는 'skill'과 'method'를 답하여라. 질문 : 밑줄 친 ➀ ~ ➄ 중 어법상 틀린 것을 찾아 바르게 고치시오., 정답 = [/INST]",
            return_tensors="pt").to(device)
        out = model.generate(input_ids=example_input["input_ids"], attention_mask=example_input["attention_mask"], max_new_tokens=30, eos_token_id=3)
        gen_text = tokenizer.decode(out[0].detach().cpu().numpy(), skip_special_tokens=True)
        label_text = "[INST] <<SYS>>\n영어문제가 주어진다. 지시사항에 답하여라. \n<<SYS>>\n\n 다음의 질문에 해당되는 'skill'과 'method'를 답하여라. 질문 : 밑줄 친 ➀ ~ ➄ 중 어법상 틀린 것을 찾아 바르게 고치시오., 정답 = [/INST] skill-method는 skill: grammar, method: find the correct / find the incorrect 또는 skill: content, method: correct the underlined이다. </s>"
        text_table = wandb.Table(columns=["epoch", 'gen_text', 'label_text'])
        text_table.add_data(epoch, gen_text, label_text)
        wandb.log({"generation_example": text_table})

        skill_pred, method_pred = [], []
        with torch.no_grad():
            for batch in test_dataset['test']:
                input_ids = torch.tensor([batch['input_ids']]).to(device)
                attention_mask= torch.tensor([batch['attention_mask']]).to(device)
                input_length = len(tokenizer.decode(input_ids[0]))
                out = model.generate(input_ids = input_ids, attention_mask = attention_mask, max_new_tokens= 20, eos_token_id=3)
                gen_sentence = tokenizer.decode(out[0].detach().cpu().numpy(), skip_special_tokens=True)
                skill_txt_preds, _ = pred_parse(gen_sentence[input_length:], pats = skill_list)
                method_txt_preds, _ = pred_parse(gen_sentence[input_length:], pats = method_list)
                skill_pred.append(skill_txt_preds)
                method_pred.append(method_txt_preds)
                
        # logging scores
        skill_pred_bin = smlb.transform(skill_pred)
        method_pred_bin = mmlb.transform(method_pred)
        
        metrics = [precision_score, recall_score, f1_score]
        averages = ['micro', 'macro', 'weighted']
        skill_metrics, method_metrics = [], []
        for metric in metrics:
            for average_type in averages:
                skill_metrics.append(metric(skill_tar_bin, skill_pred_bin, average=average_type))
                method_metrics.append(metric(method_tar_bin, method_pred_bin, average=average_type))
        output_dict = {}
        i = 0

        for metric_name in ['precision_score_', 'recall_score_', 'f1_score_']:
            for average_type in averages:
                output_dict['skill_'+metric_name+average_type] = skill_metrics[i]
                output_dict['method_'+metric_name+average_type] = method_metrics[i]
                i += 1

        output_dict.update({'train_epoch_loss':train_epoch_loss, 'eval_epoch_loss': eval_epoch_loss}) 
        wandb.log(output_dict, step=epoch)

        if epoch % 10 == 0:
            current_time = datetime.datetime.now()
            time_format = "%m-%d_%H-%M"
            formatted_time = current_time.strftime(time_format)
            model_name = f"{epoch}epoch_{formatted_time}.pt"
            save_dir = os.path.join(arg.data_dir, 'prompt_'+model_name)
            torch.save({'model_state_dict':model.state_dict(), 'epoch': epoch}, save_dir)
            print('Save completely')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=False)
    parser.add_argument('--seed', default=100, type=int, help='seed for initializing training. ')
    parser.add_argument('--device', type=str, default='cuda')
                        
    parser.add_argument("--data_dir", type=str, default='./')
    parser.add_argument('--pretrained_model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--pretrained_model_dir', type=str, default=False, help='pretrained model ckpt')
    parser.add_argument('--virtual_tokens', type=int, default=100, help='prompt length')

    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--ver', type=str, default='prompt_tuning1')
    
    arg = parser.parse_args(args=[])

    # setting default arg
    arg.token = ""
    arg.pretrained_model_name = "bigscience/bloomz-560m"
    arg.train_file = 'solvook_handout_tr.csv'
    arg.eval_file = 'solvook_handout_val.csv'
    arg.test_file = 'solvook_handout_te.csv'
    arg.excel_file = './100_Solvook_handout_DB_english.xlsx'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project= 'prompt_tuning')
    wandb.config.update(arg)
    wandb.run.name = 'Tips_prompt_tuning/bloomz-560m/multi_label'
    if not arg.token:
        ValueError('You need to input arg \'--token <your_token>\'. Get your token in (https://huggingface.co/settings/tokens)')
    torch.manual_seed(arg.seed)
    huggingface_hub.login(arg.token)
    main(arg, device)