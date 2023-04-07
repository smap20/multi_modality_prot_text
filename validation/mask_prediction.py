
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoConfig, AutoModelForMaskedLM
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import spearmanr
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mask_len', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--mask_prot', action='store_true')

parser.add_argument('--mask_keyword', action='store_true')

parser.add_argument('--k', type=int, default=10)

parser.add_argument('--model_path', type=str, default='/m-ent1/ent1/smap20/checkpoints/pubmedbert')

parser.add_argument('--validation_file', type=str)

parser.add_argument('--amino', type=str)

parser.add_argument('--keyword_file', type=str, default='/m-ent1/ent1/smap20/data/treeNumberdict_disease.txt')



args = parser.parse_args()

def mask_target_phrase(sentence, target_word):
    masked_sentence = sentence
    # Create a regular expression pattern to match the target phrase, ignoring case
    pattern = re.compile(re.escape(target_word), re.IGNORECASE)
    # Replace all occurrences of the target phrase in the sentence with the '[MASK]' token
    masked_sentence = pattern.sub('[MASK]', masked_sentence)
    return masked_sentence

def Mask_prot(text, mask_len):
    text_ = text.split()
    prot_pos = []
    pattern = r'<[A-Z]>'
    for i in range(min(len(text_), 512)):
        if re.match(pattern, text_[i]):
            prot_pos.append(i)
    rand = np.random.randint(0, len(prot_pos)-mask_len)
    mask = []
    for i in range(mask_len):
        mask.append(text_[prot_pos[rand+i]])
        text_[prot_pos[rand+i]] = '[MASK]'
    text = ' '.join(text_)
    mask = ''.join([str(i) for i in mask])
    return text, mask

def load_model(model_path, from_huggingface=False):
    if from_huggingface:
        print ('Loading from huggingface')
        config = AutoConfig.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', output_hidden_states=True)
        model = AutoModelForMaskedLM.from_config(config)
        tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', do_lower_case=True)
    
    else:
        print ('Loading from local')
        config = AutoConfig.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', output_hidden_states=True)
        config.vocab_size = 30543 # 31111
        
        model = AutoModelForMaskedLM.from_config(config)

        model.load_state_dict(torch.load(model_path+'/pytorch_model.bin'))

        tokenizer = BertTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

model, tokenizer = load_model(args.model_path, from_huggingface=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

with open(args.validation_file, 'r') as f:
    textlines = f.readlines()

if args.mask_prot:
    with open(args.amino, 'r') as f:
        animo_data = f.readlines()
    animo_data = ['<'+i.strip()+'>' for i in animo_data]

    choices = {}
    for amino in animo_data:
        choices[amino] = 0


keywords = []
treeNumberdict = {} 
if args.mask_keyword:
    with open(args.keyword_file, 'r') as f:
        tree_dict = f.readlines()
    for line in tree_dict:
        line = line.strip().split('|')
        treeNumberdict[line[0]] = line[1]
        keywords.append(line[0])

    choices_for_keywords = [keyword.strip() for keyword in keywords]

    choices_len = len(choices_for_keywords)

    choice_embedding = torch.zeros(choices_len, 768)

    for keyword in choices_for_keywords:
        token = tokenizer.tokenize(keyword)
        input_id = tokenizer.convert_tokens_to_ids(token)
        input_tensor = torch.tensor([input_id])
        input_tensor = input_tensor.to(device)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            embedding = output.hidden_states[-1][0]
            norm = torch.norm(embedding)
            embedding = embedding/norm
            choice_embedding[choices_for_keywords.index(keyword)] = embedding[0]

    choice_embedding = choice_embedding.transpose(0,1)
    choice_embedding = choice_embedding.to(device)
        
auroc_list = []

acc = 0
cnt = 0

top1 = 0
top5 = 0
top10 = 0


t_bar = tqdm(range(0, len(textlines), args.batch_size))


for i in t_bar:
    text_lines = textlines[i:i+args.batch_size]
    if args.mask_prot:
        text = []
        amino_lable = []
        for textline in text_lines:
            _text, _amino_lable = Mask_prot(textline, args.mask_len)
            text.append(_text)
            amino_lable.append(_amino_lable)
        
        if text == []:
            break
        encoded_sequences = tokenizer.batch_encode_plus(text, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt',max_length=512)
        mask_positions = torch.nonzero(encoded_sequences['input_ids'] == tokenizer.mask_token_id, as_tuple=True)
        try:
            if max(mask_positions[1]) > 512-args.mask_len or len(mask_positions[1])/args.mask_len != args.batch_size:
                continue
        except:continue
    

        encoded_sequences = encoded_sequences.to(device)
        model.eval()
        # Predict all tokens
        with torch.no_grad():
            outputs = model(**encoded_sequences)
            predictions = outputs.logits[mask_positions[0], mask_positions[1], :]
            predicted_token_ids = torch.argmax(predictions, dim=1)
            predicted_tokens = tokenizer.batch_decode(predicted_token_ids.tolist())
        for j in range(len(predicted_tokens)):
            predicted_tokens[j] = predicted_tokens[j].replace(' ', '')
        for j in range(0,len(text_lines)):
            if ''.join(predicted_tokens[j*args.mask_len:j*args.mask_len+args.mask_len]) == amino_lable[j]:
                acc += 1


            cnt += 1
        
        t_bar.set_description('acc: %f ' % (acc/cnt))
    
    elif args.mask_keyword:

        text = []
        word_lable = []
        for textline in text_lines:
            word_lable_, text_ = textline.split('|')
            text_ = mask_target_phrase(text_, word_lable_)
            text.append(text_)
            word_lable.append(word_lable_)


        
        try:encoded_sequences = tokenizer.batch_encode_plus(text, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt',max_length=512)
        except:from IPython import embed; embed()
        mask_positions = torch.nonzero(encoded_sequences['input_ids'] == tokenizer.mask_token_id, as_tuple=True)
        try : 
            if len(mask_positions[1]) != args.batch_size or max(mask_positions[1]) > 511:
                print (len(mask_positions[1]), args.batch_size, max(mask_positions[1]))
                continue
        except:
            from IPython import embed; embed()

        encoded_sequences = encoded_sequences.to(device)
        model.eval()
        # Predict all tokens
        with torch.no_grad():
            outputs = model(**encoded_sequences)
            embeddings = outputs.hidden_states[-1][mask_positions[0], mask_positions[1], :]
            scores = torch.matmul(embeddings, choice_embedding)
            prob = torch.softmax(scores, dim=1)
            pred = torch.argmax(scores, dim=1)
            topk = torch.topk(scores, args.k, dim=1)
            true = torch.zeros_like(prob)

            predicted_tokens = [[choices_for_keywords[q] for q in p] for p in topk.indices]

            for b in range(args.batch_size):
                for j in range(len(choices_for_keywords)):
                    # if match_in_tree(word_lable[b], choices_for_keywords[j]) == True:
                    if word_lable[b] == choices_for_keywords[j]:
                        true[b][j] = 1

            
            for b in range(args.batch_size):
                a = roc_auc_score(true[b].cpu(), prob[b].cpu())
                # print (a)
                # if a >= 0.6 or a <= 0.4:
                #     from IPython import embed; embed()

                auroc_list.append(a)
                        



        
        for j in range(0,len(text_lines)):
            for c in range (args.k):
                # if match_in_tree(word_lable[j], predicted_tokens[j][c]) == True:
                if word_lable[j] == predicted_tokens[j][c]:
                    if c == 0:
                        top1 += 1
                    if c <= 5:
                        top5 += 1
                    if c <= 10:
                        top10 += 1

                    print (predicted_tokens[j], word_lable[j])
                    break

            cnt += 1
        if cnt % 1000 == 0:
            print (top1/cnt, top5/cnt, top10/cnt)

        t_bar.set_description('auroc: %f' % (sum(auroc_list)/len(auroc_list)))




            
if args.mask_prot:
    print('mask_len: %d, accuracy: %f' % (args.mask_len, acc/cnt))

if args.mask_keyword:
    auroc = sum(auroc_list)/len(auroc_list)
    print ('top1: %f, top5: %f, top10: %f' % (top1/cnt, top5/cnt, top10/cnt))
    print ('auroc: %f' % auroc)

