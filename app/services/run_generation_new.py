from transformers import GPT2LMHeadModel, GPT2Tokenizer
from app.utils import catch_exceptions,download_from_s3
from constants import model_mapping
import torch
import torch.nn.functional as F
from config import Config
from tqdm import trange
import numpy as np
import pickle
import requests
import logging
import json
import os
import re
import nltk
import gc
import copy

nltk.download('wordnet')
nltk.download('stopwords')

logger = logging.getLogger(name=__name__)

MAX_LENGTH = int(300)  # Hardcoded max length to avoid infinite loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

class GenerateDescription(object):

    def __init__(self):
        pass
        
    @catch_exceptions
    def check_file_exists(self, file_name):
        try:
            abs_file_path = "nlg-models/"+file_name
            s3_path = "text_generation/GPT2_Models/"+file_name
            if not os.path.exists(abs_file_path):
                os.makedirs("/".join(abs_file_path.split('/')[:-1]), exist_ok=True)
                download_from_s3(s3_path,abs_file_path)
            return abs_file_path
        except Exception as e:
            logger.error(e,exc_info=True)

    @catch_exceptions
    def keywords_check_file_exists(self,request_data,file_name):
        try:
            abs_file_path = 'ai-models/vdezi' + '/' + request_data['category'] + '_' + file_name
            s3_path = 'ner_models/vdezi' +'/'+ request_data['category'] + '_' + file_name
            if not os.path.exists(abs_file_path):
                os.makedirs("/".join(abs_file_path.split('/')[:-1]), exist_ok=True)
                download_from_s3(s3_path,abs_file_path)
            return abs_file_path
        except Exception as e:
            logger.error(e,exc_info=True)


    @catch_exceptions
    def extract_topn_from_vector(self,feature_names, sorted_items, topn=10):   
        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []
        for idx, score in sorted_items:
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
            # print(score_vals)
        return results.keys()

    @catch_exceptions
    def sort_coo(self,coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    @catch_exceptions
    def get_keywords(self,request_data):
        try:
            vectorizer_path = self.keywords_check_file_exists(request_data,'vectorizer.pickle')
            vectorizer = pickle.load(open(vectorizer_path,'rb'))
            model_path = self.keywords_check_file_exists(request_data,'tfidf_transformer.pickle')
            tfidf_transformer = pickle.load(open(model_path,'rb'))
            tf_idf_vector = tfidf_transformer.transform(
                vectorizer.transform(
                    [
                        request_data["product_title"]
                    ]
                    )
                )
            feature_names=vectorizer.get_feature_names()
            sorted_items = self.sort_coo(tf_idf_vector.tocoo())
            keywords = self.extract_topn_from_vector(feature_names,sorted_items,10)           
            del vectorizer,model_path,tfidf_transformer,tf_idf_vector,feature_names,sorted_items
            torch.cuda.empty_cache()
            return list(keywords)
        except Exception as e:
            logger.error(e,exc_info=True)
            
    @catch_exceptions
    def load_model(self, request_data):
        try:
            supporting_file_names = [ "/config.json",  "/eval_results.txt",  "/merges.txt",  "/pytorch_model.bin",  "/special_tokens_map.json",  "/tokenizer_config.json",  "/training_args.bin",  "/vocab.json"]
            for file_name in supporting_file_names:
                self.check_file_exists( request_data["category"]+file_name)
                
        except Exception as e:
            logger.error(e,exc_info=True)
            
    @catch_exceptions
    def remove_unit_values(self, title):
        try:
            pre_hashing = ["Ohm","V","volt","Hz","W","g","oz","mm","Millimeter","m","ounce","cm","centimeter","centimeters","inches","cms","meters","meter","in","mtrs","mtr","feet","Foot","ft","inch"]
            pre_hash_list = []
            post_hash_list = []
            
            for each_unit in pre_hashing:
                matches = re.finditer(each_unit, title, re.IGNORECASE) #https://stackoverflow.com/questions/500864/case-insensitive-regular-expression-without-re-compile
                matches_positions = [match.start() for match in matches] #https://stackoverflow.com/questions/3519565/find-the-indexes-of-all-regex-matches
                pre_hash_list.extend(matches_positions)
                
            for each_start_index in pre_hash_list:
                for i in range(each_start_index-1, 0, -1):
                    if title[i].isdigit():
                        title = title[:i] + "#" + title[i + 1:]
                    elif title[i]=="*" or title[i]==' ' or title[i]=="x" or title[i]=="X" or title[i]=="/" or title[i]=="-" or title[i]==".":
                        pass 
                    else:
                        break
                        
            title = re.sub('#.#', '#', title)
            title = re.sub('# . #', '#', title)
            title = re.sub('# .#', '#', title)
            title = re.sub('#+', '#', title)
            return title
        except Exception as e:
                logger.error(e,exc_info=True)
    
    @catch_exceptions                        
    def clean_title(self, title):
        try:
            cleaned_data = []
            title = self.remove_unit_values(title)
            return title
        except Exception as e:
            logger.error(e,exc_info=True)

    @catch_exceptions         
    def set_seed(self, seed, n_gpu):
        try:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if n_gpu > 0:
                torch.cuda.manual_seed_all(seed)
        except Exception as e:
            logger.error(e,exc_info=True)

    @catch_exceptions
    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):  
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (batch size x vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        try:
            top_k = min(top_k, logits.size(-1))  # Safety check
            if top_k > 0:
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = filter_value

            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                logits[indices_to_remove] = filter_value
            gc.collect()
            return logits
        except Exception as e:
            logger.error(e,exc_info=True)

    @catch_exceptions
    def sample_sequence(self, model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                        device='cpu'):
        try:
            context = torch.tensor(context, dtype=torch.long, device=device)
            context = context.unsqueeze(0).repeat(num_samples, 1)
            generated = context
            with torch.no_grad():
                for _ in trange(length):

                    inputs = {'input_ids': generated}

                    outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
                    next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

                    # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
                    for i in range(num_samples):
                        for _ in set(generated[i].tolist()):
                            next_token_logits[i, _] /= repetition_penalty
                        
                    filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    if temperature == 0: # greedy sampling:
                        next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                    else:
                        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    generated = torch.cat((generated, next_token), dim=1)
                    
            generated_output = copy.deepcopy(generated)
            
            del outputs, generated
            gc.collect()
            return generated_output
        except Exception as e:
            logger.error(e,exc_info=True)
    
    @catch_exceptions
    def generate_text(self, request_data):
        try:
            text_generation_result = dict()
            stop_token = "\n" #stop generating text when this token occurs
            seed = 42         #random seed for initialization, default 42, type int
            length = 310
            
            with torch.no_grad():
                self.set_seed(seed, n_gpu)
                model_class, tokenizer_class = (GPT2LMHeadModel, GPT2Tokenizer)
                self.load_model(request_data) #download model in local
                tokenizer = tokenizer_class.from_pretrained("nlg-models/"+request_data["category"])
                model = model_class.from_pretrained("nlg-models/"+request_data["category"])
                model.to(device)
                model.eval()
                
                if length < 0 and model.config.max_position_embeddings > 0:
                    length = model.config.max_position_embeddings
                elif 0 < model.config.max_position_embeddings < length:
                    length = model.config.max_position_embeddings  # No generation bigger than model size 
                elif length < 0:
                    length = MAX_LENGTH  # avoid infinite loop

                test_data = request_data["product_title"]  
                test_data = str(test_data).strip()
                context_tokens = tokenizer.encode(test_data, add_special_tokens=False)
                out = self.sample_sequence()
                out = out[:, len(context_tokens):].tolist()
                for o in out:
                    text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                    text = text[: text.find(stop_token) if stop_token else None]
                text = text.replace('"', '') #delete any quotes in the result text.
                text = str(text).strip()    #frst char of result will be mostly a space. so, stripping
                text_generation_result["description"] = text
                text_generation_result["keywords"] = self.get_keywords(request_data)
                final_result = copy.deepcopy(text_generation_result)
                
                del text_generation_result, text, out
                del context_tokens, test_data
                del model, tokenizer, model_class, tokenizer_class
                gc.collect()
                del gc.garbage[:]
                torch.cuda.empty_cache() 
            return final_result
        except Exception as e:
            logger.error(e,exc_info=True)
    
    @catch_exceptions
    def get_generated_text(self,request_data):
        try:
            response_data = {}
            mandatory_fields = ["product_title","category"] #vdezi_category_name
            for field in mandatory_fields:
                if field not in request_data["data"]:
                    response_data = {
                        "status":False,
                        "message":"Required field is missing",
                        "error_obj":{
                            "description":"{} is missing".format(field),
                            "error_code":"REQUIRED_FIELD_IS_MISSING"
                        }
                    }
            
            untrained_categories_list = ["auto_accessory", "entertainment_collectibles", "food_and_beverages", "food_service_and_jan_san", "gift_card", "lab_supplies", "lighting", "mechanical_fasteners", "musical_instruments", "power_transmission", "raw_materials", "software_video_games", "sports_memorabilia", "tires_and_wheels", "toys_baby", "video", "wine_and_alcohol", "wireless"]
            if not response_data:
                if request_data["data"]["category"] in untrained_categories_list:
                    request_data["data"]["category"] = model_mapping[request_data["data"]["category"]]
                
                request_data["data"]["product_title"] = [request_data["data"]["product_title"]] #input format to extract_attributes_api is list
                get_hierarchy_obj = json.loads(requests.post(url = Config.EXTRACT_ATTRIBUTES,json= request_data).text)
                request_data["data"]["product_title"] = request_data["data"]["product_title"][0]  #making title back to string
                
                response_data = {
                    "status":True,
                    "data":{
                        "product_details":self.generate_text(request_data["data"])
                    },
                    "message": "Successfully generated description."
                }

            gc.collect()
            del gc.garbage[:]
            torch.cuda.empty_cache()
            return response_data
        except Exception as e:
            logger.error(e,exc_info=True)

DescriptionGenerationNew = GenerateDescription()

