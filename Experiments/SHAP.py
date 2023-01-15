from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import scipy as sp
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset,  ClassLabel, Value
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
    
class Transformer():
    def __init__(self, df, model_name:str, n_splits:int = 10, epochs:int = 5, batch_size:int=32):
        self.df = df
        self.n_splits = n_splits
        self.model_name = model_name
        self.df = self.df[['SENTENCE', 'label']]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.num_labels = 4
        self.training_args = TrainingArguments(output_dir="test_trainer", do_eval=False, per_device_train_batch_size=batch_size, num_train_epochs= epochs)
        self.list_shap_values = list()
        
    def tokenize_function(self, examples):
            return self.tokenizer(examples["SENTENCE"], padding="max_length", truncation=True)

    def build_dataset(self, train_frame, test_frame):
        train, test = Dataset.from_pandas(train_frame), Dataset.from_pandas(test_frame)
        return train.map(self.tokenize_function, batched=True), test.map(self.tokenize_function, batched=True)      
        
    def train(self):
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=1)
        for train, test in tqdm(kf.split(self.df,self.df['label'].to_numpy()), desc=f'{self.n_splits}-fold Cross-Validation', total=self.n_splits):
            torch.cuda.empty_cache()
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
            train_frame, test_frame = self.df.iloc[train,:].copy(), self.df.iloc[test,:].copy()
            traindata, testdata = self.build_dataset(train_frame, test_frame)
            trainer = Trainer(model=model,args=self.training_args,train_dataset=traindata)
            trainer.train()
            trainer.save_model('/home/sami/FLAIR/Results/SHAP/')
            model = AutoModelForSequenceClassification.from_pretrained('/home/sami/FLAIR/Results/SHAP/', num_labels=self.num_labels).cuda()
            yield model, test_frame
    
    def get_shap_values(self):
        for model, test_frame in self.train():
            test_frame = test_frame.rename(columns={'SENTENCE':'text'})
            test_data = Dataset.from_pandas(test_frame)
            test_data = test_data.remove_columns(['__index_level_0__'])
            new_features = test_data.features.copy()
            new_features["label"] = ClassLabel(names=["Aussage", "Meinung", "Prognose", "Sonstiges"])
            new_features["text"] = Value("string")
            test_data = test_data.cast(new_features)
            tokenizer = self.tokenizer
            pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)
            explainer = shap.Explainer(pred, seed=2022, output_names=["Aussage", "Meinung", "Prognose", "Sonstiges"])
            shap_values = explainer(test_frame['text'])
            self.list_shap_values.append(shap_values)
                
    def return_shap_values(self):
        dct = dict()
        dct_list = list()
        for shap_value in self.list_shap_values:
            for words, values in zip(shap_value, shap_value):
                for word, value in zip(words, values):
                    if word.data.strip() in dct:
                        dct[word.data.strip()] = np.concatenate((dct[word.data.strip()] , value.values), axis=0)
                    else:
                        dct[word.data.strip()] = value.values
        for key, value in dct.items():
            dct[key] = np.mean(dct[key].reshape(-1, 4), axis=0)
            dct_list.append(dict(
                Word = key,
                Aussage = dct[key][0],
                Meinung = dct[key][1],
                Prognose = dct[key][2],
                Sonstiges = dct[key][3],
            ))
        self.shap_values = pd.DataFrame(dct_list)
        return self.shap_values
    

if __name__ == '__main__':
    transformer_dict = {'DistilBert':'distilbert-base-german-cased'}
    df = pd.read_csv('/home/sami/FLAIR/Data/Experiment_Frame.csv')
    df = df.sort_values('INDEX')

    experiment = 'E4'
    data = df[df[experiment].notna()]
    data['label'] = data[experiment].astype(int)
        
    transformer = Transformer(df=data, model_name=transformer_dict['DistilBert'])
    transformer.get_shap_values()
    shap_values = transformer.return_shap_values()
    shap_values.to_csv(f'/home/sami/FLAIR/Results/SHAP/{experiment}_ShapValues.csv', index = False)
