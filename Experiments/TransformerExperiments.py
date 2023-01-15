import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from codecarbon import EmissionsTracker


class Transformer():
    def __init__(self, df, model_name:str, experiment:str, n_splits:int = 10, epochs:int = 5, batch_size:int=32):
        self.transformer_dict = {
        'Bert':'deepset/gbert-base',
        'DistilBert':'distilbert-base-german-cased',
        'GottBert': 'uklfr/gottbert-base',
        'Electra': 'deepset/gelectra-base'}
        self.df = self.adapt_data(df, experiment)
        self.model_name = model_name
        self.experiment = experiment
        self.n_splits = n_splits
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_dict[self.model_name])
        self.num_labels = 4
        self.training_args = TrainingArguments(output_dir="test_trainer", do_eval=False, per_device_train_batch_size=batch_size, num_train_epochs= epochs)
        
    def adapt_data(self, df, experiment):
        df = df.copy()
        df = df[df[experiment].notna()]
        df['label'] = df[experiment].astype(int)
        df = df[['SENTENCE', 'DOCUMENT_TYPE', 'DATE','INDEX', 'label']]
        return df
            
    def tokenize_function(self, examples):
            return self.tokenizer(examples["SENTENCE"], padding="max_length", truncation=True)
    def tokenize_function_gottbert(self, examples):
        return self.tokenizer(examples["SENTENCE"], padding="max_length", truncation=True, max_length=37)

    def build_dataset(self, train_frame, test_frame):
        train, test = Dataset.from_pandas(train_frame), Dataset.from_pandas(test_frame)
        if self.model_name == 'GottBert':
             return train.map(self.tokenize_function_gottbert, batched=True), test.map(self.tokenize_function_gottbert, batched=True) 
        else:
            return train.map(self.tokenize_function, batched=True), test.map(self.tokenize_function, batched=True)     
        
    def train(self):
        print(f'Train {self.model_name}...')
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=1)
        for train, test in tqdm(kf.split(self.df,self.df['label'].to_numpy()), desc=f'{self.n_splits}-fold Cross-Validation', total=self.n_splits):
            torch.cuda.empty_cache()
            model = AutoModelForSequenceClassification.from_pretrained(self.transformer_dict[self.model_name], num_labels=self.num_labels)
            train_frame, test_frame = self.df.iloc[train,:].copy(), self.df.iloc[test,:].copy()
            traindata, testdata = self.build_dataset(train_frame, test_frame)
            trainer = Trainer(model=model,args=self.training_args,train_dataset=traindata)
            trainer.train()
            probs = pd.DataFrame(softmax(trainer.predict(testdata).predictions, axis=1))
            probs['INDEX'] = test
            yield probs
            
    def fit(self):
        self.probs = pd.concat([df for df in self.train()]).sort_values(by=['INDEX'])
        y_pred = np.argmax(self.probs.iloc[:,:-1].to_numpy(), axis=1)
        y_true = self.df['label'].values
        sentence = self.df['SENTENCE'].values
        doc = self.df['DOCUMENT_TYPE'].values
        date = self.df['DATE'].values
        index = self.df['INDEX'].values
        results = pd.DataFrame(dict(
            SENTENCE = sentence,
            DATE = date,
            DOCUMENT_TYPE = doc,
            INDEX = index,
            GROUNDTRUTH = y_true,
            PREDICTION = y_pred
            ))
        scores = {
            'Model':self.model_name,
            'Experiment' : self.experiment,
            'Accuracy' : np.round(accuracy_score(y_true, y_pred),2),
            'F1' : np.round(f1_score(y_true, y_pred, average='weighted'),2),
            'Recall' : np.round(recall_score(y_true, y_pred, average='weighted'),2),
            'Precision' : np.round(precision_score(y_true, y_pred, average='weighted'),2)
            }
        scores = pd.DataFrame([scores])
        print(scores)
        return results, scores
    
def transformer_experiment(df, model_name, experiment):
    path = f"/home/sami/FLAIR/Results/TRANSFORMER/{experiment}/"
    tracker = EmissionsTracker(project_name=f"{experiment}_{model_name}", output_dir='/home/sami/FLAIR/Data/Emissions/TRANSFORMER')
    tracker.start()
    transformer = Transformer(df=df, model_name=model_name, experiment=experiment)
    results, scores = transformer.fit()
    tracker.stop()
    results.to_csv(f"{path}{model_name}_results.csv", index = False)
    scores.to_csv(f"{path}{model_name}_scores.csv", index = False)
    
if __name__ == '__main__':
    for model_name in ['DistilBert', 'GottBert', 'Bert', 'Electra']:
        for experiment in ['E0', 'E1', 'E2', 'E3', 'E4']:
            df = pd.read_csv('/home/sami/FLAIR/Data/Experiment_Frame.csv')
            transformer_experiment(df, model_name, experiment)
  