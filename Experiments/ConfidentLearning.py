import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_ensemble_scores
from tqdm import tqdm
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


class Transformer():
    def __init__(self, df, experiment, model_name:str, n_splits:int = 10, epochs:int = 5, batch_size:int=32):
        transformer_dict = {
        'Bert':'deepset/gbert-base',
        'DistilBert':'distilbert-base-german-cased',
        'GottBert': 'uklfr/gottbert-base',
        'Electra': 'deepset/gelectra-base'}
        self.experiment = experiment
        self.df = df[df[experiment].notna()].copy()
        self.df['label'] = self.df[experiment]
        self.df['label'] = self.df['label'].astype('int')
        self.n_splits = n_splits
        self.model_name = transformer_dict[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.num_labels = 4
        self.training_args = TrainingArguments(output_dir="test_trainer", do_eval=False, per_device_train_batch_size=batch_size, num_train_epochs= epochs)
            
    def tokenize_function(self, examples):
            return self.tokenizer(examples["SENTENCE"], padding="max_length", truncation=True)
    def tokenize_function_gottbert(self, examples):
        return self.tokenizer(examples["SENTENCE"], padding="max_length", truncation=True, max_length=37)
    
    def build_dataset(self, train_frame, test_frame):
        train, test = Dataset.from_pandas(train_frame), Dataset.from_pandas(test_frame)
        if self.model_name == 'uklfr/gottbert-base':
             return train.map(self.tokenize_function_gottbert, batched=True), test.map(self.tokenize_function_gottbert, batched=True) 
        else:
            return train.map(self.tokenize_function, batched=True), test.map(self.tokenize_function, batched=True)      
        
    def train(self):
        print(f'Train {self.model_name}...')
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=1)
        for train, test in tqdm(kf.split(self.df,self.df['label'].to_numpy()), desc=f'{self.n_splits}-fold Cross-Validation', total=self.n_splits):
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
            train_frame, test_frame = self.df.iloc[train,:].copy(), self.df.iloc[test,:].copy()
            traindata, testdata = self.build_dataset(train_frame, test_frame)
            trainer = Trainer(model=model,args=self.training_args,train_dataset=traindata)
            trainer.train()
            probs = pd.DataFrame(softmax(trainer.predict(testdata).predictions, axis=1))
            probs['INDEX'] = test
            yield probs
            
    def predict_proba(self):
        self.probs = pd.concat([df for df in self.train()]).sort_values(by=['INDEX'])
        y_true = self.df['label'].values
        return y_true, self.probs.iloc[:,:-1].to_numpy()
    
    def get_metrics(self):
        y_true = self.df['label'].values
        y_pred = np.argmax(self.probs.iloc[:,:-1].to_numpy(), axis=1)
        self.df[f"{self.experiment}_true"] = y_true
        self.df[f"{self.experiment}_pred"] = y_pred
        return {
            'Model':self.model_name,
            'Accuracy' : np.round(accuracy_score(y_true, y_pred),2),
            'F1' : np.round(f1_score(y_true, y_pred, average='weighted'),2),
            'Recall' : np.round(recall_score(y_true, y_pred, average='weighted'),2),
            'Precision' : np.round(precision_score(y_true, y_pred, average='weighted'),2)
            }
        
if __name__ == '__main__':
    df = pd.read_csv('/home/sami/FLAIR/Data/Experiment_Frame.csv')
    exp = 'E1'
    score_list = []
    prob_list = []
    
    model_name = 'GottBert'
    transformer = Transformer(df=df, experiment = exp, model_name=model_name, n_splits=10, epochs=5) 
    y_true, y_probs = transformer.predict_proba()
    prob_list.append(y_probs)
    scores = transformer.get_metrics()
    score_list.append(scores)

    model_name = 'DistilBert'
    transformer = Transformer(df=df, experiment = exp, model_name=model_name, n_splits=10, epochs=5) 
    y_true, y_probs = transformer.predict_proba()
    prob_list.append(y_probs)
    scores = transformer.get_metrics()
    score_list.append(scores)

    model_name = 'Bert'
    transformer = Transformer(df=df, experiment = exp, model_name=model_name, n_splits=10, epochs=5) 
    y_true, y_probs = transformer.predict_proba()
    prob_list.append(y_probs)
    scores = transformer.get_metrics()
    score_list.append(scores)

    model_name = 'Electra'
    transformer = Transformer(df=df, experiment = exp, model_name=model_name, n_splits=10, epochs=5) 
    y_true, y_probs = transformer.predict_proba()
    prob_list.append(y_probs)
    scores = transformer.get_metrics()
    score_list.append(scores)

    probs = np.mean(np.array(prob_list), axis=0)
    scores = pd.DataFrame(score_list)
    experiment_frame = df[df[exp].notna()]
    experiment_frame['Confidence'] = find_label_issues(y_true.astype('int'), probs, filter_by='confident_learning')
    experiment_frame['Scores'] = get_label_quality_ensemble_scores(y_true.astype('int'), prob_list, method='self_confidence')
    experiment_frame.to_csv(f'/home/sami/FLAIR/Results/CONFIDENT/Confident_Result_{exp}.csv', index=False)
    scores.to_csv(f'/home/sami/FLAIR/Results/CONFIDENT/Confident_Scores_{exp}.csv',index=False)

    df = pd.read_csv('/home/sami/FLAIR/Data/Experiment_Frame.csv')
    exp = 'E3'
    score_list = []
    prob_list = []

    model_name = 'DistilBert'
    transformer = Transformer(df=df, experiment = exp, model_name=model_name, n_splits=10, epochs=5) 
    y_true, y_probs = transformer.predict_proba()
    prob_list.append(y_probs)
    scores = transformer.get_metrics()
    score_list.append(scores)

    model_name = 'Bert'
    transformer = Transformer(df=df, experiment = exp, model_name=model_name, n_splits=10, epochs=5) 
    y_true, y_probs = transformer.predict_proba()
    prob_list.append(y_probs)
    scores = transformer.get_metrics()
    score_list.append(scores)

    model_name = 'GottBert'
    transformer = Transformer(df=df, experiment = exp, model_name=model_name, n_splits=10, epochs=5) 
    y_true, y_probs = transformer.predict_proba()
    prob_list.append(y_probs)
    scores = transformer.get_metrics()
    score_list.append(scores)

    model_name = 'Electra'
    transformer = Transformer(df=df, experiment = exp, model_name=model_name, n_splits=10, epochs=5) 
    y_true, y_probs = transformer.predict_proba()
    prob_list.append(y_probs)
    scores = transformer.get_metrics()
    score_list.append(scores)

    probs = np.mean(np.array(prob_list), axis=0)
    scores = pd.DataFrame(score_list)
    experiment_frame = df[df[exp].notna()]
    experiment_frame['Confidence'] = find_label_issues(y_true.astype('int'), probs, filter_by='confident_learning')
    experiment_frame['Scores'] = get_label_quality_ensemble_scores(y_true.astype('int'), prob_list, method='self_confidence')
    experiment_frame.to_csv(f'/home/sami/FLAIR/Results/CONFIDENT/Confident_Result_{exp}.csv', index=False)
    scores.to_csv(f'/home/sami/FLAIR/Results/CONFIDENT/Confident_Scores_{exp}.csv',index=False)