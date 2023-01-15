import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import nlpaug.augmenter.word as naw

def augmentation(df):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sentences = df.SENTENCE.to_list()
    aug_en = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-de-en', to_model_name='Helsinki-NLP/opus-mt-en-de', device=device)
    aug_sp = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-de-es', to_model_name='Helsinki-NLP/opus-mt-es-de', device=device)
    en_augmented_data = aug_en.augment(sentences)
    en_df = df.copy()
    en_df.SENTENCE = en_augmented_data
    sp_augmented_data = aug_sp.augment(sentences)
    sp_df = df.copy()
    sp_df.SENTENCE = sp_augmented_data
    augmentations = pd.concat([en_df, sp_df]).drop_duplicates('SENTENCE')
    required_augmentation = 2600 if augmentations.shape[0] > 2600 else augmentations.shape[0]  # difference between 21st and 20th century
    return pd.concat([df, augmentations.sample(n=required_augmentation, random_state=2022)]).drop_duplicates('SENTENCE').reset_index(drop=True)

def doc_distributor(df, experiment):
    frame = df[df[experiment].notna()].copy()
    frame['label'] = frame[experiment]
    frame['label'] = frame['label'].astype('int')
    for doc_type in ['PROTOKOLL', 'SOCIAL_MEDIA', 'TALKSHOW', 'NEWSPAPER', 'MANIFESTO']:
        print(f"Test set is {doc_type}...")
        train = frame[frame.DOCUMENT_TYPE != doc_type]
        test = frame[frame.DOCUMENT_TYPE == doc_type]
        yield train, test, doc_type

def topic_distributor(df, experiment):
    frame = df[df[experiment].notna()].copy()
    frame['label'] = frame[experiment]
    frame['label'] = frame['label'].astype('int')
    for topic in ['Impfpflicht', 'COP26', 'Zeitenwende']:
        print(f"Test set is {topic}...")
        train = frame[frame.TOPIC != topic]
        test = frame[frame.TOPIC == topic]
        yield train, test, topic

def time_distributor(df, experiment):
    frame = df[df[experiment].notna()].copy()
    frame['label'] = frame[experiment]
    frame['label'] = frame['label'].astype('int')
    frame.DATE = frame.DATE.astype(str)
    twenty = frame[frame.DATE.str.contains('1994|1995|1996|1997|1998')].drop_duplicates('SENTENCE')
    twenty_one = frame[~frame.DATE.str.contains('1994|1995|1996|1997|1998')].drop_duplicates('SENTENCE')
    for time in ['20th Century', '21st Century']:
        print(f"Test set is {time}...")
        if time == '20th Century':
            yield twenty_one, twenty, time
        else:
            yield augmentation(twenty), twenty_one, time


class Transformer():
    def __init__(self, df, experiment, distributor, model_name:str, epochs:int = 5, batch_size:int=32):
        self.transformer_dict = {
        'Bert':'deepset/gbert-base',
        'DistilBert':'distilbert-base-german-cased',
        'GottBert': 'uklfr/gottbert-base',
        'Electra': 'deepset/gelectra-base'}
        self.df = df
        self.experiment = experiment
        self.distributor = distributor
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.result_dict = []
        
    def format_target(self, x):
        k = ['PROTOKOLL', 'SOCIAL_MEDIA', 'TALKSHOW', 'NEWSPAPER', 'MANIFESTO', 'Impfpflicht', 'COP26', 'Zeitenwende', '20th Century', '21st Century']
        v = ['Protocol', 'Twitter', 'Talkshow', 'Newspaper', 'Manifesto', 'Impfpflicht', 'COP26', 'Zeitenwende', '20th Century', '21st Century']
        dct = {key:value for key, value in zip(k,v)}
        return dct[x]

    def tokenize_function(self, examples):
            return self.tokenizer(examples["SENTENCE"], padding="max_length", truncation=True)

    def build_dataset(self, train_frame, test_frame):
        train, test = Dataset.from_pandas(train_frame), Dataset.from_pandas(test_frame)
        return train.map(self.tokenize_function, batched=True), test.map(self.tokenize_function, batched=True)  
    
    def fit(self):
        idx = 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_dict[self.model_name])
        self.num_labels = 4
        self.training_args = TrainingArguments(output_dir="test_trainer", 
                                               do_eval=False, 
                                               per_device_train_batch_size=self.batch_size, 
                                               num_train_epochs=self.epochs)
        for train_df, test_df, target_split in self.distributor(self.df, self.experiment):
            traindata, testdata = self.build_dataset(train_df, test_df)
            model = AutoModelForSequenceClassification.from_pretrained(self.transformer_dict[self.model_name], num_labels=self.num_labels)
            trainer = Trainer(model=model,args=self.training_args,train_dataset=traindata)
            trainer.train() 
            y_pred = np.argmax(trainer.predict(testdata).predictions, axis=1)
            y_true = test_df['label'].values
            self.result_dict.append(dict(
                Model = self.model_name,
                Experiment = self.experiment,
                Target_Split = target_split,
                Train_Size = train_df.shape[0],
                Test_Size = test_df.shape[0],
                Accuracy = np.round(accuracy_score(y_true, y_pred),2),
                F1 = np.round(f1_score(y_true, y_pred, average='weighted'),2),
                Recall = np.round(recall_score(y_true, y_pred, average='weighted'),2),
                Precision = np.round(precision_score(y_true, y_pred, average='weighted'),2)
            ))
            print(self.result_dict[idx])
            idx += 1
        self.results = pd.DataFrame(self.result_dict)
        self.results.Target_Split = self.results.Target_Split.apply(lambda x: self.format_target(x))
        return self.results


if __name__ == '__main__':
    df = pd.read_csv('/home/sami/FLAIR/Data/Experiment_Frame.csv')
    experiment = 'E4'
    result_list = []
    for distributor in [topic_distributor, doc_distributor, time_distributor]:
        transformer = Transformer(df = df,
                                experiment=experiment,
                                model_name='DistilBert',
                                distributor=distributor
                                )
        result_list.append(transformer.fit())
    results = pd.concat(result_list)
    results.to_csv('/home/sami/FLAIR/Results/DOMAIN_ADAPTION/DomainScores.csv', index = False)