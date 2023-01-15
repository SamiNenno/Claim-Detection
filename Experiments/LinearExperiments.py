import pandas as pd
import nltk
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from codecarbon import EmissionsTracker

class LinearExperiments():
    def __init__(self) -> None:
        self.results = []
        self.model_names = ['NaiveBayes',
                            'LogReg',
                            'SVM',
                            'XGBoost',
                            'RandomForest',
                            'AdaBoost'
                            ]
        
    def experiment(self):
        embedding_dict = {
            'GloVe': self.get_Xy_Glove,
            'Tf-idf' : self.get_Xy_TFIDF,
            'SentEmbeddings': self.get_Xy_sentence_embeddings,
            'BoW' : self.get_Xy_BoW,
            'Fasttext_Wiki': self.get_Xy_Fasttext_WIKI,
            'Word2Vec':self.get_Xy_Word2Vec
        }
        for experiment in tqdm(['E0', 'E1', 'E2', 'E3', 'E4'], desc='Iterate over experiments', total=5, position=0):
            for emb_name, emb_func in tqdm(embedding_dict.items(), desc='Iterate over embeddings', total=len(embedding_dict), position=1):
                for model_name in tqdm(self.model_names, desc='Train different models', leave=True, position=2):
                    df = pd.read_csv('/home/sami/FLAIR/Data/Sentence_Embeddings.csv')
                    df = df[df[experiment].notna()]
                    tracker = EmissionsTracker(project_name=f"{experiment}_{model_name}_{emb_name}", output_dir='/home/sami/FLAIR/Data/Emissions/LINEAR/')
                    tracker.start()
                    clf = self.get_model(model_name, emb_name)
                    X, y = emb_func(df, experiment)
                    scores = self.predict(X, y, clf)
                    tracker.stop()
                    print(experiment, model_name, emb_name)
                    print(scores)
                    self.results.append(dict(
                        Model = model_name,
                        Embedding = emb_name,
                        Experiment = experiment,
                        Accuracy = scores['Accuracy'],
                        F1 = scores['F1'],
                        Recall = scores['Recall'],
                        Precision = scores['Precision'],
                    ))
        self.result_frame = pd.DataFrame(self.results)
        return self.result_frame
                    
    def get_model(self, model_name, emb_name):
        if model_name == 'LogReg':
            return LogisticRegression(random_state=0, max_iter=700)
        elif model_name == 'SVM':
            return SVC()
        elif model_name == 'NaiveBayes':
            if emb_name in ['BoW', 'Tf-idf']:
                return MultinomialNB()
            else:
                return GaussianNB()
        elif model_name == 'XGBoost':
            return GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        elif model_name == 'RandomForest':
            return RandomForestClassifier(max_depth=2, random_state=0)
        elif model_name == 'AdaBoost':
            return AdaBoostClassifier(n_estimators=100, random_state=0)
        else:
            return KeyError(f'This model does not exist: {model_name}')
        
    
    def predict(self, X, y, clf):
        cv = StratifiedKFold(n_splits=10)
        y_pred = cross_val_predict(estimator=clf, X=X, y=y, cv=cv, n_jobs=-1)
        scores = {
            'Accuracy' : np.round(accuracy_score(y, y_pred),2),
            'F1' : np.round(f1_score(y, y_pred, average='weighted'),2),
            'Recall' : np.round(recall_score(y, y_pred, average='weighted'),2),
            'Precision' : np.round(precision_score(y, y_pred, average='weighted'),2)
            }
        return scores
    
    def get_Xy_Glove(self, df, label_column:str):
        path = '/home/sami/ActConfLearning/FINAL_EVAL/CUSTOM_GLOVE.csv'
        return self.get_Xy_dense(df, path, label_column)
    def get_Xy_Word2Vec(self, df, label_column:str):
        path = '/home/sami/ActConfLearning/FINAL_EVAL/CUSTOM_WORD2VEC.csv'
        return self.get_Xy_dense(df, path, label_column)
    def get_Xy_Fasttext_WIKI(self, df, label_column:str):
        path = '/home/sami/ActConfLearning/FINAL_EVAL/CUSTOM_FASTTEXT_WIKI.csv'
        return self.get_Xy_dense(df, path, label_column)
        
    def get_Xy_dense(self, df, path, label_column:str):
        glove = pd.read_csv(path)
        def get_punctuation():
            dct = {'VOCAB':[]}
            temp = {f"DIMENSION_{idx}":[] for idx in range(300)}
            dct = dct|temp
            for punctuation, seed in zip(['.', '!', '?', ':', ';', ','], [2, 4, 6, 8, 10, 12]):
                dct['VOCAB'].append(punctuation)
                np.random.seed(seed)
                embedding = np.random.uniform(-1,1,300)
                for idx in range(300):
                    dct[f"DIMENSION_{idx}"].append(embedding[idx])
            return pd.DataFrame(dct)

        def glove_sentence_embeddings(df, glove):
            glove = pd.concat([glove,get_punctuation()])
            X = []
            for idx, row in tqdm(df.iterrows(), total = df.shape[0]):
                sentence = row.SENTENCE.lower()
                tokens = nltk.word_tokenize(sentence, language='german')
                sentence = np.nanmean(glove[glove.VOCAB.isin(tokens)].iloc[:,1:].to_numpy(), axis=0)
                X.append(sentence)
            return np.array(X)
        X = glove_sentence_embeddings(df, glove)
        df = df.reset_index(drop=True)
        df = df.drop(np.unique(np.argwhere(X!=X)[:,0]))
        X = X[~np.isnan(X)].reshape(-1,300)
        y = df[label_column].to_numpy()
        return X, y

    def get_Xy_sentence_embeddings(self, df, label_column:str):
        X = df.iloc[:,-1024:].to_numpy()
        y = df[label_column].to_numpy()
        return X, y

    def get_Xy_BoW(self, df,label_column:str):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df.SENTENCE.to_list())
        y = df[label_column].to_numpy()
        return X, y
    
    def get_Xy_TFIDF(self, df,label_column:str):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df.SENTENCE.to_list())
        y = df[label_column].to_numpy()
        return X, y
        
if __name__ == '__main__':
    experiments = LinearExperiments()
    results = experiments.experiment()
    results.to_csv('/home/sami/FLAIR/Results/LINEAR/LinearScores.csv', index = False)