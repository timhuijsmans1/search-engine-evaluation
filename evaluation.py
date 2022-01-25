import csv
import math
import itertools
import numpy as np
import sklearn
import scipy.stats as stats

from statistics import mean
from nltk.stem.porter import *
from gensim.models import LdaModel
from scipy.sparse import dok_matrix
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------------This section contains the code for part 1: evaluation----------------
class Evaluation:

    def __init__(self, result_path, relevant_path):
        with open(result_path, 'r') as f:
            self.results_data = f.readlines()[1:]
        with open(relevant_path, 'r') as f:
            self.relevant_data = f.readlines()[1:]

    def set_data_dictionaries(self):
        self.system_results = {}
        self.relevant_docs = {}

        # set up the system_results dictionary
        # this dictionary maps a system number to the results of each query
        for line in self.results_data:
            system_number, query_number, doc_number, rank_of_doc, score = line.split(',')
            if int(system_number) not in self.system_results:
                self.system_results[int(system_number)] = {int(query_number): {'doc_numbers': [doc_number],
                                                                    'doc_ranks': [rank_of_doc],
                                                                    'doc_scores': [float(score.strip('\n'))]}}
            else:
                if int(query_number) not in self.system_results[int(system_number)]:
                    self.system_results[int(system_number)][int(query_number)] = {'doc_numbers': [doc_number],
                                                                    'doc_ranks': [rank_of_doc],
                                                                    'doc_scores': [float(score.strip('\n'))]}
                else:
                    self.system_results[int(system_number)][int(query_number)]['doc_numbers'].append(doc_number)
                    self.system_results[int(system_number)][int(query_number)]['doc_ranks'].append(rank_of_doc)
                    self.system_results[int(system_number)][int(query_number)]['doc_scores'].append(float(score.strip('\n')))
        
        # set up the relevant_docs dictionary
        # this dictionary maps the doc_number of relevant docs 
        # from the data to their relevance
        for line in self.relevant_data:
            query_number, doc_number, relevance = line.split(',')
            if int(query_number) not in self.relevant_docs:
                self.relevant_docs[int(query_number)] = {doc_number: int(relevance)} # doc_number maps to its relevance
            else:
                self.relevant_docs[int(query_number)][doc_number] = int(relevance)
                
    def nDCG(self, k, system_number, query_number):
        # calculate iDCG@k
        idcg = 0
        # get the cutoff part of the relevant docs
        relevant_docs = list(self.relevant_docs[query_number].keys())[:k] 
        for i, doc_number in enumerate(relevant_docs):
            if i == 0:
                # maps to the relevance of a doc
                idcg += self.relevant_docs[query_number][doc_number] 
            else: 
                rank = i + 1
                relevance = self.relevant_docs[query_number][doc_number]
                idcg += relevance / math.log2(rank)
        
        # calculate DCG@k
        # get the cutoff of the system results
        doc_results = self.system_results[system_number][query_number]['doc_numbers'][:k] 
        dcg = 0
        # loop through the cutoff results of the system
        for i, doc_number in enumerate(doc_results):
            if i == 0:
                try:
                    # maps to relevance of doc number
                    dcg += self.relevant_docs[query_number][doc_number] 
                except:
                    pass
            else:
                try:
                    rank = i + 1
                    relevance = self.relevant_docs[query_number][doc_number]
                    dcg += relevance / math.log2(rank)
                except:
                    pass
        
        ndcg = dcg / idcg
        return ndcg

    def eval(self, system_number, query_number):

        # these are the dictionaries containing all the information on results and relevant docs
        results_metrics = self.system_results[system_number][query_number]
        relevant_metrics = self.relevant_docs[query_number]

        # relevant documents for the query number input
        relevant_docs = set(relevant_metrics.keys())
        
        # calculate precision@10
        cutoff_docs = set(results_metrics['doc_numbers'][:10])
        P_10 = len(cutoff_docs & relevant_docs) / len(cutoff_docs)
        
        # calculate recall@50
        cutoff_docs = set(results_metrics['doc_numbers'][:50])
        R_50 = len(cutoff_docs & relevant_docs) / len(relevant_docs)

        # calculate r-precision
        r = len(relevant_docs)
        cutoff_docs = set(results_metrics['doc_numbers'][:r])
        r_precision = len(cutoff_docs & relevant_docs) / len(cutoff_docs)

        # calculate average precision
        precisions = []
        for cutoff_number, documunt_number in enumerate(results_metrics['doc_numbers']):
            if documunt_number in relevant_docs:
                cutoff_docs = set(results_metrics['doc_numbers'][:cutoff_number + 1])
                P = len(cutoff_docs & relevant_docs) / len(cutoff_docs)
                precisions.append(P)
        try:
            average_precision = sum(precisions) / len(relevant_docs)
        except: 
            average_precision = 0
        
        # calculate nDCG@10&20
        ndcg_10 = self.nDCG(10, system_number, query_number)
        ndcg_20 = self.nDCG(20, system_number, query_number)

        return {'system_number':system_number,
                 'query_number':query_number,
                 'P@10':P_10,
                 'R@50':R_50,
                 'r-precision': r_precision,
                 'AP': average_precision,
                 'nDCG@10': ndcg_10,
                 'nDCG@20': ndcg_20}

def run_evalutaion(results, relevant_docs):
    evaluation = Evaluation(results, relevant_docs)
    evaluation.set_data_dictionaries()

    systems = list(evaluation.system_results.keys())
    queries = list(evaluation.system_results[1].keys())
    
    # this part of the function writes the eval results to a csv row by row
    # in the format of the header specified below
    with open('ir_eval.csv', 'w', newline= '') as f:
        header = ['system_number', 'query_number', 'P@10',
                 'R@50' ,'r-precision' ,'AP', 'nDCG@10','nDCG@20']
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        system_ranking = {}
        metric_scores = {}
        for system_number in systems:
            aggregate_metrics = {'system_number': system_number, 'query_number': 'mean'}
            for query_number in queries:
                # get the scores for each system/query pair
                result_dict = evaluation.eval(system_number, query_number) 
                # add the new results to the aggregate results per system
                for metric in result_dict:
                    if metric == 'system_number' or metric == 'query_number':
                        pass
                    else:
                        if metric not in aggregate_metrics:
                            aggregate_metrics[metric] = result_dict[metric]
                        else:
                            aggregate_metrics[metric] += result_dict[metric]
                        # add scores to dict of all metrics scores
                        if metric not in metric_scores:
                            metric_scores[metric] = {system_number: [result_dict[metric]]}
                        else:
                            if system_number not in metric_scores[metric]:
                                metric_scores[metric][system_number] = [result_dict[metric]]
                            else:    
                                metric_scores[metric][system_number].append(result_dict[metric])

                # write the results of one system/query pair to file
                rounded_results = {k: round(v, 3) for k,v in result_dict.items()}
                writer.writerow(rounded_results)

            # calculate the mean results from the aggregate results
            mean_dict = {}
            for key in aggregate_metrics:
                if key == 'system_number' or key == 'query_number':
                    mean_dict[key] = aggregate_metrics[key]
                else:
                    mean_dict[key] = round(aggregate_metrics[key] / len(queries), 3)
            writer.writerow(mean_dict)

            # rank systems based on mean performance per metric
            for metric in header[2:]:
                if metric not in system_ranking:
                    system_ranking[metric] = {system_number: mean_dict[metric]}
                else:
                    system_ranking[metric][system_number] = mean_dict[metric] 
    # order system ranking for each metric
    for metric in system_ranking:
        system_ranking[metric] = {
            k: v for k,v in 
            sorted(
            system_ranking[metric].items(), 
            key=lambda x: x[1], 
            reverse=True
            )
        }
            
    return metric_scores, system_ranking

def ttest(metric_scores, system_ranking):
    p_values = {}
    for metric in metric_scores:
        top_systems = list(system_ranking[metric].keys())
        system1_scores = np.array(metric_scores[metric][top_systems[0]])
        system2_scores = np.array(metric_scores[metric][top_systems[1]])
        statistic, p_value = stats.ttest_ind(system1_scores, system2_scores, equal_var=True)
        p_values[metric] = p_value
    print(system_ranking)
    print(p_values)
        


# ----------------This section contains the code for part 2: text analysis----------------

class TextAnalysis:

    def __init__(self, path):
        with open(path, 'r') as f:
            self.lines = f.readlines()
    
    def pre_process(self, string_in):
        """
        This function tokenizes, casefolds, stops and stems a string.
        The output is a list of all the words in the string
        """

        with open("englishST.txt", 'r') as f:
            STOP_WORDS = set(f.read().split('\n'))

        stemmer = PorterStemmer()

        text = string_in

        pattern = re.compile(r'[\w]+') # tokenization pattern tokenizes on word characters
        stopped = [x.lower() for x in pattern.findall(text) 
                    if x.lower() not in STOP_WORDS] # tokenize, case-fold and stop
        
        stemmed = [stemmer.stem(x) for x in stopped]

        return stemmed

    def split_by_corpus(self):
        self.corpus_documents = {}
        for line in self.lines:
            corpus, verse = line.split('\t')
            if corpus not in self.corpus_documents:
                self.corpus_documents[corpus] = [self.pre_process(verse)]
            else:
                self.corpus_documents[corpus].append(self.pre_process(verse.strip()))

    def vocabulary(self):
        self.vocabulary = set()
        for corpus in self.corpus_documents:
            for document in self.corpus_documents[corpus]:
                self.vocabulary = self.vocabulary | set(document)
    
    def document_frequencies(self):
        self.doc_frequencies = {}
        for corpus in self.corpus_documents:
            self.doc_frequencies[corpus] = dict.fromkeys(self.vocabulary, 0) # initialize 0 doc count for all terms
            documents = [set(document) for document in self.corpus_documents[corpus]] # convert each doc_list to a set
            for word in self.doc_frequencies[corpus]:
                for document in documents:
                    if word in document:
                        self.doc_frequencies[corpus][word] += 1

    def calculate_N_terms(self, target_corpus):
        other_corpora_names = list(self.corpus_documents.keys())
        other_corpora_names.remove(target_corpus)

        # get amount of docs per corpus
        target_corpus_size = len(self.corpus_documents[target_corpus])
        other_corpus1_size = len(self.corpus_documents[other_corpora_names[0]])
        other_corpus2_size = len(self.corpus_documents[other_corpora_names[1]])

        # get doc frequency dicts per corpus
        target_corpus_frequencies = self.doc_frequencies[target_corpus]
        other_corpus1_frequencies = self.doc_frequencies[other_corpora_names[0]]
        other_corpus2_frequencies = self.doc_frequencies[other_corpora_names[1]]

        N_terms = {}
        for word in target_corpus_frequencies:
            # calculate N11
            try:
                N11 = target_corpus_frequencies[word]
            except:
                N11 = 0
            
            # calculate N01
            N01 = target_corpus_size - N11

            # calculate N10 for both other corpora individually
            try:    
                N10_1 = other_corpus1_frequencies[word]
            except:
                N10_1 = 0
            try:    
                N10_2 = other_corpus2_frequencies[word]
            except:
                N10_2 = 0

            # calculate total N10 for the target corpus
            N10 = N10_1 + N10_2

            # calculate N00
            N00 = (other_corpus1_size - N10_1) + (other_corpus2_size - N10_2)

            # add the N_terms dictionary to the word to N_terms dict
            N_terms[word] = {'N11': N11, 'N01': N01, 'N10': N10, 'N00': N00}

        return N_terms
        
    def MI(self, N11, N01, N10, N00):
        N = N11 + N01 + N10 + N00
    
        try:
            first_term = (
                (N11 / N) * math.log2((N * N11) / 
                ((N11 + N10) * (N11 + N01)))
            )
        except:
            first_term = 0
        try:
            second_term = (
                (N01 / N) * math.log2((N * N01) / 
                ((N01 + N00) * (N11 + N01)))
            )
        except:
            second_term = 0
        try:
            third_term = (
                (N10 / N) * math.log2((N * N10) / 
                ((N11 + N10) * (N10 + N00)))
            )
        except:
            third_term = 0
        try:
            fourth_term = (
                (N00 / N) * math.log2((N * N00) / 
                ((N01 + N00) * (N10 + N00)))
            )
        except:
            fourth_term = 0
        
        mutual_information = first_term + second_term + third_term + fourth_term

        return mutual_information

    def chi_squared(self, N11, N01, N10, N00):
        N = N11 + N01 + N10 + N00

        chi_sq = (
            (N * (N11 * N00 - N10 * N01) ** 2) / 
            ((N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00))
        )
        return chi_sq

    def corpus_MI(self, target_corpus, top_N=10):
        corpus_N_terms = self.calculate_N_terms(target_corpus)
        mutual_informations = {}
        for word in self.vocabulary:
            mutual_information = self.MI(
                corpus_N_terms[word]['N11'], 
                corpus_N_terms[word]['N01'], 
                corpus_N_terms[word]['N10'], 
                corpus_N_terms[word]['N00'])
            mutual_informations[word] = mutual_information
        
        sorted_mutual_informations = {k: v for k, v in sorted(mutual_informations.items(), 
                                    key=lambda item: item[1], reverse=True)}
        sorted_mutual_informations = dict(itertools.islice(sorted_mutual_informations.items(), top_N))
        
        return sorted_mutual_informations

    def corpus_chi_squared(self, target_corpus, top_N=10):
        corpus_N_terms = self.calculate_N_terms(target_corpus)
        all_chi_squared = {}
        for word in corpus_N_terms:
            chi_squ = self.chi_squared(
                corpus_N_terms[word]['N11'], 
                corpus_N_terms[word]['N01'], 
                corpus_N_terms[word]['N10'], 
                corpus_N_terms[word]['N00'])
            all_chi_squared[word] = chi_squ
        
        all_chi_squared_sorted = {k: v for k, v in sorted(all_chi_squared.items(), 
                                    key=lambda item: item[1], reverse=True)}
        all_chi_squared_sorted = dict(itertools.islice(all_chi_squared_sorted.items(), top_N))                                        
        
        return all_chi_squared_sorted
    
    def LDA_model(self):
        all_documents = self.corpus_documents['OT'] + \
                        self.corpus_documents['NT'] + \
                        self.corpus_documents['Quran']

        # Create a corpus from a list of texts
        self.common_dictionary = Dictionary(all_documents)
        common_corpus = [self.common_dictionary.doc2bow(text) for text in all_documents]
        # Train the model on the corpus.
        self.lda = LdaModel(common_corpus, num_topics=20)
    
    def corpora_as_bow(self):
        self.corpus_documents_bow = {}
        for corpus in self.corpus_documents:
            common_dictionary = Dictionary(self.corpus_documents[corpus])
            self.corpus_documents_bow[corpus] = [common_dictionary.doc2bow(text) 
                                            for text in self.corpus_documents[corpus]]
    
    def ordered_topic_averages(self, document_list_bow):
        aggregate_topic_probabilities = {}
        number_of_docs = len(document_list_bow)
        for doc in document_list_bow:
            doc_scores = self.lda.get_document_topics(doc, minimum_probability=0)
            for score in doc_scores:
                if score[0] not in aggregate_topic_probabilities:
                    aggregate_topic_probabilities[score[0]] = score[1]
                else:
                    aggregate_topic_probabilities[score[0]] += score[1]
        average_topic_probabilities = {k: v/number_of_docs for k,v in 
                                        sorted(aggregate_topic_probabilities.items(), 
                                        key=lambda x: x[1], reverse=True)}

        return average_topic_probabilities

    def top_topics(self, n_topics=1):

        # find all the top topics for n-sized top
        self.top_tokens = {}
        for corpus in self.corpus_documents_bow:
            average_topics = self.ordered_topic_averages(self.corpus_documents_bow[corpus])
            for i in range(0, n_topics):
                # get the top topic string
                top_topic_id = list(average_topics.values())[i]
                top_token_id_string = self.lda.print_topic(list(average_topics.keys())[i], topn=10)
                
                # parse the top topic string
                top_token_list = [self.common_dictionary[int(token_id)] for token_id in 
                                re.findall(r'"(.*?)"', top_token_id_string)]
                top_token_score_list = re.findall(r'\d\.\d*', top_token_id_string)
                
                # convert the top topic strings to the actual topics and tokens
                if corpus not in self.top_tokens:        
                    self.top_tokens[corpus] = [(list(average_topics.keys())[i], top_topic_id, dict(zip(top_token_list, top_token_score_list)))]
                else:
                    self.top_tokens[corpus].append((list(average_topics.keys())[i], top_topic_id, dict(zip(top_token_list, top_token_score_list))))
        print(self.top_tokens)
    
def run_text_analysis(path):
    text_analysis = TextAnalysis(path)
    text_analysis.split_by_corpus()
    print('processed and split corpora')
    text_analysis.vocabulary()
    print('extracted vocabulary')
    text_analysis.document_frequencies()
    print('extracted document frequencies')
    # print('OT', text_analysis.corpus_MI('OT'))
    # print('NT', text_analysis.corpus_MI('NT'))
    # print('Quran', text_analysis.corpus_MI('Quran'))
    # print('OT', text_analysis.corpus_chi_squared('OT'))
    # print('NT', text_analysis.corpus_chi_squared('NT'))
    # print('Quran', text_analysis.corpus_chi_squared('Quran'))

    text_analysis.LDA_model()
    text_analysis.corpora_as_bow()
    text_analysis.top_topics(5) # you can input an integer here that gives the amount of top topics per corpus ranking

# ----------------This section contains the code for part 3: text classification----------------
class TextClassification:

    def __init__(self, path):
        with open(path, 'r') as f:
            self.lines = f.readlines()
    
    def pre_process(self, string_in, model_type='baseline'):
        """
        This function tokenizes, casefolds, stops and stems a string.
        The output is a list of all the words in the string
        """

        with open("englishST.txt", 'r') as f:
            STOP_WORDS = set(f.read().split('\n'))

        stemmer = PorterStemmer()

        pattern = re.compile(r'[\w]+') # tokenization pattern tokenizes on word characters
        
        if model_type == 'baseline':
            case_folded = [x.lower() for x in pattern.findall(string_in)] # tokenize, case-fold and stop
            return case_folded
        if model_type == 'MI_selection': 
            stopped = [x.lower() for x in pattern.findall(string_in) 
                        if x.lower() not in STOP_WORDS]
            stemmed = [stemmer.stem(x) for x in stopped]
            return stemmed

    def read_all_docs(self, model_type='baseline'):
        self.all_docs = []
        self.all_categories = []
        self.vocabulary = set()
        for line in self.lines:
            category, document = line.split('\t')
            pre_processed_doc = self.pre_process(document, model_type)
            self.vocabulary = self.vocabulary | set(pre_processed_doc)
            self.all_docs.append(pre_processed_doc)
            self.all_categories.append(category)
    
    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary
        
    def data_split(self, split_size):
        self.docs_train, self.docs_dev, self.categories_train, self.categories_dev = \
            train_test_split(self.all_docs, self.all_categories, test_size=split_size, random_state=3)

    def test_data(self, test_path, model_type='baseline'):
        with open(test_path, 'r') as f:
            test_lines = f.readlines()
        self.docs_test = []
        self.categories_test = []
        for line in self.lines:
            category, document = line.split('\t')
            pre_processed_doc = self.pre_process(document, model_type)
            self.docs_test.append(pre_processed_doc)
            self.categories_test.append(category)
    
    def word2id(self):
        self.word2id = {}
        for word_id, word in enumerate(self.vocabulary):
            self.word2id[word] = word_id
    
    def category2id(self):
        self.category2id = {}
        category_id = 0
        for category in self.categories_train:
            if category not in self.category2id:
                self.category2id[category] = category_id
                category_id += 1
        self.id2category = {v: k for k,v in self.category2id.items()}
        
        # used later on for the classification report
        self.category_names = [pair[0].strip('\n') for pair in sorted(self.category2id.items(), key=lambda x: x[1])]

    def feature_matrix(self, doc_list, mi_vocabulary=None, model_type='baseline'):
        rows = len(doc_list)
        columns = len(self.word2id)

        X = dok_matrix((rows, columns), dtype=np.float32)

        for doc_id, doc in enumerate(doc_list):
            # place the number at the correct tweet/word index in the sparse matrix
            for word in doc:
                if word in self.word2id: # make sure to only include those words that are in the train set vocab
                    if model_type == 'baseline':    
                        X[doc_id, self.word2id[word]] += 1
                    # truplicate the words that have the highest MI scores 
                    if model_type == 'improved':
                        check_word = self.pre_process(word, 'MI_selection')
                        # check if a word is returned and if it is in the vocabulary
                        if check_word and check_word[0] in mi_vocabulary: 
                            X[doc_id, self.word2id[word]] += 3
                        else:
                            X[doc_id, self.word2id[word]] += 1
            #print(f'{doc_id} / {len(doc_list)}')
        return X

    def train_model(self, C_param):
        X_train = self.feature_matrix(self.docs_train)
        print(X_train.shape)
        print('finished bow matrix')
        y_train = [self.category2id[category] for category in self.categories_train]
        print('starting model training')
        self.base_model = sklearn.svm.SVC(C=C_param)
        self.base_model.fit(X_train, y_train)

    def train_improved_model(self, mi_vocabulary):
        X_train = self.feature_matrix(self.docs_train, mi_vocabulary, 'improved')
        print(X_train.shape)
        print('finished bow matrix')
        y_train = [self.category2id[category] for category in self.categories_train]
        print('starting model training')
        improved_model = sklearn.svm.LinearSVC(C=1000)
        improved_model.fit(X_train, y_train)

        return improved_model

    def test_model_with_train(self, model):
        print(f'model prediction on training labels {model.predict(self.X_train)[:10]}')
        print(self.y_train[:10])

    def test_model_with_dev(self, model, mi_vocabulary=None, model_type='baseline'):
        y_dev = [self.category2id[category] for category in self.categories_dev]
        if model_type == 'improved':
            X_dev = self.feature_matrix(self.docs_dev, mi_vocabulary, 'improved')
        else:
            X_dev = self.feature_matrix(self.docs_dev)
        y_dev_predict = model.predict(X_dev)
        
        print(classification_report(y_dev, y_dev_predict, target_names=self.category_names))

        # find predictions which differ
        for i in range(len(y_dev)):
            if y_dev[i] != y_dev_predict[i]:
                print(f'expected class: {self.id2category[y_dev[i]]}')
                print(f'predicted class: {self.id2category[y_dev_predict[i]]}')
                print(f'document: {self.docs_dev[i]}')
    
    def output_row(self, model, document_data, class_labels):
        y = [self.category2id[category] for category in class_labels]
        X = self.feature_matrix(document_data)
        y_predict = model.predict(X)
        
        # get all relevant metrics
        class_report = classification_report(
                            y, 
                            y_predict, 
                            target_names=self.category_names, 
                            output_dict=True
        )
        row_dict = {
            'p-quran': round(class_report['Quran']['precision'], 3),
            'r-quran': round(class_report['Quran']['recall'], 3),
            'f-quran': round(class_report['Quran']['f1-score'], 3),
            'p-ot': round(class_report['OT']['precision'], 3),
            'r-ot': round(class_report['OT']['recall'], 3),
            'f-ot': round(class_report['OT']['f1-score'], 3),
            'p-nt': round(class_report['NT']['precision'], 3),
            'r-nt': round(class_report['NT']['recall'], 3),
            'f-nt': round(class_report['NT']['f1-score'], 3),
            'p-macro': round(class_report['macro avg']['precision'], 3),
            'r-macro': round(class_report['macro avg']['recall'], 3),
            'f-macro': round(class_report['macro avg']['f1-score'], 3)
        }
        return row_dict
    
def classification_output(train_dev_path, test_path):

    # build the output file with the correct header
    with open('classification.csv', 'w', newline= '') as f:
        header = [
        'system', 'split', 'p-quran', 'r-quran', 'f-quran', 'p-ot', 'r-ot',
        'f-ot', 'p-nt', 'r-nt', 'f-nt', 'p-macro', 'r-macro', 'f-macro'
        ]
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()

        # build base model
        text_classification_base = TextClassification(train_dev_path)
        text_classification_base.read_all_docs()
        text_classification_base.data_split(0.25)
        text_classification_base.word2id()
        text_classification_base.category2id()
        text_classification_base.test_data(test_path)
        text_classification_base.train_model(1000)

        # write base train and dev
        base_train_dict = text_classification_base.output_row(
                                    text_classification_base.base_model,
                                    text_classification_base.docs_train,
                                    text_classification_base.categories_train
                        )
        base_train_row = {**{'system':'baseline', 'split':'train'}, **base_train_dict}
        writer.writerow(base_train_row)

        base_dev_dict = text_classification_base.output_row(
                                    text_classification_base.base_model,
                                    text_classification_base.docs_dev,
                                    text_classification_base.categories_dev
                        )
        base_dev_row = {**{'system':'baseline', 'split':'dev'}, **base_dev_dict}
        writer.writerow(base_dev_row)

        # write base test
        base_test_dict = text_classification_base.output_row(
                                    text_classification_base.base_model,
                                    text_classification_base.docs_test,
                                    text_classification_base.categories_test
                        )
        base_test_row = {**{'system':'baseline', 'split':'test'}, **base_test_dict}
        writer.writerow(base_test_row)

        # build improved model
        text_classification_improved = TextClassification(train_dev_path)
        text_classification_improved.read_all_docs()
        text_classification_improved.data_split(0.05)
        text_classification_improved.word2id()
        text_classification_improved.category2id()
        text_classification_improved.test_data(test_path)
        text_classification_improved.train_model(10)

        # write improved train and dev
        improved_train_dict = text_classification_improved.output_row(
                                    text_classification_improved.base_model,
                                    text_classification_improved.docs_train,
                                    text_classification_improved.categories_train
                        )
        improved_train_row = {**{'system':'improved', 'split':'train'}, **improved_train_dict}
        writer.writerow(improved_train_row)

        improved_dev_dict = text_classification_improved.output_row(
                                    text_classification_improved.base_model,
                                    text_classification_improved.docs_dev,
                                    text_classification_improved.categories_dev
                        )
        improved_dev_row = {**{'system':'improved', 'split':'dev'}, **improved_dev_dict}
        writer.writerow(improved_dev_row)
        # write improved test
        improved_test_dict = text_classification_improved.output_row(
                                    text_classification_improved.base_model,
                                    text_classification_improved.docs_test,
                                    text_classification_improved.categories_test
                        )
        improved_test_row = {**{'system':'improved', 'split':'test'}, **improved_test_dict}
        writer.writerow(improved_test_row)

def run_text_classification(path):
    text_classification_base = TextClassification(path)

    # baseline model
    text_classification_base.read_all_docs()
    text_classification_base.data_split(0.25)
    text_classification_base.word2id()
    text_classification_base.category2id()
    base_model = text_classification_base.train_baseline_model()
    text_classification_base.test_model_with_dev(base_model)

    # MI-top model
    text_analysis_class = TextAnalysis(path)
    text_analysis_class = TextAnalysis(path)
    text_analysis_class.split_by_corpus()
    print('processed and split corpora')
    text_analysis_class.vocabulary()
    print('extracted vocabulary')
    text_analysis_class.document_frequencies()
    print('extracted document frequencies')
    ot_words = set(text_analysis_class.corpus_MI('OT', 300).keys())
    nt_words = set(text_analysis_class.corpus_MI('NT', 300).keys())
    quran_words = set(text_analysis_class.corpus_MI('Quran', 300).keys())
    top_MI_vocab = ot_words | nt_words | quran_words

    # run text classifier with new feature selection
    text_classification_improved_MI = TextClassification(path)
    text_classification_improved_MI.read_all_docs()
    text_classification_improved_MI.data_split()
    text_classification_improved_MI.word2id()
    text_classification_improved_MI.category2id()
    improved_model = text_classification_improved_MI.train_improved_model(top_MI_vocab)
    text_classification_improved_MI.test_model_with_dev(improved_model)

if __name__ == '__main__':
    # part 1: Evaluation
    RESULT_FILE = 'part1_data/system_results.csv'
    RELEVANT_FILE = 'part1_data/qrels.csv'
    # metric_scores, system_ranking = run_evalutaion(RESULT_FILE, RELEVANT_FILE)
    # ttest(metric_scores, system_ranking)


    # part 2: Text analysis
    CORPORA_PATH = 'part2_data/train_and_dev.tsv'
    TEST_PATH = 'part2_data/test.tsv.1'
    # run_text_analysis(CORPORA_PATH)

    # part 3: Text classification
    # run_text_classification(CORPORA_PATH)
    classification_output(CORPORA_PATH, TEST_PATH)
    

    