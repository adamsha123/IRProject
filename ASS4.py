import threading
from collections import defaultdict,Counter
import re
import nltk
import numpy as np
nltk.download('stopwords')

from nltk.corpus import stopwords
from tqdm import tqdm
import operator
from itertools import count
from contextlib import closing
from pathlib import Path
from operator import itemgetter
import pickle

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))


def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    return list_of_tokens


def bm25_preprocess(data):
    """
    This function goes through the data and saves relevant information for the calculation of bm25.
    Specifically, in this function, we will create 3 objects that gather information regarding document length, term frequency and
    document frequency.
    Parameters
    -----------
    data: list of lists. Each inner list is a list of tokens.
    Example of data:
    [
        ['sky', 'blue', 'see', 'blue', 'sun'],
        ['sun', 'bright', 'yellow'],
        ['comes', 'blue', 'sun'],
        ['lucy', 'sky', 'diamonds', 'see', 'sun', 'sky'],
        ['sun', 'sun', 'blue', 'sun'],
        ['lucy', 'likes', 'blue', 'bright', 'diamonds']
    ]

    Returns:
    -----------
    three objects as follows:
                a) doc_len: list of integer. Each element represents the length of a document.
                b) tf: list of dictionaries. Each dictionary corresponds to a document as follows:
                                                                    key: term
                                                                    value: normalized term frequency (by the length of document)


                c) df: dictionary representing the document frequency as follows:
                                                                    key: term
                                                                    value: document frequency
    """
    doc_len = []
    tf = []
    df = {}

    # YOUR CODE HERE
    doc_len = [len(x) for x in data]
    for lst in data:
        temp = {}
        for word in lst:
            if word not in temp.keys():
                temp[word] = 1 / len(data[data.index(lst)])
            else:
                temp[word] += 1 / len(data[data.index(lst)])
        tf.append(temp)
    for i in range(len(data)):
        for word in data[i]:
            if word not in df.keys():
                df[word] = [i]
            elif i not in df[word]:
                df[word].append(i)
    temp2 = {}
    for tup in df.items():
        temp2[tup[0]] = len(tup[1])
    df = temp2
    # raise NotImplementedError()
    return doc_len, tf, df


import math


class BM25:
    """
    Best Match 25.

    Parameters to tune
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.
        The frequnecy is normilzied by the max term frequency for each document.

    doc_len_ : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.
    """

    def __init__(self, doc_len, df, tf=None, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.tf_ = tf
        self.doc_len_ = doc_len
        self.df_ = df
        self.N_ = len(doc_len)
        self.avgdl_ = sum(doc_len) / len(doc_len)

    def calc_idf(self, query):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        # YOUR CODE HERE
        scores = {}
        for term in query:
            nq = self.df_.get(term)
            if nq is None:
                nq = 0
            scores[term] = np.log((self.N_ - nq + 0.5) / (nq + 0.5) + 1)
        return scores
        # raise NotImplementedError()



def top_N_documents(df, N):
    """
    This function sort and filter the top N docuemnts (by score) for each query.

    Parameters
    ----------
    df: DataFrame (queries as rows, documents as columns)
    N: Integer (how many document to retrieve for each query)

    Returns:
    ----------
    top_N: dictionary is the following stracture:
          key - query id.
          value - sorted (according to score) list of pairs lengh of N. Eac pair within the list provide the following information (doc id, score)
    """
    # YOUR CODE HERE
    top_N = {}
    for i in range(len(df)):
        for j in range(len(df[0])):
            if i not in top_N:
                top_N[i] = [(j + 1, df[i, j])]
            else:
                top_N[i].append((j + 1, df[i, j]))
    sorted_top_N = {}
    for tup in top_N.items():
        sorted_top_N[tup[0]] = sorted(tup[1], key=lambda x: x[1], reverse=True)
    after_filter = {}
    for tup in sorted_top_N.items():
        after_filter[tup[0]] = tup[1][:N]
    return after_filter

    # raise NotImplementedError()


# Let's start with a small block size of 30 bytes just to test things out.
BLOCK_SIZE = 199998


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

DL = {}  # We're going to update and calculate this after each document. This will be usefull for the calculation of AVGDL (utilized in BM25)


class InvertedIndex:
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents
        # the number of bytes from the beginning of the file where the posting list
        # starts.
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage
            side-effects).
        """
        DL[(doc_id)] = DL.get(doc_id, 0) + (len(tokens))
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        max_value = max(w2cnt.items(), key=operator.itemgetter(1))[1]
        # frequencies = {key: value/max_value for key, value in frequencies.items()}
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write(self, base_dir, name):
        """ Write the in-memory index to disk and populate the `posting_locs`
            variables with information about file location and offset of posting
            lists. Results in at least two files:
            (1) posting files `name`XXX.bin containing the posting lists.
            (2) `name`.pkl containing the global term stats (e.g. df).
        """
        #### POSTINGS ####
        self.posting_locs = defaultdict(list)
        with closing(MultiFileWriter(base_dir, name)) as writer:
            # iterate over posting lists in lexicographic order
            for w in sorted(self._posting_list.keys()):
                self._write_a_posting_list(w, writer, sort=True)
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def _write_a_posting_list(self, w, writer, sort=False):
        # sort the posting list by doc_id
        pl = self._posting_list[w]
        if sort:
            pl = sorted(pl, key=itemgetter(0))
        # convert to bytes
        b = b''.join([(int(doc_id) << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                      for doc_id, tf in pl])
        # write to file(s)
        locs = writer.write(b)
        # save file locations to index
        self.posting_locs[w].extend(locs)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary.
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                # read a certain number of bytes into variable b
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                # convert the bytes read into `b` to a proper posting list.

                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))

                yield w, posting_list


def get_posting_gen(index):
    """
    This function returning the generator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter())
    return words, pls


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.df)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.df.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
            df = index.df[token]
            idf = math.log((len(index.df)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    N = len(index.df)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq) * math.log(N / index.df[term], 10)) for doc_id, freq in
                               list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates

def get_term_candidates_for_thread(term,index,words,pls,candidates,N):
    if term in words:
        list_of_doc = pls[words.index(term)]
        normlized_tfidf = [(doc_id, (freq) * math.log(N / index.df[term], 10)) for doc_id, freq in
                           list_of_doc]

        for doc_id, tfidf in normlized_tfidf:
            candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf


def get_candidate_documents_and_scores_try(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    N = len(index.df)
    Threads=[]
    mutex= threading.Lock()
    for term in np.unique(query_to_search):
        thread=threading.Thread(target=get_term_candidates_for_thread,args=(term,index,words,pls,candidates,N,))
        Threads.append(thread)
        thread.start()
    for t in Threads:
        t.join()
    return candidates

def cosine_similarity(D, Q, query):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    tf_idf_dict = {}
    norm_Q = sum([x ** 2 for x in Q])
    for index, row in D.iterrows():
        numerator = 0
        denominator = 0
        norm_d = sum([x ** 2 for x in row])
        numerator += row[index] * Q[index]
        denominator = (norm_d * norm_Q) ** 0.5
        val = numerator / denominator
        tf_idf_dict[index] = val
    return tf_idf_dict


# raise NotImplementedError()

def try_func_sim_cos(D,query):
    q_dict={}
    for word in query:
        if word not in q_dict:
            q_dict[word]=1
        else:
            q_dict[word]+=1
    norm_Q = sum([x ** 2 for x in q_dict.values()])
    tf_idf_dict={}
    for index,row in D.iterrows():
        score=0
        norm_d=0
        for word in query:
            score+=row[word]*q_dict.get(word)
            norm_d+=row[word]*row[word]
        denominator = (norm_d * norm_Q) ** 0.5
        tf_idf_dict[index]=score#/denominator
    return tf_idf_dict

def try_get_top_n(sim_dict,N=3):
    items=list(sim_dict.items())
    items.sort(key=lambda x:x[0],reverse=True)
    doc_ids=[x[0] for x in items]
    return doc_ids[:N]



def get_candidate_documents(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    list of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = []
    for term in np.unique(query_to_search):
        if term in words:
            current_list = (pls[words.index(term)])
            candidates += current_list
    return np.unique(candidates)


import math

# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.words, self.pls = zip(*self.index.posting_lists_iter())

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, queries, N=3):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        ret = {}
        for q in queries.keys():
            cands = get_candidate_documents(queries.get(q), self.index, self.words, self.pls)
            self.idf = self.calc_idf(queries.get(q))
            for doc_num in cands:
                score = self._score(queries.get(q), doc_num)
                tup = (doc_num, score)
                if ret.get(q) is None:
                    ret[q] = [tup]
                else:
                    ret[q].append(tup)
            ret[q] = sorted(ret[q], key=lambda x: x[1], reverse=True)[0:N]
        return ret

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = DL[str(doc_id)]

        for term in query:
            if term in self.index.term_total.keys():
                term_frequencies = dict(self.pls[self.words.index(term)])
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score


def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    # YOUR CODE HERE
    ret = {}
    t_dict = {}
    b_dict = {}
    updated = {}
    # dict of title (query_id,doc_id):score
    for q in title_scores:
        for tup in title_scores[q]:
            td = (q, tup[0])
            t_dict[td] = tup[1]

    # dict of body (query_id,doc_id):score
    for q in body_scores:
        for tup in body_scores[q]:
            td = (q, tup[0])
            b_dict[td] = tup[1]

    # get all the keys in both dicts
    key_set = set(list(t_dict.keys()) + list(b_dict.keys()))
    for item in sorted(key_set, key=lambda x: (x[0], x[1])):
        b_val = b_dict.get(item)
        if b_val is None:  # not exist in the b_dict
            b_val = 0
        t_val = t_dict.get(item)  # not exist in the t_dict
        if t_val is None:
            t_val = 0
        ret[item] = t_val * title_weight + b_val * text_weight
    new_dict = {}
    for tup in ret.items():
        k = (tup[0][1], tup[1])
        if tup[0][0] not in new_dict:
            new_dict[tup[0][0]] = [k]
        else:
            new_dict[tup[0][0]].append(k)
    ret2 = {}
    for tup in new_dict.items():
        ret2[tup[0]] = sorted(tup[1], key=lambda x: x[1], reverse=True)[0:N]
    return ret2

def intersection(l1,l2):
    """
    This function perform an intersection between two lists.

    Parameters
    ----------
    l1: list of documents. Each element is a doc_id.
    l2: list of documents. Each element is a doc_id.

    Returns:
    ----------
    list with the intersection (without duplicates) of l1 and l2
    """
    return list(set(l1)&set(l2))


def recall_at_k(true_list, predicted_list, k=40):
    """
    This function calculate the recall@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, recall@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    return round(len(intersection(true_list, predicted_list[0:k])) / len(true_list), 3)


def precision_at_k(true_list, predicted_list, k=40):
    """
    This function calculate the precision@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    k_list = predicted_list[0:k]
    intersec = intersection(k_list, true_list)
    return round(len(intersec) / len(k_list), 3)

    # raise NotImplementedError()


def r_precision(true_list, predicted_list):
    """
    This function calculate the r-precision metric. No `k` parameter is used.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score

    Returns:
    -----------
    float, r-precision with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    r = len(true_list)
    r_list = predicted_list[0:r]
    intersec = intersection(r_list, true_list)
    return round(len(intersec) / len(r_list), 3)


def reciprocal_rank_at_k(true_list, predicted_list, k=40):
    """
    This function calculate the reciprocal_rank@k metric.
    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, reciprocal rank@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    k_list = predicted_list[0:k]
    for i in range(len(k_list)):
        if k_list[i] in true_list:
            return round(1 / (i + 1), 3)
    return 0


def fallout_rate(true_list, predicted_list, k=40):
    """
    This function calculate the fallout_rate@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, fallout_rate@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    k_list = predicted_list[0:k]
    intersec = intersection(k_list, true_list)
    not_relevant_k = len(k_list) - len(intersec)
    not_relevant_coll = len(list(DL.keys())) - len(true_list)
    if not_relevant_coll == 0:
        return 1
    else:
        return round(not_relevant_k / not_relevant_coll, 3)



def f_score(true_list, predicted_list, k=40):
    """
    This function calculate the f_score@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, f-score@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    R = recall_at_k(true_list, predicted_list, k)
    P = precision_at_k(true_list, predicted_list, k)
    if P + R == 0:
        return 0
    else:
        F = 2 * P * R / (P + R)
        return round(F, 3)


def average_precision(true_list, predicted_list, k=40):
    """
    This function calculate the average_precision@k metric.(i.e., precision in every recall point).

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, average precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    lst = []
    rel_counter = 0
    k_list = predicted_list[0:k]
    for i in range(k):
        if k_list[i] in true_list:
            rel_counter += 1
            lst.append(rel_counter / (i + 1))
    if (len(lst) == 0):
        return 0
    return round(sum(lst) / len(lst), 3)


def ndcg_at_k(true_tuple_list, predicted_list, k=40):
    """
    This function calculate the ndcg@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, ndcg@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    true_dict = dict(true_tuple_list)
    k_list = predicted_list[0:k]
    true_list = [x[0] for x in true_tuple_list]
    intersec = intersection(k_list, true_list)
    first_intersec_doc = None
    if len(intersec) > 0:
        first_intersec_doc = intersec[0]
        rel1 = true_dict.get(first_intersec_doc)
    else:
        return 0
    dcg = sum([true_dict.get(intersec[i - 1]) / math.log2(i + 1) for i in range(2, len(intersec) + 1)])
    norm = sum([true_dict.get(intersec[i - 1]) / math.log2(i + 1) for i in range(1, len(intersec) + 1)])
    return (rel1 + dcg) / norm


def evaluate(true_relevancy, predicted_relevancy, k, print_scores=True):
    """
    This function calculates multiple metrics and returns a dictionary with metrics scores across different queries.
    Parameters
    -----------
    true_relevancy: list of tuples indicating the relevancy score for a query. Each element corresponds to a query.
    Example of a single element in the list:
                                            (3, {'question': ' what problems of heat conduction in composite slabs have been solved so far . ',
                                            'relevance_assessments': [(5, 3), (6, 3), (90, 3), (91, 3), (119, 3), (144, 3), (181, 3), (399, 3), (485, 1)]})

    predicted_relevancy: a dictionary of the list. Each key represents the query_id. The value of the dictionary is a sorted list of relevant documents and their scores.
                         The list is sorted by the score.
    Example:
            key: 1
            value: [(13, 17.256625), (486, 13.539465), (12, 9.957595), (746, 9.599499999999999), (51, 9.171265), .....]

    k: integer, a number to slice the length of the predicted_list

    print_scores: boolean, enable/disable a print of the mean value of each metric.

    Returns:
    -----------
    a dictionary of metrics scores as follows:
                                                        key: metric name
                                                        value: list of metric scores. Each element corresponds to a given query.
    """

    recall_lst = []
    precision_lst = []
    f_score_lst = []
    r_precision_lst = []
    reciprocal_rank_lst = []
    avg_precision_lst = []
    fallout_rate_lst = []
    ndcg_lst = []
    metrices = {'recall@k': recall_lst,
                'precision@k': precision_lst,
                'f_score@k': f_score_lst,
                'r-precision': r_precision_lst,
                'MRR@k': reciprocal_rank_lst,
                'MAP@k': avg_precision_lst,
                'fallout@k': fallout_rate_lst,
                'ndcg@k': ndcg_lst}

    for query_id, query_info in tqdm(true_relevancy):
        predicted = [doc_id for doc_id, score in predicted_relevancy[query_id]]
        ground_true = [int(doc_id) for doc_id, score in query_info['relevance_assessments']]

        recall_lst.append(recall_at_k(ground_true, predicted, k=k))
        precision_lst.append(precision_at_k(ground_true, predicted, k=k))
        f_score_lst.append(f_score(ground_true, predicted, k=k))
        r_precision_lst.append(r_precision(ground_true, predicted))
        reciprocal_rank_lst.append(reciprocal_rank_at_k(ground_true, predicted, k=k))
        avg_precision_lst.append(average_precision(ground_true, predicted, k=k))
        fallout_rate_lst.append(fallout_rate(ground_true, predicted, k=k))
        ndcg_lst.append(ndcg_at_k(query_info['relevance_assessments'], predicted, k=k))

    return metrices


def grid_search_models(data, true_relevancy, bm25_param_list, w_list, N, idx_title, idx_body):
    """
    This function is performing a grid search upon different combination of parameters.
    The parameters can be BM25 parameters (i.e., bm25_param_list) or different weights (i.e., w_list).

    Parameters
    ----------
    data: dictionary as follows:
                            key: query_id
                            value: list of tokens

    true_relevancy: list of tuples indicating the relevancy score for a query. Each element corresponds to a query.
    Example of a single element in the list:
                                            (3, {'question': ' what problems of heat conduction in composite slabs have been solved so far . ',
                                            'relevance_assessments': [(5, 3), (6, 3), (90, 3), (91, 3), (119, 3), (144, 3), (181, 3), (399, 3), (485, 1)]})

    bm25_param_list: list of tuples. Each tuple represent (k,b1) values.

    w_list: list of tuples. Each tuple represent (title_weight,body_weight).
    N: Integer. How many document to retrieve.

    idx_title: index build upon titles
    idx_body:  index build upon bodies
    ----------
    return: dictionary as follows:
                            key: tuple indiciating the parameters examined in the model (k1,b,title_weight,body_weight)
                            value: MAP@N score
    """
    models = {}
    # YOUR CODE HERE

    for i in range(len(bm25_param_list)):
        k = bm25_param_list[i][0]
        b1 = bm25_param_list[i][1]
        bm_25_body = BM25_from_index(idx_body, k, b1)
        bm_25_title = BM25_from_index(idx_title, k, b1)
        score_title = bm_25_title.search(data, N)
        score_body = bm_25_body.search(data, N)

        for j in range(len(w_list)):
            title_weight = w_list[i][0]
            body_weight = w_list[i][1]
            tup = (k, b1, title_weight, body_weight)
            merge = merge_results(score_title, score_body, title_weight, body_weight, N)
            counter = 0
            for k in range(N):
                predicted = [doc_id for doc_id, score in merge[true_relevancy[k][0]]]
                ground_true = [int(doc_id) for doc_id, score in true_relevancy[k][1]['relevance_assessments']]
                counter += average_precision(ground_true, predicted, k)
                models[tup] = counter / N
    return models

