import pickle

BLOCK_SIZE = 1999998
#change PATH when not working localy
PATH='text_bucket13/postings_gcp/'
from inverted_index_gcp import MultiFileReader
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing


def get_inverted_index():
    with open(PATH+'index.pkl','rb') as inp: #postings_gcp_index
        invi=pickle.load(inp)
        return invi

def read_posting_list(inverted, w):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list

def get_term_total(inverted,w):
    pl=read_posting_list(inverted,w)
    terms=[x for x in pl]
    counter=0
    for doc_tup in pl:
        doc_num,term_count=doc_tup
        counter += term_count
    return counter

inv=get_inverted_index() #inverted index


