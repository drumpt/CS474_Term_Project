import itertools 
from progressbar import progressbar
import random

from tensorflow.keras.utils import to_categorical
import numpy as np


def data_generator(token_ids_by_doc_id, window_size, vocab_size, section_ids_by_doc_id):
    assert window_size % 2 == 0, 'window_size must be even'

    offset = window_size // 2

    doc_ids = list(token_ids_by_doc_id.keys())
  
    for doc_id in progressbar(itertools.cycle(doc_ids)):
        token_ids = token_ids_by_doc_id[doc_id]
        num_tokens = len(token_ids)
    
        if num_tokens <= window_size:
            continue
    
        target_idx = random.randint(offset, (num_tokens - offset) - 1)
    
        target_id = token_ids[target_idx]
      
        context_window = token_ids[target_idx-offset:target_idx] + token_ids[target_idx+1:target_idx+offset+1]
    
        yield (section_ids_by_doc_id[doc_id], 
	       context_window,
	       to_categorical(target_id, num_classes=vocab_size))
    

def batch(data, batch_size=32):
    while True:
        batch = itertools.islice(data, batch_size)
    
        x = []
        y = []
        x_s = []
    
        for item in batch:
            sec_id, context_window, target_ids = item
      
            x.append(context_window)
            y.append(target_ids)
            x_s.append(sec_id)
      
        yield [np.array(x_s), np.array(x)], np.array(y)

