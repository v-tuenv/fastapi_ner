import os
from typing import List

import numpy as np
import onnxruntime as rt
from transformers import AutoTokenizer
from objects import WordOut, DocumentOut

L2I={'B-location': 0,
                'I-location': 5,
                'B-city': 1,
                'I-city': 6,
                'B-country': 2,
                'I-country': 7,
                'B-location_type': 3,
                'I-location_type': 8,
                'B-location_type_exclude': 4,
                'I-location_type_exclude': 9,
                'O': 10,
                'Special': -100}

i2l={0: 'B-location',
                    5: 'I-location',
                    1: 'B-city',
                    6: 'I-city',
                    2: 'B-country',
                    7: 'I-country',
                    3: 'B-location_type',
                    8: 'I-location_type',
                    4: 'B-location_type_exclude',
                    9: 'I-location_type_exclude',
                    10: 'O',
                    -100: 'Special'}

def tokenizer_query(query, tokenizer):
  o = tokenizer(query,
                  return_offsets_mapping=True,
                  return_overflowing_tokens=True)
  offset_mapping = o["offset_mapping"]
  o['input_ids'] = o['input_ids'][0]
  o['attention_mask'] = o['attention_mask'][0]
  return o

NCLASS = 5

def get_class(c):
    if c == NCLASS*2: return 'Other'
    else: return i2l[c][2:]

def pred2span(pred, example, viz=False, test=False):
    example['input_ids'] = example['input_ids'][0]
    n_tokens = len(example['input_ids'])
    classes = []
    all_span = []
    for i, c in enumerate(pred.tolist()):
        if i == n_tokens-1:
            break
        if i == 0:
            cur_span = list(example['offset_mapping'][0][i])
            classes.append(get_class(c))
        elif i > 0 and (c == pred[i-1] or (c-NCLASS) == pred[i-1]):
            cur_span[1] = example['offset_mapping'][0][i][1]
        else:
            all_span.append(cur_span)
            cur_span = list(example['offset_mapping'][0][i])
            classes.append(get_class(c))
    all_span.append(cur_span)

    text = example['query']
    predstrings = []
    for span in all_span:
        span_start = span[0]
        span_end = span[1]
        before = text[:span_start]
        token_start = len(before.split())
        if len(before) == 0: token_start = 0
        elif before[-1] != ' ': token_start -= 1
        num_tkns = len(text[span_start:span_end+1].split())
        tkns = [str(x) for x in range(token_start, token_start+num_tkns)]
        predstring = ' '.join(tkns)
        predstrings.append(predstring)

    row = {
        'query': text,
        'entity_tag': []
    }
    es = []
    for c, span, predstring in zip(classes, all_span, predstrings):
        if c!='Other':
          e = {
              'type': c,
              'predictionstring': predstring,
              'start': span[0],
              'end': span[1],
              'text': text[span[0]:span[1]]
          }
          es.append(e)
    row['entity_tag'] = es


    return row


class NerModel:
    def __init__(self, *args, **kwargs) -> None:
        
        # params
        # tokenizer
        
        # ONNXRuntime stuff
        weights_path = os.getenv("MODEL_PATH", "model/")
        self.tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_PATH", 'model/'))

        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = os.getenv("NUM_THREADS", 4)
        self.ort_session = rt.InferenceSession(os.path.join(weights_path, 'model.onnx'), sess_options)
        
    def __call__(self, query, *args, **kwargs) -> List[DocumentOut]:
        model_inputs = self.collate_fn(query)
        ort_inputs = {self.ort_session.get_inputs()[0].name: model_inputs['input_ids']}
        ort_outs = self.ort_session.run(None, ort_inputs)[0]
        value_pre_dict = self.make_output(ort_outs[0], model_inputs)
        return value_pre_dict
    
    def collate_fn(self, query):
        o = tokenizer_query(query, self.tokenizer)
        o['query'] = query
        o['input_ids'] = np.array(o['input_ids']).reshape([1, -1])

        return o

    def make_output(self, predict, sample):
        # print(predict.shape, ' predict')
        span = pred2span(predict, sample)
        tokens = []
        for tag in sorted(span['entity_tag'], key=lambda x:x['start']):
            tokens.append(
                WordOut(index=len(tokens)+1, text=tag['text'], ner_tag=tag['type'], start=tag['start'], end=tag['end'])
            )
        
        service_output = DocumentOut(
                tokens=tokens
            )
        
        return service_output
        