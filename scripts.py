
import os
import argparse
import pickle
import tokenizers
import qarac.corpora.BNCorpus
import qarac.corpora.Batcher
import qarac.models.qarac_base_model
import keras
import tensorflow
import spacy
import spacy_experimental
import pandas

def decoder_loss(y_true,y_pred):
    return keras.losses.sparse_categorical_crossentropy(y_true,
                                                        y_pred.logits,
                                                        logits=True)

def capitalise(token,i):
    return token.text_with_ws.title() if i==0 or token.tag_.startswith('NNP') else token.text_with_ws.lower()

def clean_question(doc):
    words = [capitalise(token,i) for (i,token) in enumerate(doc)]
    if words[-1]!='?':
        words.append('?')
    return ''.join(words)

def prepare_wiki_qa(filename,outfilename):
    data = pandas.read_csv(filename,sep='\t')
    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe('experimental_coref')
    data['Resolved_answer'] = pandas.Series([sent.text 
                                              for doc in nlp.pipe(data.groupby('DocumentID')['Sentence'].apply(lambda x: ' '.join(x)))
                                              for sent in doc.sentences])
    data['Cleaned_questions']=pandas.Series([clean_question(doc) for doc in nlp.pipe(data)])
    data[['Cleaned_questions','Resolved_answers','Label']].to_csv(outfilename)

        
def train_base_model(task,filename):
    tokenizer = tokenizers.Tokenizer.from_pretrained('xlm-roberta-base')
    tokenizer.add_special_tokens(['<start>','<end>','<pad>'])
    tokenizer.save('/'.join([os.environ['HOME'],
                            'QARAC',
                            'models',
                            'tokenizer.json']))
    bnc = qarac.corpora.BNCorpus.BNCorpus(tokenizer=tokenizer,
                                          task=task)
    (train,test)=bnc.split(0.01)
    train_data=qarac.corpora.Batcher.Batcher(train)
    model = qarac.models.qarac_base_model.qarac_base_model(tokenizer.get_vocab_size(), 
                                                           768, 
                                                           12,
                                                           task=='decode')
    optimizer = keras.optimizers.Nadam(learning_rate=keras.optimizers.schedules.ExponentialDecay(1.0e-5, 100, 0.99))
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics='accuracy')
    model.fit(train_data,
              epochs=100,
              workers = 16,
              use_multiprocessing=True)
    test_data=qarac.corpora.Batcher.Batcher(test)
    print(model.evaluate(test_data))
    model.save(filename)
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='QARAC',
                                     description='Experimental NLP system, aimed at improving factual accuracy')
    parser.add_argument('task')
    parser.add_argument('-f','--filename')
    parser.add_argument('-t','--training-task')
    args = parser.parse_args()
    if args.task == 'train_base_model':
        train_base_model(args.training_task,args.filename)
   