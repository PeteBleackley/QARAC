
import os
import re
import argparse
import pickle
import tokenizers
import transformers
import huggingface_hub
import qarac.corpora.BNCorpus
import qarac.corpora.Batcher
import qarac.models.qarac_base_model
import qarac.models.QaracTrainerModel
import qarac.corpora.CombinedCorpus
import keras
import tensorflow
import spacy
import pandas
import qarac.utils.CoreferenceResolver



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
    data['QNum']=data['QuestionID'].apply(lambda x: int(x[1:]))
    nlp = spacy.load('en_core_web_trf')
    predictor = qarac.utils.CoreferenceResolver.CoreferenceResolver()
    data['Resolved_answer'] = data.groupby('QNum')['Sentence'].transform(predictor)
    unique_questions = data.groupby('QNum')['Question'].first()
    cleaned_questions = pandas.Series([clean_question(doc)
                                       for doc in nlp.pipe(unique_questions)],
                                      index = unique_questions.index)
    for (i,question) in cleaned_questions.items():
        data.loc[data['QNum']==i,'Cleaned_question']=question
    data[['Cleaned_question','Resolved_answer','Label']].to_csv(outfilename)

        
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
    
def prepare_training_datasets():
    wikiqa = pandas.read_csv('corpora/WikiQA.csv')
    avicenna = pandas.read_csv('corpora/Avicenna_Train.csv',encoding='iso-8859-1')
    snli = pandas.read_csv('corpora/snli_1.0_train.csv')
    question_answering = wikiqa.loc[wikiqa['Label']==1,
                                    ['Cleaned_question',
                                     'Resolved_answer']].rename(columns={'Cleaned_question':'question',
                                                                         'Resolved_answer':'answer'})
    reasoning = avicenna.loc[avicenna['Syllogistic relation']=='yes',
                             ['Premise 1',
                              'Premise 2',
                              'Conclusion']].rename(columns={'Premise 1':'proposition0',
                                                              'Premise 2':'proposition1',
                                                              'Conclusion':'conclusion'})
    consistency = snli.loc[snli['gold_label']!='-',
                           ['sentence1',
                            'sentence2']].rename(columns={'sentence1':'statement0',
                                                          'sentence2':'statement1'})
    mapping = {'entailment':1.0,
               'neutral':0.0,
               'contradiction':-1.0}
    consistency['consistency'] = snli.loc[snli['gold_label']!='-',
                                          'gold_label'].apply(lambda x:mapping[x])
    all_text = pandas.concat([wikiqa['Resolved_answer'],
                              avicenna['Premise 1'],
                              avicenna['Premise 1'],
                              reasoning['conclusion'],
                              snli['sentence1'],
                              snli['sentence2']]).to_frame(name='all_text').reset_index(drop=True)
    all_text.to_csv('corpora/all_text.csv')
    question_answering.to_csv('corpora/question_answering.csv')
    reasoning.to_csv('corpora/reasoning_train.csv')
    consistency.to_csv('corpora/consistency.csv')
    
def train_models(path):
    encoder_base = transformers.TFRobertaModel.from_pretrained('roberta-base')
    config = encoder_base.config
    config.is_decoder = True
    decoder_base = transformers.TFRobertaModel.from_pretrained('roberta-base',
                                                               config=config)
    tokenizer = tokenizers.Tokenizer.from_pretrained('roberta-base')
    trainer = qarac.models.QaracTrainerModel.QaracTrainerModel(encoder_base, 
                                                               decoder_base, 
                                                               tokenizer)
    losses={'encode_decode':decoder_loss,
            'question_answering':keras.losses.mean_squared_error,
            'reasoning':decoder_loss,
            'consistency':keras.losses.mean_squared_error}
    optimizer = keras.optimizers.Nadam(learning_rate=keras.optimizers.schedules.ExponentialDecay(1.0e-5, 100, 0.99))
    trainer.compile(optimizer=optimizer,
                    loss=losses)
    training_data = qarac.corpora.CombinedCorpus(tokenizer,
                                                 all_text='corpora/all_text.csv',
                                                 question_answering='corpora/question_answering.csv',
                                                 reasoning='corpora/reasoning_train.csv',
                                                 consistency='corpora/consistency.csv')
    trainer.fit(training_data,
                epochs=10,
                workers=16,
                use_multiprocessing=True)
    huggingface_hub.login(token=os.environ['HUGGINGFACE_TOKEN'])
    trainer.question_encoder.push_to_hub('{}/qarac-roberta-question-encoder'.format(path))
    trainer.answer_encoder.push_to_hub('{}/qarac-roberta-answer-encoder'.format(path))
    trainer.decoder.push_to_hub('{}/qarac-roberta-decoder'.format(path))
    with open('model_summaries.txt') as summaries:
        summaries.write('TRAINER MODEL\n')
        summaries.write(trainer.summary())
        summaries.write('QUESTION ENCODER\n')
        summaries.write(trainer.question_encoder.summary())
        summaries.write('ANSWER ENCODER\n')
        summaries.write(trainer.answer_encoder.summary())
        summaries.write('DECODER\n')
        summaries.write(trainer.decoder.summary())
    keras.utils.plot_model(trainer,'trainer_model.png')
    keras.utils.plot_model(trainer.answer_encoder,'encoder_model.png')
    keras.utils.plot_model(trainer.decoder,'decoder_model.png')
                                                              
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='QARAC',
                                     description='Experimental NLP system, aimed at improving factual accuracy')
    parser.add_argument('task')
    parser.add_argument('-f','--filename')
    parser.add_argument('-t','--training-task')
    parser.add_argument('-o','--outputfile')
    args = parser.parse_args()
    if args.task == 'train_base_model': 
        train_base_model(args.training_task,args.filename)
    elif args.task == 'prepare_wiki_qa':
        prepare_wiki_qa(args.filename,args.outputfile)
    elif args.task == 'prepare_training_datasets':
        prepare_training_datasets()
    elif args.task == 'train_models':
        train_models(args.filename)
   