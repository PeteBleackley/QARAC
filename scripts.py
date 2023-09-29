
import os
import re
import argparse
import pickle
import json
import numpy
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
import nltk.corpus
import difflib
import scipy.stats
import scipy.spatial



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
    training_data = qarac.corpora.CombinedCorpus.CombinedCorpus(tokenizer,
                                                                all_text='corpora/all_text.csv',
                                                                question_answering='corpora/question_answering.csv',
                                                                reasoning='corpora/reasoning_train.csv',
                                                                consistency='corpora/consistency.csv')
    history = trainer.fit(training_data,
                          epochs=10,
                          workers=16,
                          use_multiprocessing=True)
    with open('history.json','w') as jsonfile:
        json.dump(history.history,jsonfile)
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
    
def test_encode_decode(path):
    encoder = transformers.Transformer.from_pretrained('{}/qarac-roberta-answer-encoder'.format(path))
    decoder = transformers.Transformer.from_pretrained('{}/qarac-robeerta-decoder'.format(path))
    tokenizer=tokenizers.Tokenizer.from_pretrained('roberta-base')
    exclude = tokenizer.encode('<s> </s> <pad>').ids
    analyser = difflib.SequenceMatcher(lambda x: x in exclude)
    bnc = nltk.corpus.reader.bnc.BNCCorpusReader('/'.join([os.environ['HOME'],
                                                                'BNC',
                                                                'Texts']),  
                                                      fileids=r'[A-K]/\w*/\w*\.xml')
    matches = []
    batch = []
    pad_token = tokenizer.token_to_id('<pad>')
    for sent in bnc.sents(strip_space=False):
        batch.append(tokenizer.encode(''.join(sent)))
        if len(batch)==32:
            maxlen = max((len(sentence) for sentence in batch))
            for sample in batch:
                sample.pad(maxlen,pad_id=pad_token)
            input_ids = tensorflow.constant([sample.ids for sample in batch])
            attention_mask = tensorflow.constant(numpy.notequal(input_ids.numpy(),
                                                                pad_token).astype(int))
            vectors = encoder(input_ids,
                              attention_mask)
            decoded = decoder.generate(vector=vectors)
            for (s1,s2) in zip(batch,decoded):
                analyser.set_seqs(s1.ids, s2)
                matches.append(analyser.ratio())
            batch = []
    if len(batch)!=0:
        maxlen = max((len(sentence) for sentence in batch))
        for sample in batch:
            sample.pad(maxlen,pad_id=pad_token)
        input_ids = tensorflow.constant([sample.ids for sample in batch])
        attention_mask = tensorflow.constant(numpy.notequal(input_ids.numpy(),
                                                            pad_token).astype(int))
        vectors = encoder(input_ids,
                          attention_mask)
        decoded = decoder.generate(vector=vectors)
        for (s1,s2) in zip(batch,decoded):
            analyser.set_seqs(s1.ids, s2)
            matches.append(analyser.ratio())
        matches = numpy.array(matches)
        print("Accuracy: mean = {0}, sd = {1}".format(matches.mean(),
                                                  matches.sd()))
        (alpha,beta,loc,scale)=scipy.stats.beta.fit(matches,floc=0.0,fscale=1.0)
        print("Beta distribution parameters alpha = {0}, beta = {1}".format(alpha,beta))
        (hist,bins) = numpy.histogram(matches,bins='fd')
        with pandas.option_context('plotting.backend','matploblib.backends.backend_svg') as options:
            axes = pandas.Series(hist,index=(bins[1:]+bins[:-1]/2)).plot.bar()
            axes.get_figure().savefig('encode_decode_histogram.svg')
        percent = numpy.linspace(0.0,1.0,101)
        percentiles = numpy.quantile(matches,percent)
        with pandas.option_context('plotting.backend','matplotlib.backends.backend_svg') as options:
            axes = pandas.Series(percentiles, index=percent).plot.bar()
            axes.get_figure().savefig('encode_decode_percentile.svg')
        
        
def test_question_answering(path):
    question_encoder = transformers.Transformer.from_pretrained('{}/qarac-roberta-question-encoder'.format(path))
    answer_encoder = transformers.Transformer.from_pretrained('{}/qarac-roberta-answer-encoder'.format(path))
    tokenizer = tokenizers.Tokenizer.from_pretrained('roberta-base')
    data = pandas.read_csv('WikiQA.tsv',sep='\t')
    data['QNum']=data['QuestionID'].apply(lambda x: int(x[1:]))
    nlp = spacy.load('en_core_web_trf')
    predictor = qarac.utils.CoreferenceResolver.CoreferenceResolver()
    data['Resolved_answer'] = data.groupby('QNum')['Sentence'].transform(predictor)
    unique_questions = data.groupby('QNum')['Question'].first()
    cleaned_questions = pandas.Series([clean_question(doc)
                                       for doc in nlp.pipe(unique_questions)],
                                      index = unique_questions.index)

    def tokenize(column):
        return tokenizer.encode_batch(column.apply(lambda x:tokenizers.TextInputSequence(x)),
                                      add_special_tokens=False)
    questions = tokenize(cleaned_questions)
    maxlen=max((len(question) for question in questions))
    pad_token = tokenizer.token_to_id('<pad>')
    for question in questions:
        question.pad(maxlen,pad_id=pad_token)
    question_ids = tensorflow.constant([question.ids
                                        for question in questions])
    attention_mask = tensorflow.constant(numpy.not_equal(question_ids.numpy(),
                                                         pad_token).astype(int))
    q_vectors = question_encoder(question_ids,
                                 attention_mask=attention_mask).numpy()
    answers = tokenize(data['Resolved_answer'])
    maxlen = max((len(answer) for answer in answers))
    for answer in answers:
        answer.pad(maxlen,pad_id=pad_token)
    answer_ids = tensorflow.constant([answer.ids
                                      for answer in answers])
    attention_mask = tensorflow.constant(numpy.not_equal(answer_ids.numpy(),
                                                         pad_token).astype(int))
    answer_lookup = scipy.spatial.KDTree(answer_encoder(answer_ids,
                                                        attention_mask=attention_mask).numpy())
    n_correct = 0
    all_distances = 0.0
    correct_distances = 0.0
    wrong_distances = 0.0
    all_sq = 0.0
    correct_sq = 0.0
    wrong_sq = 0.0
    for (i,qv) in enumerate(q_vectors):
        (d,row) = answer_lookup.query(qv)
        dsq=d**2.0
        correct = (row['QNum']==i and row['Label']==1)
        all_distances+=d
        all_sq+=dsq
        if correct:
            n_correct+=1
            correct_distances+=d
            correct_sq+=dsq
        else:
            wrong_distances+=d
            wrong_sq+=dsq
    N = cleaned_questions.shape[0]
    print("{0} questions, {1} possible answers, {2} correct answers".format(N,
                                                                            data.shape[0],
                                                                            n_correct))
    accuracy = n_correct/N
    baseline = N/data.shape[0] 
    kappa = 1.0 - ((1.0-accuracy)/(1.0-baseline))      
    print(("Accuracy: {0}, Baseline {1}, kappa{2} ".format(accuracy,baseline,kappa)))   
    mean_dist =all_distances/N
    mean_sq = all_sq/N
    all_sd = numpy.sqrt(mean_sq-(mean_dist**2.0))  
    print("Question-answer distances")        
    print("All: mean {0}, sd {1}".format(mean_dist,all_sd))   
    correct_mean = correct_distances/n_correct
    correct_meansq = correct_sq/n_correct
    correct_sd = numpy.sqrt(correct_meansq - (correct_mean**2.0))    
    print("Correct: mean {0}, sd {1}".format(correct_mean,correct_sd))     
    wrong_mean = wrong_distances/(N-n_correct)  
    wrong_meansq = wrong_sq/(N-n_correct)     
    wrong_sd = numpy.sqrt(wrong_meansq - (wrong_mean**2.0))    
    print("Wrong: mean {0}, sd {1}".format(wrong_mean,wrong_sd))               
    
    
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
    elif args.task == 'test_encode_decode':
        test_encode_decode(args.filename)
    elif args.task== 'test_question_answering':
        test_question_answering(args.filename)
   