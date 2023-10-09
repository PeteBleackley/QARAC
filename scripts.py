
import os
import argparse
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
import torch
import spacy
import pandas
import qarac.utils.CoreferenceResolver
import nltk.corpus
import difflib
import scipy.stats
import scipy.spatial
import seaborn
import tqdm

EPSILON = 1.0e-12

class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super(CombinedLoss,self).__init__()
        self.component_losses = (torch.nn.CrossEntropyLoss(),
                                 torch.nn.MSELoss(),
                                 torch.nn.CrossEntropyLoss(),
                                 torch.nn.MSELoss())
        
    def forward(self,y_pred,y_true):
        return torch.sum((fn(pred,obs)
                          for (fn,pred,obs) in zip(self.component_losses,
                                                   y_pred,
                                                   y_true)))


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
    #optimizer = keras.optimizers.Nadam(learning_rate=keras.optimizers.schedules.ExponentialDecay(1.0e-5, 100, 0.99))
    #model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics='accuracy')
    #model.fit(train_data,
    #          epochs=100,
    #          workers = 16,
    #          use_multiprocessing=True)
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
    tokenizer = tokenizers.Tokenizer.from_pretrained('roberta-base')
    trainer = qarac.models.QaracTrainerModel.QaracTrainerModel('roberta_base', 
                                                               tokenizer)
    loss_fn = CombinedLoss()
    optimizer = torch.optim.NAdam(trainer.parameters(),lr=5.0e-5)
    scheduler = torch.optim.ExponentialDecay(optimizer,gamma=0.9)
    training_data = qarac.corpora.CombinedCorpus.CombinedCorpus(tokenizer,
                                                                all_text='corpora/all_text.csv',
                                                                question_answering='corpora/question_answering.csv',
                                                                reasoning='corpora/reasoning_train.csv',
                                                                consistency='corpora/consistency.csv')
    n_batches = len(training_data)
    history = []
    for epoch in range(10):
        print("Epoch",epoch)
        epoch_history = []
        for (batch,(X,Y)) in enumerate(tqdm.tqdm(training_data)):
            prediction = trainer(X['all_text'],
                                 X['offset_text'],
                                 X['question'],
                                 X['answer'],
                                 X['proposition0'],
                                 X['proposition1'],
                                 X['conclusion_offset'],
                                 X['statement0'],
                                 X['statement1'])
            loss = loss_fn(prediction,Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 1024 == 0 or batch == n_batches-1:
                epoch_history.append({'batch':batch,
                                      'loss':loss.item()})
        scheduler.step()
        history.append(epoch_history)
    with open('training_history.json','w') as jsonfile:
        json.dump(history,jsonfile)
    huggingface_hub.login(token=os.environ['HUGGINGFACE_TOKEN'])
    trainer.question_encoder.push_to_hub('{}/qarac-roberta-question-encoder'.format(path))
    trainer.answer_encoder.push_to_hub('{}/qarac-roberta-answer-encoder'.format(path))
    trainer.decoder.push_to_hub('{}/qarac-roberta-decoder'.format(path))
    
    
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
            input_ids = torch.tensor([sample.ids for sample in batch])
            attention_mask = torch.not_equal(input_ids,pad_token)
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
        input_ids = torch.tensor([sample.ids for sample in batch])
        attention_mask = torch.not_equal(input_ids, pad_token)
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
    question_ids = torch.tensor([question.ids
                                 for question in questions])
    attention_mask = torch.not_equal(question_ids,
                                     pad_token)
    q_vectors = question_encoder(question_ids,
                                 attention_mask=attention_mask).numpy()
    answers = tokenize(data['Resolved_answer'])
    maxlen = max((len(answer) for answer in answers))
    for answer in answers:
        answer.pad(maxlen,pad_id=pad_token)
    answer_ids = torch.tensor([answer.ids
                               for answer in answers])
    attention_mask = torch.not_equal(answer_ids,
                                     pad_token)
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

def test_reasoning(path):
    encoder = transformers.Transformer.from_pretrained('{}/qarac-roberta-answer-encoder'.format(path))
    decoder = transformers.Transformer.from_pretrained('{}/qarac-robeerta-decoder'.format(path))
    tokenizer=tokenizers.Tokenizer.from_pretrained('roberta-base')
    exclude = tokenizer.encode('<s> </s> <pad>').ids
    analyser = difflib.SequenceMatcher(lambda x: x in exclude)
    data = pandas.read_csv('corpora/Avicenna_Test.csv',encoding='iso-8859-1')
    data = data.loc[data['Syllogistic relation']=='yes']
    def tokenize(column):
        return tokenizer.encode_batch(column.apply(lambda x:tokenizers.TextInputSequence(x)),
                                      add_special_tokens=False)
    p0 = tokenize(data['Premise 1'])
    p1 = tokenize(data['Premise 2'])
    c = tokenize(data['Conclusion'])
    p0_batch = []
    p1_batch = []
    c_batch = []
    n=0
    pad_token = tokenizer.token_to_id('<pad>')
    matches=[]
    for (p0_sample,p1_sample,c_sample) in zip(p0,p1,c):
        p0_batch.append(p0_sample)
        p1_batch.append(p1_sample)
        c_batch.append(c_sample)
        n+=1
        if n==32:
            maxlen=max((len(sample for sample in p0_batch)))
            for sample in p0_batch:
                sample.pad(maxlen,pad_token)
            p0_in = torch.tensor([sample.ids for sample in p0.batch])
            p0_attn = torch.not_equal(p0_in,
                                      pad_token)
            maxlen=max((len(sample for sample in p1_batch)))
            for sample in p1_batch:
                sample.pad(maxlen,pad_token)
            p1_in = torch.tensor([sample.ids for sample in p1.batch])
            p1_attn = torch.not_equal(p0_in,
                                      pad_token)
            predictions = decoder.generate(vector=(encoder(p0_in,
                                                           attention_mask=p0_attn)
                                                   +encoder(p1_in,
                                                            attention_mask=p1_attn)))
            for (s1,s2) in zip(c_batch,predictions):
                analyser.set_seqs(s1.ids, s2)
                matches.append(analyser.ratio())
            n=0
            p0_batch=[]
            p1_batch=[]
            c_batch=[]
        if n!=0:
            maxlen=max((len(sample for sample in p0_batch)))
            for sample in p0_batch:
                sample.pad(maxlen,pad_token)
            p0_in = torch.tensor([sample.ids for sample in p0.batch])
            p0_attn = torch.not_equal(p0_in,
                                      pad_token)
            maxlen=max((len(sample for sample in p1_batch)))
            for sample in p1_batch:
                sample.pad(maxlen,pad_token)
            p1_in = torch.tensor([sample.ids for sample in p1.batch])
            p1_attn = torch.not_equal(p0_in,
                                      pad_token)
            predictions = decoder.generate(vector=(encoder(p0_in,
                                                           attention_mask=p0_attn)
                                                   +encoder(p1_in,
                                                            attention_mask=p1_attn)))
            for (s1,s2) in zip(c_batch,predictions):
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
            axes.get_figure().savefig('reasoning_histogram.svg')
        percent = numpy.linspace(0.0,1.0,101)
        percentiles = numpy.quantile(matches,percent)
        with pandas.option_context('plotting.backend','matplotlib.backends.backend_svg') as options:
            axes = pandas.Series(percentiles, index=percent).plot.bar()
            axes.get_figure().savefig('reasoning_percentile.svg')
        
            
def test_consistency(path):
    encoder = transformers.Transformer.from_pretrained('{}/qarac-roberta-answer-encoder'.format(path))
    tokenizer = tokenizer=tokenizers.Tokenizer.from_pretrained('roberta-base')
    data = pandas.read_csv('corpora/snli_1.0_test.csv')
    data = data.loc[data['gold_label']!='-']
    pad_token=tokenizer.token_to_id('<pad>')
    def tokenize(column):
        return tokenizer.encode_batch(column.apply(lambda x:tokenizers.TextInputSequence(x)),
                                      add_special_tokens=False)
    s0 =tokenize(data['sentence1'])
    s1 = tokenize(data['sentence1'])
    maxlen = max((len(sentence for sentence in s0)))
    for sentence in s0:
        sentence.pad(maxlen,pad_id=pad_token)
    s0_in = torch.tensor([sentence.ids for sentence in s0])
    s0_attn = torch.not_equal(s0_in,
                              pad_token)
    maxlen = max((len(sentence for sentence in s1)))
    for sentence in s1:
        sentence.pad(maxlen,pad_id=pad_token)
    s1_in = torch.tensor([sentence.ids for sentence in s1])
    s1_attn = torch.not_equal(s1_in,
                              pad_token)
    s0_vec = encoder(s0_in,attention_mask=s0_attn)
    s0_norm = torch.max(torch.linalg.vector_norm(s0_vec,dim=1),EPSILON)
    s0 = s0_vec/s0_norm
    s1_vec = encoder(s1_in,attention_mask=s1_attn)
    s1_norm = torch.max(torch.linalg.vector_norm(s1_vec,dim=1),EPSILON)
    s1 = s1_vec/s1_norm
    consistency = torch.einsum('ij,ij->i',s0,s1).numpy()
    results = pandas.DataFrame({'label':data['gold_label'],
                                'score':consistency})
    third = 1.0/3.0
    def predicted_labels(x):
        return 'entailment' if x>third else 'contradiction' if x<-third else 'neutral'
    results['prediction'] = results['score'].apply(predicted_labels)
    confusion=results.groupby('label')['prediction'].value_counts().fillna(0)
    seaborn.heatmap(confusion).save('consistency_confusion_matrix.svg')
    correct = pandas.Series({label:confusion[label,label]
                             for label in confusion.index})
    print("Accuracy: {}".format(correct.sum()/data.shape[0]))
    print("Precision")
    print(correct/confusion.sum(axis='columns'))
    print("Recall")
    print(correct/confusion.sum(axis='rows'))
    def stats(group):
        (alpha,beta,loc,scale) = scipy.stats.beta.fit(group)
        mean = group.mean()
        sd = group.std()
        return pandas.Series({'mean':mean,
                              'sd':sd,
                              'min':loc,
                              'max':loc+scale,
                              'alpha':alpha,
                              'beta':beta})
    print(results.groupby('label')['score'].apply(stats))
    quartiles = numpy.quantile(consistency,[0.0,0.25,0.5,0.75,1.0])
    IQR = quartiles[3]-quartiles[1]
    bin_width = 2.0*IQR/(data.shape[0]**1.5)
    n_bins = int((quartiles[4]-quartiles[0])/bin_width)
    bins = numpy.linspace(quartiles[0],quartiles[4],n_bins)
    def hist(col):
        (result,_) = numpy.histogram(col,bins)
        return result
    histograms = results.groupby('label')['score'].apply(hist)
    histograms.coluumns = (bins[1:]+bins[:-1])/2
    with pandas.option_context('plotting.backend','matploblib.backends.backend_svg') as options:
        axes=histograms.T.plot.bar(stacked=True)
        axes.get_figure().savefig('consistency_histograms.svg')
    percent = numpy.linspace(0.0,1.0,101)
    percentiles = results.groupby('label')['score'].apply(lambda x: numpy.percentile(x,percent))
    with pandas.option_context('plotting.backend','matploblib.backends.backend_svg') as options:
        axes=percentiles.T.plot.line()
        axes.get_figure().savefig('consistency_percentiles.svg')
    

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
    elif args.task=="test_reasoning":
        test_reasoning(args.filename)
    elif args.task=='test_consistency':
        test_consistency(args.filename)
   