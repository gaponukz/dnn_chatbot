from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.framework import ops
import tensorflow
import tflearn
import nltk
import numpy
import random

class chatbot(object):
    def __init__(self):
        super(chatbot, self).__init__()
        self.stemmer = LancasterStemmer()
        self.data = {}
        self.model = None
        self.words = None
        self.labels = None
        nltk.download('punkt')
    
    def add_topic(self, *args, **kwargs) -> 'chatbot':
        self.data[kwargs['topic_name']] = {
            "patterns": kwargs['patterns'],
            "responses": kwargs['responses']
        }

        return self
    
    def train_model(self, model_name: str) -> 'chatbot':
        self.words, self.labels = [], []
        docs_x, docs_y = [], []

        for topic_name in self.data:
            intent = self.data[topic_name]
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                self.words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(topic_name)

            if topic_name not in self.labels:
                self.labels.append(topic_name)

        self.words = sorted(list(set([self.stemmer.stem(w.lower()) for w in self.words if w != "?"])))
        self.labels = sorted(self.labels)

        training, output = [], []
        out_empty = [0 for _ in range(len(self.labels))]

        for x, doc in enumerate(docs_x):
            wrds = [self.stemmer.stem(w.lower()) for w in doc]
            bag = [int(w in wrds) for w in self.words]

            output_row = out_empty[:]
            output_row[self.labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = numpy.array(training)
        output = numpy.array(output)

        ops.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)

        self.model.fit(
            training, 
            output, 
            n_epoch = 1000, 
            batch_size = 8,
            show_metric = True
        )
        
        self.model.save(f"{model_name}.tflearn")
        
        return self
    
    def __bag_of_words(self, u_input: str, words: list) -> numpy.array:
        self.bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(u_input)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    self.bag[i] = 1
                    
        return numpy.array(self.bag)
    
    def get_result(self, pattern: str) -> str:
        result = self.model.predict([self.__bag_of_words(pattern, self.words)])[0]
        result_index = numpy.argmax(result)
        tag = self.labels[result_index]
        if result[result_index] > 0.5:
            return random.choice(self.data[tag]["responses"])

        else:
            return "None("
