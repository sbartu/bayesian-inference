"""
Machine Learning - Author Attribution.

By Bartu Sivri.
"""
import sys
import os
import glob
import re
import string
import math
import matplotlib.pyplot as plt


class Sample:
    """Data Holder For Sample Text Files."""

    features_count = 0

    def __init__(self, path, words):
        """Store Words and Store Path For Accuracy Check Later."""
        self.path = path.split('problems/')[1]
        self.words = set(words)


    def fill_feature_vector(self, features):
        self.feature_vector = [0] * len(features)
        
        for index in range(0, len(features)):
            if features[index] in self.words:
                self.feature_vector[index] += 1


    def debug_print(self):
        print('Sample path = {}'.format(self.path))
        print('Total document count is = {}'.format(Training.total_docs))
        print('Label Probability P(c) = {}'.format(self.label_prob))
        print('Number of Features = {}'.format(Training.features_count))
        print('')
        print('-----Feature Vector-----')
        print(self.feature_vector)
        print('')
        print('-----P(fi | c)-----')
        print(self.fi_probs)
        print('')


class Training:
    """Data Holder For Training Text Files."""
    
    total_docs = 0
    features_count = 0

    def __init__(self, author, words_list, count):
        """Store All Words Used by an Author Across All given Training Texts."""
        self.author = 'Author' + author
        self.word_list = []
        self.word_sets = []

        for words in words_list:
            self.word_list.append(words)
            self.word_sets.append(set(words))

        self.num_of_docs = count
        Training.total_docs += self.num_of_docs


    def calc_label_prob(self):
        self.label_prob = self.num_of_docs / Training.total_docs
        

    def debug_print(self):
        print('Author = {} has {} docs'.format(self.author, self.num_of_docs))
        print('Total document count is = {}'.format(Training.total_docs))
        print('Label Probability P(c) = {}'.format(self.label_prob))
        print('Number of Features = {}'.format(Training.features_count))
        print('')
        print('-----Feature Vector-----')
        print(self.feature_vector)
        print('')
        print('-----P(fi | c)-----')
        print(self.fi_probs)
        print('')


    def fill_feature_vector(self, features):
        self.feature_vector = [0] * Training.features_count
        
        for index in range(0, len(features)):
            for words in self.word_sets:
                if features[index] in words:
                    self.feature_vector[index] += 1


    def count_occurence(self, features):
        self.count_vector = [0] * len(features)
        
        for index in range(0, len(features)):
            for words in self.word_list:
                self.count_vector[index] += words.count(features[index])


    def freq_calcs(self, sub_features, size):
        self.feature_vector = [0] * size
        
        for index in range(0, size):
            for words in self.word_sets:
                if sub_features[index] in words:
                    self.feature_vector[index] += 1
                    
        self.fi_probs = [0] * size
        
        for index in range(0, len(self.feature_vector)):
            self.fi_probs[index] = (self.feature_vector[index] + 1) / (self.num_of_docs + 2)

    def calc_fi_prob(self):
        self.fi_probs = [0] * Training.features_count
        
        for index in range(0, len(self.feature_vector)):
            self.fi_probs[index] = (self.feature_vector[index] + 1) / (self.num_of_docs + 2)


def fill_features_list(stop_words, features):
    """Store Each Given Feature In a List."""
    with open(stop_words, errors='ignore') as input_file:
            for line in input_file:
                modified_line = line.replace('\n', '')
                features.append(modified_line)

    n = len(features)
    Training.features_count = n
    Sample.features_count = n


def tokenize(input_files, samples, trainings):
    """Tokenize Every File and Parse Accordingly."""
    def strip_whitespace(input_string):
        """Strip Whitespace from Input String."""
        return re.sub("\s+", " ", input_string.strip())

    training_by_author = {}

    for file_path in input_files:
        file_path = file_path.replace('\\', '/')
        words = []

        with open(file_path, errors='ignore') as input_file:
            for line in input_file:
                whitespace_stripped = strip_whitespace(line)
                punctuation_removed = "".join([x for x in whitespace_stripped
                                               if x not in string.punctuation])
                lowercased = punctuation_removed.lower()
                words.extend(lowercased.split())

        file_name = os.path.basename(file_path)

        # if its a sample file
        if file_name[1] == 's':
            samples.append(Sample(file_path, words))

        # if its a training file
        else:
            author_no = file_name.split('train')[1].split('-')[0]

            if author_no not in training_by_author:
                training_by_author[author_no] = {
                    'words' : [],
                    'count' : 1
                }
            else:
                training_by_author[author_no]['count'] += 1

            training_by_author[author_no]['words'].append(words)

    for author in training_by_author:
        trainings.append(Training(
            author,
            training_by_author[author]['words'],
            training_by_author[author]['count']
        ))


def do_training(trainings, features):

    for train in trainings:
        train.calc_label_prob()

        train.fill_feature_vector(features)

        train.calc_fi_prob()


def sum_count_vectors(trainings, features):

    for train in trainings:
        train.count_occurence(features)

    total_count_vector = [0] * Training.features_count

    for train in trainings:
        count_vec = train.count_vector

        for i in range(0, len(count_vec)):
            total_count_vector[i] += count_vec[i]

    return total_count_vector


def do_testing(sample, trainings, features, answers):

    answers[sample.path] = {
        'max' : float('-inf'),
        'author' : ''
    }
    sample.fill_feature_vector(features)

    for training in trainings:
        result = calculate_class(sample, training)

        if result > answers[sample.path]['max']:
            answers[sample.path]['max'] = result
            answers[sample.path]['author'] = training.author


def calculate_class(sample, training):

    result = math.log2(training.label_prob)

    for i in range(0, len(sample.feature_vector)):
        if sample.feature_vector[i] == 1:
            result += math.log(training.fi_probs[i], 2)
        else:
            result += math.log(1 - training.fi_probs[i], 2)

    return result


def build_answer_dict(answer_file, problem_label, truth_dict):

    with open(answer_file, errors='ignore') as input_file:
        for line in input_file:
            if len(line) > 1 and line[7] == problem_label:
                mod_line = line[:-1]
                answer_line = mod_line.split(' ')
                truth_dict[answer_line[0]] = answer_line[1]


def test_acc(answers, truth_dict):

    x = 0
    n = len(answers)

    for sample, val in answers.items():
        if truth_dict[sample] == val['author']:
            x += 1

    acc = (x / n) * 100
    print('Accuracy:')
    print('---------')
    print(acc)
    print('')


def computeConfusionMatrix(predicted, groundTruth, nAuthors):
    confusionMatrix = [[0 for i in range(nAuthors+1)] for j in range(nAuthors+1)]

    for i in range(len(groundTruth)):
        confusionMatrix[predicted[i]][groundTruth[i]] += 1

    return confusionMatrix


def outputConfusionMatrix(confusionMatrix):
    columnWidth = 4

    print('Confusion Matrix:')
    print('-----------------')
    
    print(str(' ').center(columnWidth),end=' ')
    for i in range(1,len(confusionMatrix)):
        print(str(i).center(columnWidth),end=' ')

    print()

    for i in range(1,len(confusionMatrix)):
        print(str(i).center(columnWidth),end=' ')
        for j in range(1,len(confusionMatrix)):
            print(str(confusionMatrix[j][i]).center(columnWidth),end=' ')
        print()
    print('')


def calc_cce(trainings, features):

    top = 20
    CCE = [0] * Training.features_count

    for i in range(0, len(features)):
        res = 0
        for train in trainings:
            fi_prob = train.fi_probs[i]
            res += train.label_prob * fi_prob * math.log(fi_prob, 2)

        CCE[i] = -res


    print('Top Features:')
    print('-------------')
    for x in range(0, top):
        m = max(CCE)
        max_index = CCE.index(m)
        print('{}: {}'.format(features[max_index], m))
        CCE[max_index] = 0
    print('')


def frequency_training(total_feature_vector, features, trainings, samples, truth_dict, freq_acc):

    feature_ranking = total_feature_vector.copy()
    sub_features = []
    current_num = 10

    print('Training w/ Frequent Features:')
    print('------------------------------')

    while current_num < Training.features_count:
        sub_answers = dict()
        for i in range(0, 10):
            m = max(feature_ranking)
            max_index = feature_ranking.index(m)
            sub_features.append(features[max_index])
            feature_ranking[max_index] = -1

        for train in trainings:
            train.freq_calcs(sub_features, current_num)

        for sample in samples:
            do_testing(sample, trainings, sub_features, sub_answers)

        x = 0
        n = len(sub_answers)

        for sample, val in sub_answers.items():
            if truth_dict[sample] == val['author']:
                x += 1

        acc = (x / n)
        print('{}: {}'.format(current_num, acc))
        freq_acc.append(acc)

        current_num += 10


def create_plot(freq_acc, problem_label):

    feature_nums = []
    name = 'graph_' + problem_label
    i = 10
    while i < Training.features_count:
        feature_nums.append(i)
        i += 10

    plt.rcParams["figure.figsize"] = [16,9]
    plt.scatter(feature_nums, freq_acc)
    plt.plot(feature_nums, freq_acc)
    plt.savefig(name)
    plt.show()


def main():
    """Author Attribution.

    Train on Data With Given Features.
    Find Author of Given Sample Files.
    """
    n_authors = 13
    samples = []
    trainings = []
    features = []
    answers = dict()
    truth_dict = dict()
    input_folder = sys.argv[1]
    problem_label = input_folder[-2]

    stop_words = 'stopwords.txt'
    ground_truth = 'test_ground_truth.txt'
    build_answer_dict(ground_truth, problem_label, truth_dict)

    fill_features_list(stop_words, features)

    input_files = glob.glob(input_folder + '/*.txt')
    tokenize(input_files, samples, trainings)

    do_training(trainings, features)
    total_count_vector = sum_count_vectors(trainings, features)

    for sample in samples:
        do_testing(sample, trainings, features, answers)

    test_acc(answers, truth_dict)

    predicted_authors = []
    truth_authors = []
    freq_acc = []

    for key, val in answers.items():
        no = val['author'][-2:]
        if no[0] == '0':
            no = no[-1:]
        predicted_authors.append(int(no))

    for key, val in truth_dict.items():
        no = val[-2:]
        if no[0] == '0':
            no = no[-1:]
        truth_authors.append(int(no))

    # print(predicted_authors)
    # print(truth_authors)
    outputConfusionMatrix(computeConfusionMatrix(predicted_authors, truth_authors, n_authors))

    calc_cce(trainings, features)

    frequency_training(total_count_vector, features, trainings, samples, truth_dict, freq_acc)

    # create_plot(freq_acc, problem_label)


if __name__ == '__main__':

    main()
