import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = dict()

    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(directory, filename)) as f:
            corpus[filename] = f.read()

    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = list()
    tokens = nltk.word_tokenize(document.lower())

    for token in tokens:
        if token not in string.punctuation and token not in nltk.corpus.stopwords.words("english"):
            words.append(token)

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    # 'key' = word, 'count' = number of times used in all documents
    word_count = dict()
    num_docs = len(documents)

    # count number of times words appear in each document
    for document in documents:
        # To make sure we don't count each word multiple times
        words = set(documents[document])

        for word in words:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
    
    # Compute idfs values for each word
    for word in word_count:
        idfs[word] = math.log(num_docs/word_count[word])

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    doc_scores = {file:0 for file in files}

    # Iterate through each word in the query
    # and update each file's tf-idf value
    for word in query:
        if word in idfs:
            for file in files:
                tf = files[file].count(word)
                tf_idf = tf*idfs[word]
                doc_scores[file] += tf_idf
    
    # Sort and return the top 'n' files
    sorted_scores = [k for k, v in sorted(doc_scores.items(), key=lambda x:x[1], reverse=True)]

    return sorted_scores[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = dict()

    # Get the sentence score and density score for each sentence
    for sentence, words in sentences.items():
        score = 0
        for word in query:
            if word in words:
                score += idfs[word]
        
        if score != 0:
            density = sum([words.count(x) for x in query])/len(words)
            sentence_scores[sentence] = (score, density)
    

    sorted_scores = [k for k, v in sorted(sentence_scores.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)]

    return sorted_scores[:n]


if __name__ == "__main__":
    main()
