import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
from keytotext import pipeline
from rake_new2 import Rake
from nltk.corpus import wordnet


# Get the input sentence 
text = input("Enter a sentence: \n")

# Initialize Rake for keyword extraction 
# Refer to: https://github.com/BALaka-18/rake_new2
tokenizer = Rake()
tokenizer.get_keywords_from_raw_text(text)
keywords = list(tokenizer.get_kw_degree().keys())

# A dictionary to store the synonyms as key value pairs
# Sample:
# {
#      "magnificent": ["brilliant", "glorious"],
# } 
synonyms = {}
num_synonyms = 2

# Iterate over all words and get the synonyms from wordnet
for word in keywords:
    synonyms[word] = []
    for syn in wordnet.synsets(word):
        if len(synonyms[word]) > num_synonyms:
            break
        for name in enumerate(syn.lemma_names()):
            if len(synonyms[word]) > num_synonyms:
                break
            if word not in name[1]:
                synonyms[word].append(name[1].replace("_"," "))

# Initialize a trained model with a keytotext pipeline
# Refer to: https://github.com/gagan3012/keytotext
model = pipeline("mrm8488/t5-base-finetuned-common_gen")

# Iterate over each word to print a sentance with the synonyms 
# of the words using the model
for word in synonyms:
    if synonyms[word] == []:
        continue
    print("Word :  ", word)
    for synonym in synonyms[word]:
        print(synonym, " ----" ,model([synonym]))
