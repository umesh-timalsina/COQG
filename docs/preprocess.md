# Preprocessing the COQA Dataset

## Preprocessing the JSONs

Each story has a bunch of questions attached to it, their turn ID as well as answers and a rationale in the paragraph for the answer. Following steps were taken preprocess the COQA dataset. Since, we are training the dataset with Open_NMT. This requires to create two files <code>src-train.txt</code> and <code>tgt-train.txt</code>. Where the files are line based.

### How to generate the <code>src-train.txt</code>?
1. Take each story. Fix the question and rationale for this turn
2. For each paragraph in the story. Tokenize the story using Stanford Core NLP. 
3. Annotate each token with linguistic feature. Features used are: POS, NER and CASE
4. If the token happens to be in the current rationale, add an extra feature : Answer
5. Based on number of questions to put in history, append the tokenized and annotated previous question to this story.

A sample line in <code>src-train.txt</code> looks like this:

```
The|U|DT|O|- Vatican|U|NNP|O|- Apostolic|U|NNP|IDEOLOGY|- Library|U|NNP|O|- (|L|-LRB-|O|- )|L|-RRB-|O|- ,|L|,|O|- more|L|RBR|O|- commonly|L|RB|O|- called|L|VBN|O|- the|L|DT|O|- Vatican|U|NNP|LOCATION|- Library|U|NNP|LOCATION|- or|L|CC|O|- simply|L|RB|O|- the|L|DT|O|- Vat|U|NNP|O|- ,|L|,|O|- is|L|VBZ|O|- the|L|DT|O|- library|L|NN|O|- of|L|IN|O|- the|L|DT|O|- Holy|U|NNP|O|- See|U|VB|O|- ,|L|,|O|- located|L|JJ|O|- in|L|IN|O|- Vatican|U|NNP|LOCATION|- City|U|NNP|LOCATION|- .|L|.|O|- Formally|U|RB|O|- established|L|VBN|O|- in|L|IN|O|- 1475|L|CD|DATE|- ,|L|,|O|- although|L|IN|O|- it|L|PRP|O|- is|L|VBZ|O|- much|L|JJ|O|- older|L|JJR|O|- ,|L|,|O|- it|L|PRP|O|- is|L|VBZ|O|- one|L|CD|NUMBER|- of|L|IN|O|- the|L|DT|O|- oldest|L|JJS|O|- libraries|L|NNS|O|- in|L|IN|O|- the|L|DT|O|- world|L|NN|O|- and|L|CC|O|- contains|L|VBZ|O|- one|L|CD|NUMBER|- of|L|IN|O|- the|L|DT|O|- most|L|RBS|O|- significant|L|JJ|O|- collections|L|NNS|O|- of|L|IN|O|- historical|L|JJ|O|- texts|L|NNS|O|- .|L|.|O|- It|U|PRP|O|- has|L|VBZ|O|- 75,000|L|CD|NUMBER|- codices|L|NNS|O|- from|L|IN|O|- throughout|L|IN|O|- history|L|NN|O|- ,|L|,|O|- as|L|RB|O|- well|L|RB|O|- as|L|IN|O|- 1.1|L|CD|NUMBER|- million|L|CD|NUMBER|- printed|L|VBN|O|- books|L|NNS|O|- ,|L|,|O|- which|L|WDT|O|- include|L|VBP|O|- some|L|DT|O|- 8,500|L|CD|NUMBER|- incunabula|L|NN|O|- .|L|.|O|- The|U|DT|O|- Vatican|U|NNP|ORGANIZATION|A|- Library|U|NNP|ORGANIZATION|A|- is|L|VBZ|O|A|- a|L|DT|O|A|- research|L|NN|O|A|- library|L|NN|O|A|- for|L|IN|O|A|- history|L|NN|O|A|- ,|L|,|O|A|- law|L|NN|O|A|- ,|L|,|O|- philosophy|L|NN|O|- ,|L|,|O|- science|L|NN|O|- and|L|CC|O|- theology|L|NN|O|- .|L|.|O|- The|U|DT|O|- Vatican|U|NNP|ORGANIZATION|- Library|U|NNP|ORGANIZATION|- is|L|VBZ|O|- open|L|JJ|O|- to|L|TO|O|- anyone|L|NN|O|- who|L|WP|O|- can|L|MD|O|- document|L|VB|O|- their|L|PRP$|O|- qualifications|L|NNS|O|- and|L|CC|O|- research|L|NN|O|- needs|L|NNS|O|- .|L|.|O|- Photocopies|U|NNS|O|- for|L|IN|O|- private|L|JJ|O|- study|L|NN|O|- of|L|IN|O|- pages|L|NNS|O|- from|L|IN|O|- books|L|NNS|O|- published|L|VBN|O|- between|L|IN|O|- 1801|L|CD|DATE|- and|L|CC|DATE|- 1990|L|CD|DATE|- can|L|MD|O|- be|L|VB|O|- requested|L|VBN|O|- in|L|IN|O|- person|L|NN|O|- or|L|CC|O|- by|L|IN|O|- mail|L|NN|O|- .|L|.|O|- In|U|IN|O|- March|U|NNP|DATE|- 2014|L|CD|DATE|- ,|L|,|O|- the|L|DT|O|- Vatican|U|NNP|LOCATION|- Library|U|NNP|LOCATION|- began|L|VBD|O|- an|L|DT|O|- initial|L|JJ|O|- four-year|L|JJ|DURATION|- project|L|NN|O|- of|L|IN|O|- digitising|L|VBG|O|- its|L|PRP$|O|- collection|L|NN|O|- of|L|IN|O|- manuscripts|L|NNS|O|- ,|L|,|O|- to|L|TO|O|- be|L|VB|O|- made|L|VBN|O|- available|L|JJ|O|- online|L|NN|O|- .|L|.|O|- The|U|DT|O|- Vatican|U|NNP|O|- Secret|U|NNP|O|- Archives|U|NNPS|O|- were|L|VBD|O|- separated|L|VBN|O|- from|L|IN|O|- the|L|DT|O|- library|L|NN|O|- at|L|IN|O|- the|L|DT|DATE|- beginning|L|NN|DATE|- of|L|IN|DATE|- the|L|DT|DATE|- 17th|L|JJ|DATE|- century|L|NN|DATE|- ;|L|:|O|- they|L|PRP|O|- contain|L|VBP|O|- another|L|DT|O|- 150,000|L|CD|NUMBER|- items|L|NNS|O|- .|L|.|O|- Scholars|U|NNS|O|- have|L|VBP|O|- traditionally|L|RB|O|- divided|L|VBN|O|- the|L|DT|O|- history|L|NN|O|- of|L|IN|O|- the|L|DT|O|- library|L|NN|O|- into|L|IN|O|- five|L|CD|NUMBER|- periods|L|NNS|O|- ,|L|,|O|- Pre-Lateran|L|NNP|O|- ,|L|,|O|- Lateran|U|NNP|PERSON|- ,|L|,|O|- Avignon|U|NNP|CITY|- ,|L|,|O|- Pre-Vatican|L|NNP|O|- and|L|CC|O|- Vatican|U|NNP|LOCATION|- .|L|.|O|- The|U|DT|O|- Pre-Lateran|L|JJ|MISC|- period|L|NN|O|- ,|L|,|O|- comprising|L|VBG|O|- the|L|DT|DURATION|- initial|L|JJ|DURATION|- days|L|NNS|DURATION|- of|L|IN|O|- the|L|DT|O|- library|L|NN|O|- ,|L|,|O|- dated|L|VBN|O|- from|L|IN|O|- the|L|DT|O|- earliest|L|JJS|O|- days|L|NNS|DURATION|- of|L|IN|O|- the|L|DT|O|- Church|U|NNP|ORGANIZATION|- .|L|.|O|- Only|U|RB|O|- a|L|DT|O|- handful|L|NN|O|- of|L|IN|O|- volumes|L|NNS|O|- survive|L|VBP|O|- from|L|IN|O|- this|L|DT|O|- period|L|NN|O|- ,|L|,|O|- though|L|IN|O|- some|L|DT|O|- are|L|VBP|O|- very|L|RB|O|- significant|L|JJ|O|- .|L|.|O|-|| <q> what|L|WP|O|- is|L|VBZ|O|- the|L|DT|O|- library|L|NN|O|- for|L|IN|O|- ?|L|.|O|- <q>

```

### How to generate <code>tgt-train.txt</code> 
Note that both <code>src-train</code> and <code>tgt-train</code> are generated simultaneously. The distinction is made here for clarity's sake:
1. For a turn, tokenize and annotate the questions
2. Append the question to a new line in the file.

__Potential PitFall__: POS, NER and CASE tags are used for <code>tgt-train</code>, which in itself is not an enhancement or not fruitful.

## Preparing the files for training
Open-NMT provides a script to embed the files mentioned before. We used the following two scripts to generate word embeddings with <code>glove-8B-300d</code>.

```bash
cd ../OpenNMT-py
python3 preprocess.py \
-train_src ../data/train/coqa-src-train-v1.0.txt \
-train_tgt ../data/train/coqa-tgt-train-v1.0.txt \
-valid_src ../data/dev/coqa-src-valid-v1.0.txt \
-valid_tgt ../data/dev/coqa-tgt-valid-v1.0.txt \
-save_data ../data/data.feat.coqg \
-src_seq_length 10000 -tgt_seq_length 10000 \
-dynamic_dict  # Creates a .pt Files for training and validation

PYTHONPATH=../OpenNMT-py python ../OpenNMT-py/tools/embeddings_to_torch.py \
 -emb_file_enc ../data/glove.42B.300d.txt \
 -emb_file_dec ../data/glove.42B.300d.txt \
 -dict_file ../data/data.feat.coqg.vocab.pt \
 -output_file ../data/data.feat.coqg.embed  # Creates a vocabulary dictionary with Glove

```