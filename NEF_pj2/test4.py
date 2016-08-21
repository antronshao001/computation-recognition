words_list = ['blue', 'yellow', 'purple', 'white', 'pink', 'green', 'soxoneonta', 'stripes', 'dragonhood', 'hooped',
              'red', 'turquoise', 'rotunda', 'rentable', 'enix', 'frontage', 'cupola', 'triangle', 'sowwah', 'mickewicz',
              'manezhnaya', 'hillman', 'llc', 'widett', 'landon']

for word in words_list:
    vocab.add(word.upper(), corpus[word])

dimensions = 400