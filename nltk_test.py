
import json
import nltk

def main():
	sentences = [ \
		"Madras curry powder", \
		"cream cheese soften", \
		"A Taste Thai Rice Noodles", \
		"oz diced tomato", \
		"Alexia Waffle Fries", \
		"Alaskan king crab leg", \
	]

	for sentence in sentences:
		tokens = nltk.word_tokenize(sentence)
		tokens = filter(lambda x: x!='oz', tokens)
		
		tagged = nltk.pos_tag(tokens)
		
		entities = nltk.chunk.ne_chunk(tagged)
		
		print entities
		#print tokens
		#print tagged

if __name__=='__main__':
	main()
