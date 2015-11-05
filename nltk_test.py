
import json
import nltk
import io
from nltk import pos_tag, word_tokenize
from nltk.corpus import brown

def main():
	dictionary = {}
	sentences = []
	
	with io.open("out/ingredient_list_train_cleaned.json", encoding = 'utf8') as data_file:
		sentences = json.load(data_file)
		data_file.close()
	
	"""	
	sentences = [ 
		"diced tomato",
		"pork roast",
		"boneless skinless chicken breast half",
		"hard boiled egg",
		"hard cheese",
		"chopped ham"
	]
	"""
	
	for sentence in sentences:

		tagged = pos_tag(word_tokenize(sentence), tagset='universal')
				
		for row in tagged:
			if row[0] not in dictionary:
				dictionary[ row[0] ] = {}

			if row[1] not in dictionary[ row[0] ]:
				dictionary[row[0]][row[1]] = 1
			else:
				dictionary[row[0]][row[1]] += 1

	#print dictionary

	with io.open("dictionary.json" , encoding = 'utf8', mode = 'w') as _file:
		try:
			_data = json.dumps(dictionary, ensure_ascii=False, indent=4, sort_keys=True).decode('utf8')
			_file.write(_data)
		except TypeError: 
			_file.write(_data.decode('utf8'))
		except OverflowError: raise
		finally:
			_file.close()


if __name__=='__main__':
	main()
