
import json
import nltk
import io
from time import sleep
from nltk import pos_tag, word_tokenize
from nltk.corpus import brown
from threading import Thread
from Queue import Queue, Empty

NUM_THREAD = 16 

def thrower(queue, input_array, id):
	one_unit = len(input_array) /NUM_THREAD
	startpoint = id * one_unit
	endpoint = (id+1)*one_unit 
	endpoint = endpoint if endpoint<len(input_array) else len(input_array)

	for i in range(startpoint, endpoint):
		sentence = input_array[i]
		tagged = pos_tag(word_tokenize(sentence), tagset='universal')

		#print tagged_text
		queue.put(tagged)

        queue.put(1)


def main():
	dictionary = {}
	sentences = []
	result_q = Queue()
        
        
	with io.open("out/ingredient_list_train_cleaned.json", encoding = 'utf8') as data_file:
		sentences = json.load(data_file)
		data_file.close()
		#print "Data Fetched."
	
	"""	
	sentences = [ 
		"diced tomato",
		"pork roast",
		"boneless skinless chicken breast half",
		"hard boiled egg",
		"hard cheese",
		"chopped ham",
		"iced lemon tea"
	]
        """
	threads = map(lambda index: Thread(target=thrower, args=(result_q, sentences, index)), range(NUM_THREAD))
	
	map(lambda th: th.start(), threads)
        
        die_num = 0

	while(True):
                try:
			tagged = result_q.get_nowait()
			#print tagged
		except Empty:
			#die = map(lambda th: not th.is_alive(), threads)
			#all_die = reduce(lambda p,n: p&n, die)
			#if all_die:
			#	break
			#else:
			sleep(1)
                        continue

                if type(tagged) is int and tagged is 1:
                    die_num+=1
                    if die_num is NUM_THREAD:
                        break

		else:
                    for row in tagged:
			if row[0] not in dictionary:
				dictionary[row[0]] = {}

			if row[1] not in dictionary[row[0]]:
				dictionary[row[0]][row[1]] = 1
			else:
				dictionary[row[0]][row[1]] += 1


	print json.dumps(dictionary, ensure_ascii=False, indent=4, sort_keys=True)
	
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
