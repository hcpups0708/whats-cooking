
from WhatsCooking import Test, Train

def main():
	filter_words = [
		'oz', 'lb', 'le', ''
	]

	test = testWhatsCooking('test_cleaned.json')
	test.addFilterWords(filter_words)
	#print test.getIngredientList()

	train = trainWhatsCooking('train_cleaned.json')
	train.addFilterWords(filter_words)
	#print train.getIngredientList()
	#print train.getCuisineList()
	#print train.getIngredientAsKey()

if __name__=='__main__':
	main()
