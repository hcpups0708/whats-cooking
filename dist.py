
import json
import io
from sets import Set
"""
class baseWhatsCooking(object):
	def __init__(self, fn):
		self._fn = fn
		self._filterWords = Set()
		self._allIngredient = Set()
		self._allCuisineType = Set()
		self._filterWords_dirty = True
		self.rawData = None

		_load()

	""" PROTECTED METHOD (mechanism approach) """

	def _load(self):
		with io.open(self._fn, encoding = 'utf8') as data_file:
			self.rawData = json.load(data_file)

		assert self.rawData not None

	def _parseRaw_getAllIngredientList(self):
		for dish in self.rawData:
			_ingredients = dish['ingredients']
			_ingredients = map(self._ingredient_tokfilter, _ingredients)
			self._allIngredient |= Set(_ingredients)
	
	def _parseRaw_getAllCuisineList(self):
		for dish in self.rawData:
			self._allCuisineType = dish['cuisine']
	


	def _ingredient_tokfilter(self, name):
		tokens = name.split(' ')
		
		for fword in self._filterWords:
			tokens = filter(lambda x: x!=fword, tokens)

		joined = ' '.join(tokens)
		return joined

	""" PUBLIC METHOD (policy approach) """

	def addFilterWords(self, t):
		if type(t) is str:
			t = [t]
		if type(t) is list:
			t = Set(t)
		if type(t) is Set:
			self._filterWords |= t
			self._filterWords_dirty = True
			return
		# false occured
		assert not True

	def getIngredientList(self):
		if self._filterWord_dirty is True:
			_parseRaw_getAllIngredientList()
			self._filterWords_dirty = False

		assert self._filterWords_dirty is not True

		return list(self._allIngredient)

	def getCuisineList(self):
		return list(self._allCuisineType)

	def dump():
		pass

"""

filter_words = [
	'oz', 'lb', 'le', ''
]

def getIngredientList(fn, f_filter=None):
	""" Get list of ingredients available in test.json/train.json
	Args:
		fn (str): file location of test.json
		f_filter (func): function of filter of ingredient words
			def f_filter(params):
				Args: 
					str: string of ingredients 
				Returns:
					str: filtered string of ingredients
	Returns:
		list: result of test.json, filtered
	Examples:
		To 
		>>> getIngredientList('test_cleaned.json')
		["A Taste Thai Rice Noodles", "Accent Seasoning", ...]
		
		>>> def ingredient_name_process(in):
		>>>		return in.replace(' ', '_')
		>>> ingredientList = getIngredientList('test_cleaned.json', ingredient_name_process)
		["A_Taste_Thai_Rice_Noodles", "Accent_Seasoning", ...]
	"""
	assert type(fn) is str
	assert hasattr(ing_filter, '__call__')

	data = None
	total_ingredient_test = Set()

	with io.open(fn, encoding = 'utf8') as data_file:
		data = json.load(data_file)

	assert data is not None

	for dish in data:
		_ingredients = dish['ingredients']
		_ingredients = map(lambda x: f_filter(x), _ingredients)
		total_ingredient_set |= Set(_ingredients)

	return list(total_ingredient_test)

def ingredientAsKey(fn, f_filter=None, l_filter:None):
	""" Get list of ingredients available in train.json
	Args:
		fn (str): file location of test.json
		f_filter (func): function of filter, refer to @getIngredientList(fn,f_filter)
		l_filter (func): function of filter of ingredient array
			def l_filter(params):
				Args: 
					list: ingredients 
				Returns:
					list: filtered ingredients list
	Returns:
		dict: ingredient_cuisine_distribution
	Examples:
		>>> ingredientAsKey('train_cleaned.json')

	"""
	assert type(fn) is str
	assert hasattr(ffilter, '__call__')

	data = None
	ingredient_cuisine_dist={}

	with io.open(fn, encoding = 'utf8') as data_file:
		data = json.load(data_file)

	assert data is not None and type(data) is list

	for dish in data:
		assert dish['cuisine'] is not None and dish['ingredients'] is not None

		ingredients = dish['ingredients']
		for ing_name in ingredients:
			ing_name = f_filter(ing_name)
			
			if ing_name in ingredient_cuisine_dist:
				if cuisine_type in ingredient_cuisine_dist[ing_name]:
					ingredient_cuisine_dist[ing_name][cuisine_type] += 1
				else :
					ingredient_cuisine_dist[ing_name][cuisine_type] = 1
			else:
				ingredient_cuisine_dist[ing_name] = {}
				ingredient_cuisine_dist[ing_name][cuisine_type] = 1

	return ingredient_cuisine_dist

def ingredient_name_process(name):
	global filter_words

	tokens = name.split(' ')
	
	for fword in filter_words:
		tokens = filter(lambda x: x!=fword, tokens)

	joined = ' '.join(tokens)
	return joined

def main():
	testIngredientList = getTestIngredientList('test_cleaned.json', ingredient_name_process)
	
	ingredient_cuisine_dist={}
	total_ingredient_freq = {}
	total_cuisine_freq = {}

	with io.open('train_cleaned.json', encoding = 'utf8') as data_file:
		data = json.load(data_file)

	for dish in data:
		cuisine_type = dish['cuisine']
		if cuisine_type in total_cuisine_freq:
			total_cuisine_freq[cuisine_type] += 1
		else:
			total_cuisine_freq[cuisine_type] = 1

		ingredients = dish['ingredients']
		for ing_name in ingredients:
			ing_name = ingredient_name_process(ing_name)
			if ing_name in total_ingredient_freq:
				total_ingredient_freq[ing_name] += 1
			else:
				total_ingredient_freq[ing_name] = 1

			if ing_name in ingredient_cuisine_dist:
				if cuisine_type in ingredient_cuisine_dist[ing_name]:
					ingredient_cuisine_dist[ing_name][cuisine_type] += 1
				else :
					ingredient_cuisine_dist[ing_name][cuisine_type] = 1
			else:
				ingredient_cuisine_dist[ing_name] = {}
				ingredient_cuisine_dist[ing_name][cuisine_type] = 1

	#print json.dumps(ingredient_cuisine_dist, indent=4, sort_keys=True)
	#print "length of total ingredients", len(total_ingredient_freq)
	print json.dumps(total_cuisine_freq, indent=4, sort_keys=True)


if __name__=='__main__':
	main()
