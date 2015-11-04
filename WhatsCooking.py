
import json
import time
import cStringIO
import re, os, io
from sets import Set

class Raw(object):
	def __init__(self, fn):
		self._fn = fn
		self.rawData = None
		self.outputPrefix = './'
		self._load()

	""" PROTECTED METHOD (mechanism approach) """

	def _load(self):
		with io.open(self._fn, encoding = 'utf8') as data_file:
			self.rawData = json.load(data_file)
			data_file.close()

		assert self.rawData is not None

	""" PUBLIC METHOD (policy approach) """

	def addFilterIngredients(self, ingredients):
		pass

	def getIngredientFilteredRaw(self, filter):
		pass

	def setOutputPrefix(self, prefix):
		if re.match('.*\/$', prefix):
			self.outputPrefix = prefix
		else:
			self.outputPrefix = prefix+'/'

	def dumpToFile(self, obj, extra_string = ''):
		assert type(extra_string) is str
		
		if not os.path.exists(self.outputPrefix):
			os.makedirs(self.outputPrefix)

		with io.open(self.outputPrefix+extra_string+'_'+self._fn , encoding = 'utf8', mode = 'w') as _file:
			try:
				_data = json.dumps(obj, ensure_ascii=False, indent=4, sort_keys=True).decode('utf8')
				_file.write(_data)
			except TypeError: 
				_file.write(_data.decode('utf8'))
			except OverflowError: raise
			finally:
				_file.close()


class Test(Raw):
	def __init__(self, fn):
		#self._fn = fn
		self._filterWords = Set()
		self._filterWords_dirty = True

		Raw.__init__(self, fn)
		#self.rawData = None

		#self._load()

	""" PROTECTED METHOD (mechanism approach) """

	def _parseRaw_getAllIngredientList(self):
		for dish in self.rawData:
			_ingredients = dish['ingredients']
			_ingredients = map(self._ingredient_tokfilter, _ingredients)
			self._allIngredient |= Set(_ingredients)

	def _parseRaw_getIngredientFrequent(self):
		for dish in self.rawData:
			_ingredients = dish['ingredients']
			_ingredients = map(self._ingredient_tokfilter, _ingredients)
			for ing_name in _ingredients:
				if ing_name in self._ingredientFrequent:
					self._ingredientFrequent[ing_name] += 1
				else :
					self._ingredientFrequent[ing_name] = 1

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
		if not hasattr(self, '_allIngredient'):
			self._allIngredient = Set()
			self._filterWords_dirty = True

		if self._filterWords_dirty is True:
			self._parseRaw_getAllIngredientList()
			self._filterWords_dirty = False

		assert self._filterWords_dirty is False

		return list(self._allIngredient)

	def getIngredientFrequent(self):
		if not hasattr(self, '_ingredientFrequent'):
			self._ingredientFrequent = {}
			self._filterWords_dirty = True

		if self._filterWords_dirty is True:
			self._parseRaw_getIngredientFrequent()
			self._filterWords_dirty = False

		return self._ingredientFrequent


class Train(Test):
	def __init__(self,fn):
		Test.__init__(self, fn)
		
	""" PROTECTED METHOD (mechanism approach) """

	def _parseRaw_getAllCuisineList(self):
		for dish in self.rawData:
			self._allCuisineType.add(dish['cuisine'])

	def _parseRaw_getIngredientAsKey(self):
		for dish in self.rawData:
			assert dish['cuisine'] is not None and dish['ingredients'] is not None

			cuisine_type = dish['cuisine']
			ingredients = dish['ingredients']
			for ing_name in ingredients:
				ing_name = self._ingredient_tokfilter(ing_name)
				
				if ing_name in self._ingredientAsKey:
					if cuisine_type in self._ingredientAsKey[ing_name]:
						self._ingredientAsKey[ing_name][cuisine_type] += 1
					else :
						self._ingredientAsKey[ing_name][cuisine_type] = 1
				else:
					self._ingredientAsKey[ing_name] = {}
					self._ingredientAsKey[ing_name][cuisine_type] = 1

	""" PUBLIC METHOD (policy approach) """

	def getCuisineList(self):
		if not hasattr(self, '_allCuisineType'):
			self._allCuisineType = Set()
			self._parseRaw_getAllCuisineList()

		return list(self._allCuisineType)

	def getIngredientAsKey(self):
		if not hasattr(self, '_ingredientAsKey'):
			self._ingredientAsKey = {}
			self._filterWords_dirty = True

		if self._filterWords_dirty is True:
			self._parseRaw_getIngredientAsKey()
			self._filterWords_dirty = False

		return self._ingredientAsKey
