
import urllib2
import json
import re
import sys
import cgi
from bs4 import BeautifulSoup as bsoup

AGENT = "%s/1.0" % __name__

obj = 'wiki butternut squash'.replace(' ', '+')
url = 'https://www.google.com.tw/search?q='+obj

wikiFilter = re.compile('^\/url[?]q=(https?:\/\/[a-z]{2,3}\.wikipedia\.org\/wiki\/\S+?)[%&]')

content,req, handle = (None, None, None)

try:
	req = urllib2.Request(url)
	req.add_header("User-Agent", AGENT)
	handle = urllib2.build_opener()
except IOError, e:
	print "ERROR: %s" % e

if handle:
	try:
		content = unicode(handle.open(req).read(), "utf-8", errors="replace")
	except urllib2.HTTPError, e:
		print "ERROR: %s" % e
	except urllib2.URLError, e:
		print "ERROR: %s" % e

	if content:
		soup = bsoup(content, 'html.parser')
		all_a = soup.find_all('a')
		urls = []
		for a in all_a:
			#print a['href']
                        urls = wikiFilter.findall(a['href'])
                        if urls.counts is not 0:

                        
