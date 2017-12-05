#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains methods for parsing and preprocessing strings. Let's consider the most noticeable:

* :func:`~gensim.parsing.preprocessing.remove_stopwords` - remove all stopwords from string
* :func:`~gensim.parsing.preprocessing.preprocess_string` -  preprocess string (in default NLP meaning)

Examples:
---------
>>> from gensim.parsing.preprocessing import remove_stopwords
>>> remove_stopwords("Better late than never, but better never late.")
u'Better late never, better late.'
>>>
>>> preprocess_string("<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?")
[u'hel', u'rld', u'weather', u'todai', u'isn']


Data:
-----

.. data:: STOPWORDS - Set of stopwords from Stone, Denis, Kwantes (2010).
.. data:: RE_PUNCT - Regexp for search an punctuation.
.. data:: RE_TAGS - Regexp for search an tags.
.. data:: RE_NUMERIC - Regexp for search an numbers.
.. data:: RE_NONALPHA - Regexp for search an non-alphabetic character.
.. data:: RE_AL_NUM - Regexp for search a position between letters and digits.
.. data:: RE_NUM_AL - Regexp for search a position between digits and letters .
.. data:: RE_WHITESPACE - Regexp for search space characters.
.. data:: DEFAULT_FILTERS - List of function for string preprocessing.

"""

import re
import string
import glob

from gensim import utils
from gensim.parsing.porter import PorterStemmer
from nltk.stem.snowball import SpanishStemmer


'sale','digo','miro','tarde','saludo','dejan','dado','quería','necesitaría','decir','día','hacerlo','hace','muchas','pedimos','ido','genial','preguntar','quedo','pasa','días','tardes','buenas','necesito','buenos','hola','gracias','quieres','quiero','de','la','que','el','en','y','a','los','del','se','las','por','un','para','con','no','una','su','al','lo','como','más','pero','sus','le','ya','o','este','sí','porque','esta','entre','cuando','muy','sin','sobre','también','me','hasta','ay','donde','quien','desde','todo','nos','durante','todos','uno','les','ni','contra','otros','ese','eso','ante','ellos','e','esto','mí','antes','algunos','qué','unos','yo','otro','otras','otra','él','tanto','esa','estos','mucho','quienes','nada','muchos','cual','poco','ella','estar','estas','algunas','algo','nosotros','mi','mis','tú','te','ti','tu','tus','ellas','nosotras','vosotros','vosotras','os','mío','mía','míos','mías','tuyo','tuya','tuyos','tuyas','suyo','suya','suyos','suyas','nuestro','nuestra','nuestros','nuestras','vuestro','vuestra','vuestros','vuestras','esos','esas','estoy','estás','está','estamos','estáis','están','esté','estés','estemos','estéis','estén','estaré','estarás','estará','estaremos','estaréis','estarán','estaría','estarías','estaríamos','estaríais','estarían','estaba','estabas','estábamos','estabais','estaban','estuve','estuviste','estuvo','estuvimos','estuvisteis','estuvieron','estuviera','estuvieras','estuviéramos','estuvierais','estuvieran','estuviese','estuvieses','estuviésemos','estuvieseis','estuviesen','estando','estado','estada','estados','estadas','estad','he','has','ha','hemos','habéis','han','haya','hayas','hayamos','hayáis','hayan','habré','habrás','habrá','habremos','habréis','habrán','habría','habrías','habríamos','habríais','habrían','había','habías','habíamos','habíais','habían','hube','hubiste','hubo','hubimos','hubisteis','hubieron','hubiera','hubieras','hubiéramos','hubierais','hubieran','hubiese','hubieses','hubiésemos','hubieseis','hubiesen','habiendo','habido','habida','habidos','habidas','soy','eres','es','somos','sois','son','sea','seas','seamos','seáis','sean','seré','serás','será','seremos','seréis','serán','sería','serías','seríamos','seríais','serían','era','eras','éramos','erais','eran','fui','fuiste','fue','fuimos','fuisteis','fueron','fuera','fueras','fuéramos','fuerais','fueran','fuese','fueses','fuésemos','fueseis','fuesen','siendo','sido','tengo','tienes','tiene','tenemos','tenéis','tienen','tenga','tengas','tengamos','tengáis','tengan','tendré','tendrás','tendrá','tendremos','tendréis','tendrán','tendría','tendrías','tendríamos','tendríais','tendrían','tenía','tenías','teníamos','teníais','tenían','tuve','tuviste','tuvo','tuvimos','tuvisteis','tuvieron','tuviera','tuvieras','tuviéramos','tuvierais','tuvieran','tuviese','tuvieses','tuviésemos','tuvieseis','tuviesen','teniendo','tenido','tenida','tenidos','tenidas','tened','ahí','ajena','ajeno','ajenas','ajenos','algúna','allá','ambos','aquello','aquellas','aquellos','así','atrás','aun','aunque','bajo','bastante','bien','cabe','cada','casi','cierto','cierta','ciertos','ciertas','conmigo','conseguimos','conseguir','consigo','consigue','consiguen','consigues','cualquier','cualquiera','cualquieras','cuan','cuanto','cuanta','cuantas','cuantos','de','dejar','demás','demasiadas','demasiados','dentro','dos','ello','emplean','emplear','empleas','encima','entonces','era','eras','eramos','eses','estes','gueno','hacer','hacemos','hacia','hago','incluso','intenta','intentas','intentamos','intentan','intento','ir','mismo','ningúno','nunca','parecer','podemos','podría','podrías','podríais','podríamos','podrían','primero','puedes','pueden','pues','querer','quiénes','quienesquiera','quienquiera','quizás','sabe','sabes','saben','sabéis','sabemos','saber','sino','solo','esta','tampoco','tan','tanta','tantas','tantos','tener','tiempo','toda','todas','tomar','trabaja','trabajas','tras','último','ultimo','última','ultima','unas','ustedes','variasos','verdadera','pocas','pocos','podéis','podemos','poder','podría','podrías','podríais','podríamos','podrían','primero','puede','puedo','pueda','pues','querer','quiénes','quienesquiera','quienquiera','quizás','mas','sabe','sabes','saben','sabéis','sabemos','saber','según','ser','si','siempre','sino','so','solamente','solo','sólo','sr','sra','sres','sta','tal','tales','tampoco','tan','tanta','tantas','tantos','tener','tiempo','toda','den','queria','todas','tomar','trabaja','trabajo','trabajáis','trabajamos','trabajan','trabajar','trabajas','tras','último','ultimo','unas','usa','usas','usáis','usamos','usan','usar','uso','usted','ustedes','va','van','vais','valor','vamos','varias','varias','varios','vaya','verdadera','voy','vez','más','ok']
STOPWORDS = frozenset(['vuelva','realizar','vimos','semana','pasada','luego','dices','k','poner','hablamos','favor',
)


RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
RE_NONALPHA = re.compile(r"\W", re.UNICODE)
RE_AL_NUM = re.compile(r"([a-z]+)([0-9]+)", flags=re.UNICODE)
RE_NUM_AL = re.compile(r"([0-9]+)([a-z]+)", flags=re.UNICODE)
RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)


def remove_stopwords(s):
    """Remove :const:`~gensim.parsing.preprocessing.STOPWORDS` from `s`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without :const:`~gensim.parsing.preprocessing.STOPWORDS`.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import remove_stopwords
    >>> remove_stopwords("Better late than never, but better never late.")
    u'Better late never, better late.'

    """
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w not in STOPWORDS)


def strip_punctuation(s):
    """Replace punctuation characters with spaces in `s` using :const:`~gensim.parsing.preprocessing.RE_PUNCT`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without punctuation characters.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_punctuation
    >>> strip_punctuation("A semicolon is a stronger break than a comma, but not as much as a full stop!")
    u'A semicolon is a stronger break than a comma  but not as much as a full stop '

    """
    s = utils.to_unicode(s)
    return RE_PUNCT.sub(" ", s)


strip_punctuation2 = strip_punctuation


def strip_tags(s):
    """Remove tags from `s` using :const:`~gensim.parsing.preprocessing.RE_TAGS`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without tags.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_tags
    >>> strip_tags("<i>Hello</i> <b>World</b>!")
    u'Hello World!'

    """
    s = utils.to_unicode(s)
    return RE_TAGS.sub("", s)


def strip_short(s, minsize=3):
    """Remove words with length lesser than `minsize` from `s`.

    Parameters
    ----------
    s : str
    minsize : int, optional

    Returns
    -------
    str
        Unicode string without short words.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_short
    >>> strip_short("salut les amis du 59")
    u'salut les amis'
    >>>
    >>> strip_short("one two three four five six seven eight nine ten", minsize=5)
    u'three seven eight'

    """
    s = utils.to_unicode(s)
    return " ".join(e for e in s.split() if len(e) >= minsize)


def strip_numeric(s):
    """Remove digits from `s` using :const:`~gensim.parsing.preprocessing.RE_NUMERIC`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode  string without digits.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_numeric
    >>> strip_numeric("0text24gensim365test")
    u'textgensimtest'

    """
    s = utils.to_unicode(s)
    return RE_NUMERIC.sub("", s)


def strip_non_alphanum(s):
    """Remove non-alphabetic characters from `s` using :const:`~gensim.parsing.preprocessing.RE_NONALPHA`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string with alphabetic characters only.

    Notes
    -----
    Word characters - alphanumeric & underscore.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_non_alphanum
    >>> strip_non_alphanum("if-you#can%read$this&then@this#method^works")
    u'if you can read this then this method works'

    """
    s = utils.to_unicode(s)
    return RE_NONALPHA.sub(" ", s)


def strip_multiple_whitespaces(s):
    r"""Remove repeating whitespace characters (spaces, tabs, line breaks) from `s`
    and turns tabs & line breaks into spaces using :const:`~gensim.parsing.preprocessing.RE_WHITESPACE`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without repeating in a row whitespace characters.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import strip_multiple_whitespaces
    >>> strip_multiple_whitespaces("salut" + '\r' + " les" + '\n' + "         loulous!")
    u'salut les loulous!'

    """
    s = utils.to_unicode(s)
    return RE_WHITESPACE.sub(" ", s)


def split_alphanum(s):
    """Add spaces between digits & letters in `s` using :const:`~gensim.parsing.preprocessing.RE_AL_NUM`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string with spaces between digits & letters.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import split_alphanum
    >>> split_alphanum("24.0hours7 days365 a1b2c3")
    u'24.0 hours 7 days 365 a 1 b 2 c 3'

    """
    s = utils.to_unicode(s)
    s = RE_AL_NUM.sub(r"\1 \2", s)
    return RE_NUM_AL.sub(r"\1 \2", s)


def stem_text(text):
    """Transform `s` into lowercase and stem it.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Unicode lowercased and porter-stemmed version of string `text`.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import stem_text
    >>> stem_text("While it is quite useful to be able to search a large collection of documents almost instantly.")
    u'while it is quit us to be abl to search a larg collect of document almost instantly.'

    """
    text = utils.to_unicode(text)
    # p = PorterStemmer() For english
    p = SpanishStemmer()

    return ' '.join(p.stem(word) for word in text.split())


stem = stem_text


DEFAULT_FILTERS = [
    lambda x: x.lower(), strip_tags, strip_punctuation,
    strip_multiple_whitespaces, strip_numeric,
    remove_stopwords, strip_short, stem_text
]


def preprocess_string(s, filters=DEFAULT_FILTERS):
    """Apply list of chosen filters to `s`.

    Default list of filters:

    * :func:`~gensim.parsing.preprocessing.strip_tags`,
    * :func:`~gensim.parsing.preprocessing.strip_punctuation`,
    * :func:`~gensim.parsing.preprocessing.strip_multiple_whitespaces`,
    * :func:`~gensim.parsing.preprocessing.strip_numeric`,
    * :func:`~gensim.parsing.preprocessing.remove_stopwords`,
    * :func:`~gensim.parsing.preprocessing.strip_short`,
    * :func:`~gensim.parsing.preprocessing.stem_text`.

    Parameters
    ----------
    s : str
    filters: list of functions, optional

    Returns
    -------
    list of str
        Processed strings (cleaned).

    Examples
    --------
    >>> from gensim.parsing.preprocessing import preprocess_string
    >>> preprocess_string("<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?")
    [u'hel', u'rld', u'weather', u'todai', u'isn']
    >>>
    >>> s = "<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?"
    >>> CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation]
    >>> preprocess_string(s, CUSTOM_FILTERS)
    [u'hel', u'9lo', u'wo9', u'rld', u'th3', u'weather', u'is', u'really', u'g00d', u'today', u'isn', u't', u'it']

    """
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s.split()


def preprocess_documents(docs):
    """Apply :const:`~gensim.parsing.preprocessing.DEFAULT_FILTERS` to the documents strings.

    Parameters
    ----------
    docs : list of str

    Returns
    -------
    list of (list of str)
        Processed documents split by whitespace.

    Examples
    --------
    >>> from gensim.parsing.preprocessing import preprocess_documents
    >>> preprocess_documents(["<i>Hel 9lo</i> <b>Wo9 rld</b>!", "Th3     weather_is really g00d today, isn't it?"])
    [[u'hel', u'rld'], [u'weather', u'todai', u'isn']]

    """
    return [preprocess_string(d) for d in docs]


def read_file(path):
    with utils.smart_open(path) as fin:
        return fin.read()


def read_files(pattern):
    return [read_file(fname) for fname in glob.glob(pattern)]
