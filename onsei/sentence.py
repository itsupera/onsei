import os
import unicodedata
import xml.etree.cElementTree as etree
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, List

import jaconv
from pyjuliusalign.jProcessingSnippet import cabocha, CabochaOutputError

CABOCHA_PATH = os.environ.get("CABOCHA_PATH", "/usr/local/bin/cabocha")
CABOCHA_ENCODING = os.environ.get("CABOCHA_ENCODING", "utf-8")


TextWithFurigana = namedtuple('TextWithFurigana', ['text', 'furigana'])


class Sentence:
    def __init__(self, raw_sentence: str):
        self.raw_sentence = raw_sentence.strip()

        # Call a wrapper of the CaboCha segmenter
        xml_str = cabocha(self.raw_sentence, CABOCHA_ENCODING, CABOCHA_PATH).encode('utf-8')
        try:
            sentence_element = etree.fromstring(xml_str)
        except etree.ParseError:
            raise CabochaOutputError(xml_str)

        self.words = parse_words_from_cabocha_output(sentence_element)

        self.julius_transcript = generate_julius_transcript_from_words(self.words)

    def to_html(self):
        return ' '.join((word.to_html() for word in self.words))


def parse_words_from_cabocha_output(sentence: etree.Element) -> List['Word']:
    words = []
    for chunk in sentence:
        for tok in chunk.findall('tok'):
            raw = tok.text
            features = tok.get("feature").split(',')
            pos_type = features[0]

            phonetics = ruby = None
            if pos_type != u"記号":
                phonetics = features[-1]
                # Some katakana words (e.g., パロメーター) have an "*" instead
                if phonetics == '*':
                    phonetics = raw
                if not all((is_katakana(_) for _ in phonetics)):
                    raise Exception(f"Not only katakanas in phonetic transcription: {phonetics}")

                katakanas = features[-2]
                ruby = extract_ruby(raw, katakanas)

            word = Word(raw, ruby, phonetics)
            words.append(word)
    return words


def generate_julius_transcript_from_words(words: List['Word']) -> str:
    katakanas = ''.join(((word.phonetics if word.phonetics else '') for word in words))
    return jaconv.hiragana2julius(jaconv.kata2hira(katakanas))


@dataclass
class Word:
    raw: str
    ruby: Optional[TextWithFurigana]
    phonetics: Optional[str]

    def to_html(self) -> str:
        if self.ruby:
            parts = []
            for text, furigana in self.ruby:
                if furigana:
                    parts.append(f"<rb>{text}<rb><rt>{furigana}</rt>")
                else:
                    parts.append(text)
            return f"<ruby>{''.join(parts)}</ruby>"
        else:
            return self.raw


def extract_ruby(raw: str, katakanas: str) -> Optional[List[TextWithFurigana]]:
    ruby = None
    if any(is_kanji(ch) for ch in raw):
        hiraganas = jaconv.kata2hira(katakanas)
        ruby = split_okurigana(raw, hiraganas)
    return ruby


def split_okurigana(text: str, hiragana: str) -> List[TextWithFurigana]:
    """
    Borrowed from https://github.com/mymro/furigana
    """
    split = []
    i = 0
    j = 0

    while i < len(text):
        start_i = i
        start_j = j

        # take care of hiragana only parts
        if is_kana(text[i]):
            while i < len(text) and j < len(hiragana) and is_kana(text[i]):
                i += 1
                j += 1

            split.append(TextWithFurigana(text[start_i:i], None))

            if i >= len(text):
                break

            start_i = i
            start_j = j

        # find next non kanji
        while i < len(text) and not is_kana(text[i]):
            i += 1

        # if there only kanji left
        if i >= len(text):
            split.append(TextWithFurigana(text[start_i:i], hiragana[start_j:len(hiragana)]))
            break

        # get reading of kanji
        # j-start_j < i - start_i every kanji has at least one sound associated with it
        while j < len(hiragana) and ((hiragana[j] != text[i] and jaconv.hira2kata(hiragana[j]) !=
                                      text[i]) or j - start_j < i - start_i):
            j += 1

        split.append(TextWithFurigana(text[start_i:i], hiragana[start_j:j]))

    return split


def is_kana(ch):
    return is_hiragana(ch) or is_katakana(ch)


def is_kanji(ch: str) -> bool:
    """ Borrowed from https://github.com/MikimotoH/furigana """
    return 'CJK UNIFIED IDEOGRAPH' in unicodedata.name(ch)


def is_hiragana(ch):
    """ Borrowed from https://github.com/MikimotoH/furigana """
    return 'HIRAGANA' in unicodedata.name(ch)


def is_katakana(ch):
    return 'KATAKANA' in unicodedata.name(ch)


def parsing_test():
    """ Simple parsing test on the JSUT sample data """
    with open('data/jsut_basic5000_sample/transcript_utf8.txt') as f:
        for line in f:
            raw = line.split(':')[1]
            s = Sentence(raw)
            print(s.to_html())


if __name__ == '__main__':
    parsing_test()
