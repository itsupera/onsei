import os
import unicodedata
import xml.etree.cElementTree as etree
from dataclasses import dataclass
from typing import Optional, List

import jaconv
from pyjuliusalign.jProcessingSnippet import cabocha, CabochaOutputError

CABOCHA_PATH = os.environ.get("CABOCHA_PATH", "/usr/local/bin/cabocha")
CABOCHA_ENCODING = os.environ.get("CABOCHA_ENCODING", "utf-8")


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
        return ''.join((word.to_html() for word in self.words))


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
    ruby: Optional[str]
    phonetics: Optional[str]

    def to_html(self) -> str:
        if self.ruby:
            parts = []
            for _tuple in self.ruby:
                if len(_tuple) == 1:
                    parts.append(_tuple[0])
                else:
                    rb, rt = _tuple
                    parts.append(f"<rb>{rb}<rb><rt>{rt}</rt>")
            return f"<ruby>{''.join(parts)}</ruby>"
        else:
            return self.raw


def extract_ruby(raw: str, katakanas: str) -> Optional[List[tuple]]:
    ruby = None
    if any(is_kanji(ch) for ch in raw):
        hiraganas = jaconv.kata2hira(katakanas)
        ruby = []
        for pair in split_okurigana(raw, hiraganas):
            ruby.append(pair)
    return ruby


def split_okurigana(text, hiragana):
    """
    Borrowed from https://github.com/MikimotoH/furigana
    送り仮名 processing
    tested:
      * 出会(であ)う
      * 明(あか)るい
      * 駆(か)け抜(ぬ)け
    """
    if is_hiragana(text[0]):
        yield from split_okurigana_reverse(text, hiragana)
    if all(is_kanji(_) for _ in text):
        yield text, hiragana
        return
    text = list(text)
    ret = (text[0], [hiragana[0]])
    for hira in hiragana[1:]:
        for char in text:
            if hira == char:
                text.pop(0)
                if ret[0]:
                    if is_kanji(ret[0]):
                        yield ret[0], ''.join(ret[1][:-1])
                        yield (ret[1][-1],)
                    else:
                        yield (ret[0],)
                else:
                    yield (hira,)
                ret = ('', [])
                if text and text[0] == hira:
                    text.pop(0)
                break
            else:
                if is_kanji(char):
                    if ret[1] and hira == ret[1][-1]:
                        text.pop(0)
                        yield ret[0], ''.join(ret[1][:-1])
                        yield char, hira
                        ret = ('', [])
                        text.pop(0)
                    else:
                        ret = (char, ret[1]+[hira])
                else:
                    # char is also hiragana
                    if hira != char:
                        break
                    else:
                        break


def split_okurigana_reverse(text, hiragana):
    """
    Borrowed from https://github.com/MikimotoH/furigana

    Tested:
      お茶(おちゃ)
      ご無沙汰(ごぶさた)
      お子(こ)さん
    """
    yield (text[0],)
    yield from split_okurigana(text[1:], hiragana[1:])


def is_kanji(ch: str) -> bool:
    """ Borrowed from https://github.com/MikimotoH/furigana """
    return 'CJK UNIFIED IDEOGRAPH' in unicodedata.name(ch)


def is_hiragana(ch):
    """ Borrowed from https://github.com/MikimotoH/furigana """
    return 'HIRAGANA' in unicodedata.name(ch)


def is_katakana(ch):
    return 'KATAKANA' in unicodedata.name(ch)


if __name__ == '__main__':
    # Simple parsing test on the JSUT sample data
    with open('data/jsut_basic5000_sample/transcript_utf8.txt') as f:
        for line in f:
            raw = line.split(':')[1]
            s = Sentence(raw)
            print(s.to_html())
