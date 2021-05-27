import unittest

from lib.modernisation.syllable_tokenizer import SyllableTokenizer


class TestSyllableTokenizer(unittest.TestCase):
    def test_syllable_tokenizer(self):
        pass


words = """
partyen
inschryvingen inschrijvingen startje starten startten opschrijf
ijlen lijken rijen gejat gokje pitten perziken kauwen eeuwen
alinea pagina waarom computermarkt tweedehandsautomarkt wandelen vrijetijdskleding
moeilijkheeden geteekent agtbare
extra zinnetje erbij verifieert zekerheidshalve randgevallen
nog een zin met korte woordjes om te testen of dat sneller gaat
"""[1:-1].split()

ref = """
par.ty.en in.schry.vin.gen in.schrij.vin.gen start.je star.ten start.ten op.schrijf
ij.len lij.ken rij.en ge.jat gok.je pit.ten per.zi.ken kauw.en eeuw.en
a.li.nea pa.gi.na waa.rom com.pu.ter.markt twee.de.hand.sau.to.markt wan.de.len vrij.e.tijds.kle.ding
moei.lijk.hee.den ge.tee.kent agt.ba.re
ex.tra zin.net.je er.bij ve.ri.fieert ze.ker.heid.shal.ve rand.ge.val.len
nog een zin met kor.te woord.jes om te tes.ten of dat snel.ler gaat
"""[1:-1].split()

st = SyllableTokenizer()
for _ in range(1000):
    tokenized = [st.encode(word) for word in words]

assert ['.'.join(sylls) for sylls in tokenized] == ref
