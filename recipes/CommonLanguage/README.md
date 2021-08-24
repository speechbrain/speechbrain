# VoxLingua107 Dataset [[download]](http://bark.phon.ioc.ee/voxlingua107/)

VoxLingua107 is a speech dataset for training spoken language identification models. 
The dataset consists of short speech segments automatically extracted from YouTube videos and labeled according the language of the video title and description, with some post-processing steps to filter out false positives.

VoxLingua107 contains data for 107 languages. The total amount of speech in the training set is 6628 hours. 
The average amount of data per language is 62 hours. However, the real amount per language varies a lot. There is also a seperate development set containing 1609 speech segments from 33 languages, validated by at least two volunteers to really contain the given language.

For more information, see the paper [J&ouml;rgen Valk, Tanel Alum&auml;e. _VoxLingua107: a Dataset for Spoken Language Recognition_. Proc. SLT 2021](https://arxiv.org/abs/2011.12998).

## Collection method

We extracted audio data from YouTube videos that are retrieved using language-specific search phrases (random phrases from Wikipedia of the particular language).
If the language of the video title and description matched with the language of the search phrase, 
the audio in the video was deemed likely to be in that particular language. This allowed to collect large amounts of somewhat noisy data relatively cheaply.
Speech/non-speech detection and speaker diarization was used to segment the videos into short sentence-like utterances.
A data-driven post-filtering step was applied to remove clips that were very different from other clips in this language's dataset, and thus likely not in the given language.
Due to the automatic data collection process, there are still clips in the dataset that are not in the given language or contain non-speech  (around 2% overall) 



## Statistics of VoxLingua107:

Training data per language:

- [Abkhazian](ab.zip) (10 hours, 980M), [sample](samples/ab.wav) 
- [Afrikaans](af.zip) (108 hours, 10G), [sample](samples/af.wav) 
- [Amharic](am.zip) (81 hours, 7.7G), [sample](samples/am.wav) 
- [Arabic](ar.zip) (59 hours, 5.5G), [sample](samples/ar.wav) 
- [Assamese](as.zip) (155 hours, 15G), [sample](samples/as.wav) 
- [Azerbaijani](az.zip) (58 hours, 5.6G), [sample](samples/az.wav) 
- [Bashkir](ba.zip) (58 hours, 5.5G), [sample](samples/ba.wav) 
- [Belarusian](be.zip) (133 hours, 13G), [sample](samples/be.wav) 
- [Bulgarian](bg.zip) (50 hours, 4.7G), [sample](samples/bg.wav) 
- [Bengali](bn.zip) (55 hours, 5.4G), [sample](samples/bn.wav) 
- [Tibetan](bo.zip) (101 hours, 9.3G), [sample](samples/bo.wav) 
- [Breton](br.zip) (44 hours, 4.2G), [sample](samples/br.wav) 
- [Bosnian](bs.zip) (105 hours, 9.7G), [sample](samples/bs.wav) 
- [Catalan](ca.zip) (88 hours, 8.1G), [sample](samples/ca.wav) 
- [Cebuano](ceb.zip) (6 hours, 589M), [sample](samples/ceb.wav) 
- [Czech](cs.zip) (67 hours, 6.3G), [sample](samples/cs.wav) 
- [Welsh](cy.zip) (76 hours, 6.6G), [sample](samples/cy.wav) 
- [Danish](da.zip) (28 hours, 2.6G), [sample](samples/da.wav) 
- [German](de.zip) (39 hours, 3.7G), [sample](samples/de.wav) 
- [Greek](el.zip) (66 hours, 6.2G), [sample](samples/el.wav) 
- [English](en.zip) (49 hours, 4.6G), [sample](samples/en.wav) 
- [Esperanto](eo.zip) (10 hours, 916M), [sample](samples/eo.wav) 
- [Spanish](es.zip) (39 hours, 3.7G), [sample](samples/es.wav) 
- [Estonian](et.zip) (38 hours, 3.5G), [sample](samples/et.wav) 
- [Basque](eu.zip) (29 hours, 2.8G), [sample](samples/eu.wav) 
- [Persian](fa.zip) (56 hours, 5.2G), [sample](samples/fa.wav) 
- [Finnish](fi.zip) (33 hours, 3.1G), [sample](samples/fi.wav) 
- [Faroese](fo.zip) (67 hours, 6.0G), [sample](samples/fo.wav) 
- [French](fr.zip) (67 hours, 6.2G), [sample](samples/fr.wav) 
- [Galician](gl.zip) (72 hours, 6.7G), [sample](samples/gl.wav) 
- [Guarani](gn.zip) (2 hours, 250M), [sample](samples/gn.wav) 
- [Gujarati](gu.zip) (46 hours, 4.5G), [sample](samples/gu.wav) 
- [Manx](gv.zip) (4 hours, 374M), [sample](samples/gv.wav) 
- [Hausa](ha.zip) (106 hours, 10G), [sample](samples/ha.wav) 
- [Hawaiian](haw.zip) (12 hours, 1.2G), [sample](samples/haw.wav) 
- [Hindi](hi.zip) (81 hours, 7.7G), [sample](samples/hi.wav) 
- [Croatian](hr.zip) (118 hours, 11G), [sample](samples/hr.wav) 
- [Haitian](ht.zip) (96 hours, 9.2G), [sample](samples/ht.wav) 
- [Hungarian](hu.zip) (73 hours, 6.9G), [sample](samples/hu.wav) 
- [Armenian](hy.zip) (69 hours, 6.6G), [sample](samples/hy.wav) 
- [Interlingua](ia.zip) (3 hours, 241M), [sample](samples/ia.wav) 
- [Indonesian](id.zip) (40 hours, 3.8G), [sample](samples/id.wav) 
- [Icelandic](is.zip) (92 hours, 8.4G), [sample](samples/is.wav) 
- [Italian](it.zip) (51 hours, 4.8G), [sample](samples/it.wav) 
- [Hebrew](iw.zip) (96 hours, 8.9G), [sample](samples/iw.wav) 
- [Japanese](ja.zip) (56 hours, 5.1G), [sample](samples/ja.wav) 
- [Javanese](jw.zip) (53 hours, 5.0G), [sample](samples/jw.wav) 
- [Georgian](ka.zip) (98 hours, 9.2G), [sample](samples/ka.wav) 
- [Kazakh](kk.zip) (78 hours, 7.3G), [sample](samples/kk.wav) 
- [Central Khmer](km.zip) (41 hours, 4.0G), [sample](samples/km.wav) 
- [Kannada](kn.zip) (46 hours, 4.4G), [sample](samples/kn.wav) 
- [Korean](ko.zip) (77 hours, 7.1G), [sample](samples/ko.wav) 
- [Latin](la.zip) (67 hours, 6.0G), [sample](samples/la.wav) 
- [Luxembourgish](lb.zip) (75 hours, 7.1G), [sample](samples/lb.wav) 
- [Lingala](ln.zip) (90 hours, 8.7G), [sample](samples/ln.wav) 
- [Lao](lo.zip) (42 hours, 4.0G), [sample](samples/lo.wav) 
- [Lithuanian](lt.zip) (82 hours, 7.7G), [sample](samples/lt.wav) 
- [Latvian](lv.zip) (42 hours, 4.0G), [sample](samples/lv.wav) 
- [Malagasy](mg.zip) (109 hours, 11G), [sample](samples/mg.wav) 
- [Maori](mi.zip) (34 hours, 3.2G), [sample](samples/mi.wav) 
- [Macedonian](mk.zip) (112 hours, 11G), [sample](samples/mk.wav) 
- [Malayalam](ml.zip) (47 hours, 4.6G), [sample](samples/ml.wav) 
- [Mongolian](mn.zip) (71 hours, 6.4G), [sample](samples/mn.wav) 
- [Marathi](mr.zip) (85 hours, 8.1G), [sample](samples/mr.wav) 
- [Malay](ms.zip) (83 hours, 7.8G), [sample](samples/ms.wav) 
- [Maltese](mt.zip) (66 hours, 6.1G), [sample](samples/mt.wav) 
- [Burmese](my.zip) (41 hours, 4.0G), [sample](samples/my.wav) 
- [Nepali](ne.zip) (72 hours, 7.1G), [sample](samples/ne.wav) 
- [Dutch](nl.zip) (40 hours, 3.8G), [sample](samples/nl.wav) 
- [Norwegian Nynorsk](nn.zip) (57 hours, 4.8G), [sample](samples/nn.wav) 
- [Norwegian](no.zip) (107 hours, 9.7G), [sample](samples/no.wav) 
- [Occitan](oc.zip) (15 hours, 1.5G), [sample](samples/oc.wav) 
- [Panjabi](pa.zip) (54 hours, 5.2G), [sample](samples/pa.wav) 
- [Polish](pl.zip) (80 hours, 7.6G), [sample](samples/pl.wav) 
- [Pushto](ps.zip) (47 hours, 4.5G), [sample](samples/ps.wav) 
- [Portuguese](pt.zip) (64 hours, 6.1G), [sample](samples/pt.wav) 
- [Romanian](ro.zip) (65 hours, 6.1G), [sample](samples/ro.wav) 
- [Russian](ru.zip) (73 hours, 6.9G), [sample](samples/ru.wav) 


### Citing

```
@inproceedings{valk2021slt,
  title={{VoxLingua107}: a Dataset for Spoken Language Recognition},
  author={J{\"o}rgen Valk and Tanel Alum{\"a}e},
  booktitle={Proc. IEEE SLT Workshop},
  year={2021},
}
