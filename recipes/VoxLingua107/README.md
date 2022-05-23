# VoxLingua107

VoxLingua107 is a speech dataset for training spoken language identification models.
The dataset consists of short speech segments automatically extracted from YouTube videos and labeled according the language of the video title and description, with some post-processing steps to filter out false positives.

VoxLingua107 contains data for 107 languages. The total amount of speech in the training set is 6628 hours.
The average amount of data per language is 62 hours. However, the real amount per language varies a lot. There is also a seperate development set containing 1609 speech segments from 33 languages, validated by at least two volunteers to really contain the given language.

For more information, see the paper [J&ouml;rgen Valk, Tanel Alum&auml;e. _VoxLingua107: a Dataset for Spoken Language Recognition_. Proc. SLT 2021].

## Why

VoxLingua107 can be used for training spoken language recognition models that work well with real-world, varying speech data.

## How

We extracted audio data from YouTube videos that are retrieved using language-specific search phrases .
If the language of the video title and description matched with the language of the search phrase,
the audio in the video was deemed likely to be in that particular language. This allowed to collect large amounts of somewhat noisy data relatively cheaply.
Speech/non-speech detection and speaker diarization was used to segment the videos into short sentence-like utterances.
A data-driven post-filtering step was applied to remove clips that were very different from other clips in this language's dataset, and thus likely not in the given language.
Due to the automatic data collection process, there are still clips in the dataset that are not in the given language or contain non-speech.

## Languages

Amount of training data per language:

- Abkhazian  (10 hours, 980M)
- Afrikaans  (108 hours, 10G)
- Amharic  (81 hours, 7.7G)
- Arabic  (59 hours, 5.5G)
- Assamese  (155 hours, 15G)
- Azerbaijani  (58 hours, 5.6G)
- Bashkir  (58 hours, 5.5G)
- Belarusian  (133 hours, 13G)
- Bulgarian  (50 hours, 4.7G)
- Bengali  (55 hours, 5.4G)
- Tibetan  (101 hours, 9.3G)
- Breton  (44 hours, 4.2G)
- Bosnian  (105 hours, 9.7G)
- Catalan  (88 hours, 8.1G)
- Cebuano  (6 hours, 589M)
- Czech  (67 hours, 6.3G)
- Welsh  (76 hours, 6.6G)
- Danish  (28 hours, 2.6G)
- German  (39 hours, 3.7G)
- Greek  (66 hours, 6.2G)
- English  (49 hours, 4.6G)
- Esperanto  (10 hours, 916M)
- Spanish  (39 hours, 3.7G)
- Estonian  (38 hours, 3.5G)
- Basque  (29 hours, 2.8G)
- Persian  (56 hours, 5.2G)
- Finnish  (33 hours, 3.1G)
- Faroese  (67 hours, 6.0G)
- French  (67 hours, 6.2G)
- Galician  (72 hours, 6.7G)
- Guarani  (2 hours, 250M)
- Gujarati  (46 hours, 4.5G)
- Manx  (4 hours, 374M)
- Hausa  (106 hours, 10G)
- Hawaiian  (12 hours, 1.2G)
- Hindi  (81 hours, 7.7G)
- Croatian  (118 hours, 11G)
- Haitian  (96 hours, 9.2G)
- Hungarian  (73 hours, 6.9G)
- Armenian  (69 hours, 6.6G)
- Interlingua  (3 hours, 241M)
- Indonesian  (40 hours, 3.8G)
- Icelandic  (92 hours, 8.4G)
- Italian  (51 hours, 4.8G)
- Hebrew  (96 hours, 8.9G)
- Japanese  (56 hours, 5.1G)
- Javanese  (53 hours, 5.0G)
- Georgian  (98 hours, 9.2G)
- Kazakh  (78 hours, 7.3G)
- Central Khmer  (41 hours, 4.0G)
- Kannada  (46 hours, 4.4G)
- Korean  (77 hours, 7.1G)
- Latin  (67 hours, 6.0G)
- Luxembourgish  (75 hours, 7.1G)
- Lingala  (90 hours, 8.7G)
- Lao  (42 hours, 4.0G)
- Lithuanian  (82 hours, 7.7G)
- Latvian  (42 hours, 4.0G)
- Malagasy  (109 hours, 11G)
- Maori  (34 hours, 3.2G)
- Macedonian  (112 hours, 11G)
- Malayalam  (47 hours, 4.6G)
- Mongolian  (71 hours, 6.4G)
- Marathi  (85 hours, 8.1G)
- Malay  (83 hours, 7.8G)
- Maltese  (66 hours, 6.1G)
- Burmese  (41 hours, 4.0G)
- Nepali  (72 hours, 7.1G)
- Dutch  (40 hours, 3.8G)
- Norwegian Nynorsk  (57 hours, 4.8G)
- Norwegian  (107 hours, 9.7G)
- Occitan  (15 hours, 1.5G)
- Panjabi  (54 hours, 5.2G)
- Polish  (80 hours, 7.6G)
- Pushto  (47 hours, 4.5G)
- Portuguese  (64 hours, 6.1G)
- Romanian  (65 hours, 6.1G)
- Russian  (73 hours, 6.9G)
- Sanskrit  (15 hours, 1.6G)
- Scots  (3 hours, 269M)
- Sindhi  (84 hours, 8.3G)
- Sinhala  (67 hours, 6.4G)
- Slovak  (40 hours, 3.7G)
- Slovenian  (121 hours, 12G)
- Shona  (30 hours, 2.9G)
- Somali  (103 hours, 9.9G)
- Albanian  (71 hours, 6.6G)
- Serbian  (50 hours, 4.7G)
- Sundanese  (64 hours, 6.2G)
- Swedish  (34 hours, 3.1G)
- Swahili  (64 hours, 6.1G)
- Tamil  (51 hours, 5.0G)
- Telugu  (77 hours, 7.5G)
- Tajik  (64 hours, 6.1G)
- Thai  (61 hours, 5.8G)
- Turkmen  (85 hours, 8.1G)
- Tagalog  (93 hours, 8.7G)
- Turkish  (59 hours, 5.7G)
- Tatar  (103 hours, 9.6G)
- Ukrainian  (52 hours, 4.9G)
- Urdu  (42 hours, 4.1G)
- Uzbek  (45 hours, 4.3G)
- Vietnamese  (64 hours, 6.1G)
- Waray  (11 hours, 1.1G)
- Yiddish  (46 hours, 4.4G)
- Yoruba  (94 hours, 9.1G)
- Mandarin Chinese  (44 hours, 4.1G)

