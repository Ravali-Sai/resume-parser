#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# training data
TRAIN_DATA = [
    ("Who is Shaka Khan May,18 ?", {"entities": [(7, 17, "i1"), (18, 24, "Resdte")]}),
    ("I like London and not Berlin.", {"entities": [(7, 13, "i2"), (22, 28, "i2")]}),
    ("charter communications Apr,17 - May,18 ", {"entities": [(23, 29, "Resdte"), (32, 38, "Resdte")]}),
    ("Hunting Energy Services – Houston, TX        June, 2018 to October, 2018 ", {"entities": [(45,55,"Resdte"), (59,72,"Resdte")]}),
    ("Bjerkeli Consulting – Spring,TX        August, 2016 to June, 2018", {"entities": [(39,51,"Resdte"), (55,65,"Resdte")]}),
    ("ASSET InterTech, Inc.-  Richardson, TX         April, 2001 to July, 2016 ", {"entities": [(47,58,"Resdte"), (62,72,"Resdte")]}),
    ("06/13 – 7/14 Sapient Nitro – New York, NY    Sr. Associate Platform L2", {"entities": [(0,5,"Resdte"), (8,12,"Resdte")]}),
    ("07/14 – 10/15        LyonsCG – Chicago,IL (Remote from New York)    Senior Technical Architect", {"entities": [(0,5,"Resdte"), (8,13,"Resdte")]}),
    ('charter communications Jan,17 - Feb,18 charter communications Feb,17 - Mar,18Â\xa0charter communications Mar,17 - Apr,18 charter communications Apr,17 - May,18Â\xa0charter communications May,17 - Jun,18 charter communications Jun,17 - Jul,18Â\xa0charter communications Jul,17 - Aug,18 charter communications Aug,17 - Sep,18 charter communications Sep,17 - Oct,18 charter communications Oct,17 - Nov,18 charter communications Nov,17 - Dec,18 charter communications Dec,17 - Jan,18.', {'entities': [(23, 29, 'Resdte'), (32, 38, 'Resdte'), (62, 68, 'Resdte'), (71, 77, 'Resdte'), (101, 107, 'Resdte'), (110, 116, 'Resdte'), (140, 146, 'Resdte'), (149, 155, 'Resdte'), (179, 185, 'Resdte'), (188, 194, 'Resdte'), (218, 224, 'Resdte'), (227, 233, 'Resdte'), (257, 263, 'Resdte'), (266, 272, 'Resdte'), (296, 302, 'Resdte'), (305, 311, 'Resdte'), (335, 341, 'Resdte'), (344, 350, 'Resdte'), (374, 380, 'Resdte'), (383, 389, 'Resdte'), (413, 419, 'Resdte'), (422, 428, 'Resdte'), (452, 458, 'Resdte'), (461, 467, 'Resdte')]}),
    ('Texas Instruments Jan, 17 - Feb, 18Â\xa0 Texas Instruments Feb, 17 - Mar, 18 Texas Instruments Mar, 17 - Apr, 18 Texas Instruments Apr, 17 - May, 18 Texas Instruments May, 17 - Jun, 18 Texas Instruments Jun, 17 - Jul, 18 Texas Instruments Jul, 17 - Aug, 18 Texas Instruments Aug, 17 - Sep, 18 Texas Instruments Sep, 17 - Oct, 18 Texas Instruments Oct, 17 - Nov, 18 Texas Instruments Nov, 17 - Dec, 18Â\xa0Texas Instruments Dec, 17 - Jan, 18', {'entities': [(18, 25, 'Resdte'), (28, 35, 'Resdte'), (55, 62, 'Resdte'), (65, 72, 'Resdte'), (91, 98, 'Resdte'), (101, 108, 'Resdte'), (127, 134, 'Resdte'), (137, 144, 'Resdte'), (163, 170, 'Resdte'), (173, 180, 'Resdte'), (199, 206, 'Resdte'), (209, 216, 'Resdte'), (235, 242, 'Resdte'), (245, 252, 'Resdte'), (271, 278, 'Resdte'), (281, 288, 'Resdte'), (307, 314, 'Resdte'), (317, 324, 'Resdte'), (343, 350, 'Resdte'), (353, 360, 'Resdte'), (379, 386, 'Resdte'), (389, 396, 'Resdte'), (415, 422, 'Resdte'), (425, 432, 'Resdte')]}),
    ('TEXAS INSTRUMENTS JAN, 17 - FEB, 18Â\xa0 TEXAS INSTRUMENTS FEB, 17 - MAR, 18 TEXAS INSTRUMENTS MAR, 17 - APR, 18 TEXAS INSTRUMENTS APR, 17 - MAY, 18 TEXAS INSTRUMENTS MAY, 17 - JUN, 18 TEXAS INSTRUMENTS JUN, 17 - JUL, 18 TEXAS INSTRUMENTS JUL, 17 - AUG, 18 TEXAS INSTRUMENTS AUG, 17 - SEP, 18 TEXAS INSTRUMENTS SEP, 17 - OCT, 18 TEXAS INSTRUMENTS OCT, 17 - NOV, 18 TEXAS INSTRUMENTS NOV, 17 - DEC, 18 TEXAS INSTRUMENTS DEC, 17 - JAN, 18', {'entities': [(18, 25, 'Resdte'), (28, 35, 'Resdte'), (55, 62, 'Resdte'), (65, 72, 'Resdte'), (91, 98, 'Resdte'), (101, 108, 'Resdte'), (127, 134, 'Resdte'), (137, 144, 'Resdte'), (163, 170, 'Resdte'), (173, 180, 'Resdte'), (199, 206, 'Resdte'), (209, 216, 'Resdte'), (235, 242, 'Resdte'), (245, 252, 'Resdte'), (271, 278, 'Resdte'), (281, 288, 'Resdte'), (307, 314, 'Resdte'), (317, 324, 'Resdte'), (343, 350, 'Resdte'), (353, 360, 'Resdte'), (379, 386, 'Resdte'), (389, 396, 'Resdte'), (415, 422, 'Resdte'), (425, 432, 'Resdte')]}),
    ('TEXAS INSTRUMENTS JAN,17 - FEB,18Â\xa0 TEXAS INSTRUMENTS FEB,17 - MAR,18 TEXAS INSTRUMENTS MAR,17 - APR,18 TEXAS INSTRUMENTS APR,17 - MAY,18 TEXAS INSTRUMENTS MAY,17 - JUN,18 TEXAS INSTRUMENTS JUN,17 - JUL,18 TEXAS INSTRUMENTS JUL,17 - AUG,18 TEXAS INSTRUMENTS AUG,17 - SEP,18 TEXAS INSTRUMENTS SEP,17 - OCT,18 TEXAS INSTRUMENTS OCT,17 - NOV,18 TEXAS INSTRUMENTS NOV,17 - DEC,18 TEXAS INSTRUMENTS DEC,17 - JAN,18', {'entities': [(18, 24, 'Resdte'), (27, 33, 'Resdte'), (53, 59, 'Resdte'), (62, 68, 'Resdte'), (87, 93, 'Resdte'), (96, 103, 'Resdte'), (121, 127, 'Resdte'), (130, 136, 'Resdte'), (155, 161, 'Resdte'), (164, 170, 'Resdte'), (189, 195, 'Resdte'), (198, 205, 'Resdte'), (223, 229, 'Resdte'), (232, 238, 'Resdte'), (257, 263, 'Resdte'), (266, 272, 'Resdte'), (291, 297, 'Resdte'), (300, 307, 'Resdte'), (325, 331, 'Resdte'), (334, 340, 'Resdte'), (359, 365, 'Resdte'), (368, 374, 'Resdte'), (393, 399, 'Resdte'), (402, 408, 'Resdte')]}), ('ASSET InterTech, Inc. Richardson, TX January, 2001 to February, 2016 ASSET InterTech, Inc. Richardson, TX February, 2001 to March, 2016 ASSET InterTech, Inc. Richardson, TX March, 2001 to April, 2016 ASSET InterTech, Inc. Richardson, TX April, 2001 to May, 2016 ASSET InterTech, Inc. Richardson, TX May, 2001 to June, 2016 ASSET InterTech, Inc. Richardson, TX June, 2001 to July, 2016 ASSET InterTech, Inc. Richardson, TX July, 2001 to August, 2016 ASSET InterTech, Inc. Richardson, TX August, 2001 to September, 2016 ASSET InterTech, Inc. Richardson, TX September, 2001 to October, 2016 ASSET InterTech, Inc. Richardson, TX October, 2001 to November, 2016 ASSET InterTech, Inc. Richardson, TX November, 2001 to December, 2016 ASSET InterTech, Inc. Richardson, TX December, 2001 to January, 2016', {'entities': [(37, 50, 'Resdte'), (54, 68, 'Resdte'), (106, 120, 'Resdte'), (124, 135, 'Resdte'), (173, 184, 'Resdte'), (188, 199, 'Resdte'), (237, 248, 'Resdte'), (252, 261, 'Resdte'), (299, 309, 'Resdte'), (312, 322, 'Resdte'), (360, 370, 'Resdte'), (374, 384, 'Resdte'), (422, 432, 'Resdte'), (436, 448, 'Resdte'), (486, 498, 'Resdte'), (502, 517, 'Resdte'), (555, 570, 'Resdte'), (574, 587, 'Resdte'), (625, 638, 'Resdte'), (642, 656, 'Resdte'), (694, 709, 'Resdte'), (712, 726, 'Resdte'), (764, 778, 'Resdte'), (782, 795, 'Resdte')]}), ('ASSET INTERTECH, INC. RICHARDSON, TX JANUARY, 2001 TO FEBRUARY, 2016 ASSET INTERTECH, INC. RICHARDSON, TX FEBRUARY, 2001 TO MARCH, 2016 ASSET INTERTECH, INC. RICHARDSON, TX MARCH, 2001 TO APRIL, 2016 ASSET INTERTECH, INC. RICHARDSON, TX APRIL, 2001 TO MAY, 2016 ASSET INTERTECH, INC. RICHARDSON, TX MAY, 2001 TO JUNE, 2016 ASSET INTERTECH, INC. RICHARDSON, TX JUNE, 2001 TO JULY, 2016 ASSET INTERTECH, INC. RICHARDSON, TX JULY, 2001 TO AUGUST, 2016 ASSET INTERTECH, INC. RICHARDSON, TX AUGUST, 2001 TO SEPTEMBER, 2016 ASSET INTERTECH, INC. RICHARDSON, TX SEPTEMBER, 2001 TO OCTOBER, 2016 ASSET INTERTECH, INC. RICHARDSON, TX OCTOBER, 2001 TO NOVEMBER, 2016 ASSET INTERTECH, INC. RICHARDSON, TX NOVEMBER, 2001 TO DECEMBER, 2016 ASSET INTERTECH, INC. RICHARDSON, TX DECEMBER, 2001 TO JANUARY, 2016', {'entities': [(37, 50, 'Resdte'), (54, 68, 'Resdte'), (106, 120, 'Resdte'), (124, 135, 'Resdte'), (173, 184, 'Resdte'), (188, 199, 'Resdte'), (237, 248, 'Resdte'), (252, 261, 'Resdte'), (299, 309, 'Resdte'), (312, 322, 'Resdte'), (360, 370, 'Resdte'), (374, 384, 'Resdte'), (422, 432, 'Resdte'), (436, 448, 'Resdte'), (486, 498, 'Resdte'), (502, 517, 'Resdte'), (555, 570, 'Resdte'), (574, 587, 'Resdte'), (625, 638, 'Resdte'), (642, 656, 'Resdte'), (694, 709, 'Resdte'), (712, 726, 'Resdte'), (764, 778, 'Resdte'), (782, 795, 'Resdte')]}), ('Asset Intertech, Inc. Richardson, Tx January,2001 To February,2016 Asset Intertech, Inc. Richardson, Tx February,2001 To March,2016 Asset Intertech, Inc. Richardson, Tx March,2001 To April,2016 Asset Intertech, Inc. Richardson, Tx April,2001 To May,2016 Asset Intertech, Inc. Richardson, Tx May,2001 To June,2016 Asset Intertech, Inc. Richardson, Tx June,2001 To July,2016 Asset Intertech, Inc. Richardson, Tx July,2001 To August,2016 Asset Intertech, Inc. Richardson, Tx August,2001 To September,2016 Asset Intertech, Inc. Richardson, Tx September,2001 To October,2016 Asset Intertech, Inc. Richardson, Tx October,2001 To November,2016 Asset Intertech, Inc. Richardson, Tx November,2001 To December,2016 Asset Intertech, Inc. Richardson, Tx December,2001 To January,2016', {'entities': [(37, 49, 'Resdte'), (53, 66, 'Resdte'), (104, 117, 'Resdte'), (121, 131, 'Resdte'), (169, 179, 'Resdte'), (183, 193, 'Resdte'), (231, 241, 'Resdte'), (245, 253, 'Resdte'), (291, 299, 'Resdte'), (303, 312, 'Resdte'), (350, 359, 'Resdte'), (363, 372, 'Resdte'), (410, 419, 'Resdte'), (423, 434, 'Resdte'), (472, 483, 'Resdte'), (487, 501, 'Resdte'), (539, 553, 'Resdte'), (557, 569, 'Resdte'), (607, 619, 'Resdte'), (623, 636, 'Resdte'), (674, 687, 'Resdte'), (691, 704, 'Resdte'), (742, 755, 'Resdte'), (759, 771, 'Resdte')]}), ('ASSET INTERTECH, INC. RICHARDSON, TX JANUARY,2001 TO FEBRUARY,2016 ASSET INTERTECH, INC. RICHARDSON, TX FEBRUARY,2001 TO MARCH,2016 ASSET INTERTECH, INC. RICHARDSON, TX MARCH,2001 TO APRIL,2016 ASSET INTERTECH, INC. RICHARDSON, TX APRIL,2001 TO MAY,2016 ASSET INTERTECH, INC. RICHARDSON, TX MAY,2001 TO JUNE,2016 ASSET INTERTECH, INC. RICHARDSON, TX JUNE,2001 TO JULY,2016 ASSET INTERTECH, INC. RICHARDSON, TX JULY,2001 TO AUGUST,2016 ASSET INTERTECH, INC. RICHARDSON, TX AUGUST,2001 TO SEPTEMBER,2016 ASSET INTERTECH, INC. RICHARDSON, TX SEPTEMBER,2001 TO OCTOBER,2016 ASSET INTERTECH, INC. RICHARDSON, TX OCTOBER,2001 TO NOVEMBER,2016 ASSET INTERTECH, INC. RICHARDSON, TX NOVEMBER,2001 TO DECEMBER,2016 ASSET INTERTECH, INC. RICHARDSON, TX DECEMBER,2001 TO JANUARY,2016', {'entities': [(37, 49, 'Resdte'), (53, 66, 'Resdte'), (104, 117, 'Resdte'), (121, 131, 'Resdte'), (169, 179, 'Resdte'), (183, 193, 'Resdte'), (231, 241, 'Resdte'), (245, 253, 'Resdte'), (291, 299, 'Resdte'), (303, 312, 'Resdte'), (350, 359, 'Resdte'), (363, 372, 'Resdte'), (410, 419, 'Resdte'), (423, 434, 'Resdte'), (472, 483, 'Resdte'), (487, 501, 'Resdte'), (539, 553, 'Resdte'), (557, 569, 'Resdte'), (607, 619, 'Resdte'), (623, 636, 'Resdte'), (674, 687, 'Resdte'), (691, 704, 'Resdte'), (742, 755, 'Resdte'), (759, 771, 'Resdte')]})
]

TEST_DATA = [
    ("Texas Instruments Apr,17 - May,18 ", {"entities": [(7, 17, "i1"), (18, 24, "Resdte")]}),
    ("infosys limited Apr,17 - May,18 ", {"entities": [(7, 13, "i2"), (18, 24, "i2")]}),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir="Model", n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    #modelfile = input("Enter your Model Name: ")
    #nlp.to_disk(modelfile)
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TEST_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # Entities [('Shaka Khan', 'PERSON')]
    # Tokens [('Who', '', 2), ('is', '', 2), ('Shaka', 'PERSON', 3),
    # ('Khan', 'PERSON', 1), ('?', '', 2)]
    # Entities [('London', 'LOC'), ('Berlin', 'LOC')]
    # Tokens [('I', '', 2), ('like', '', 2), ('London', 'LOC', 3),
    # ('and', '', 2), ('Berlin', 'LOC', 3), ('.', '', 2)]
