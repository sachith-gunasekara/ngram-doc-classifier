## Table of Contents
-   [Notes](#notes)
    -   [Tasks](#tasks)

## Notes

### Tasks

1.  Motivation

    1.  structure of language with importance of morphology:
        <https://en.wikipedia.org/wiki/Linguistics>

    2.  mention older articles use data more wisely -\> interesting
        workarounds

    3.  mention usage of modern techniques ommitted due to no libraries
        allowed -\> keep things simple like in the older days

    4.  mention data set used for evaluation since nothing was provided
        -\> 235 languages with 235,000 paragraphs + balanced data set

2.  Workflow

    1.  cite and describe classical problem -\> use
        `Cavnar and Trenkle (1994)`

    2.  mention why using words was not so easy and why character
        n-grams were preferred -\> generalization due to morphology

    3.  mention lowercasing and removal of special tokens and digits for
        noise removal

    4.  use pseudocode to describe problem-solving process and mention
        hyperparameters

    5.  if possible, show some examples of morphology captured which
        could be interesting

3.  Limitations and workaround

    1.  problematic on short phrases due to lack of morphology profiles
        being built up -\> new NN approaches to handle short phrases -\>
        cite if possible

    2.  heavy solution could also be a large word level vocabulary which
        can handle everything -\> would actually be ideal if possible

    3.  mention word level strategy and challenges with non Latin
        scripts -\> hard to split in all cases

    4.  talk about efficiency and other techniques as further work

4.  Admin

    1.  prepare maximum 5 slides within 15 minutes for HSLU interview
