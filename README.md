# language-detection

This repository provides a workflow for language detection using a character n-gram language profiling technique from Cavnar and Trenkle (1994). The WiLI-2018 language identification data set (235 languages) is used for both training and evaluation. Our best language detection model achieved a weighted F<sub>1</sub> score of 89.8% on the WiLI-2018 test set. A presentation summarizing this project can be found [here](./docs/presentation/main.pdf).

## Dependencies :neckbeard:

This repository's code was tested with Python versions `3.7.*`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages via `pip`:

```
$ pip install -r requirements.txt
```

## Initialization :fire:

1. Automatically download and prepare the WiLI-2018 [data set](https://zenodo.org/record/841984):

    ```
    $ bash scripts/prepare_data.sh
    ```

2. **Optional:** Initialize git hooks to manage development workflows such as linting shell scripts, keeping python dependencies up-to-date and formatting the development log:

    ```
    $ bash scripts/prepare_git_hooks.sh
    ```

## Usage :snowflake:

<details><summary>i. Training</summary>
<p>

```
usage: train.py [-h] [--logging-level {debug,info,warning,error,critical}]
                [--models-directory <dir_path>] [--ngram-cutoff <int>]
                [--ngrams <int>] [--train-data <file_path>]
                [--train-labels <file_path>]

optional arguments:
  --logging-level     {debug,info,warning,error,critical}
                      Set logging level (default: info)
  --models-directory  <dir_path>
                      Directory to dump models and logs (default: ./models)
  --ngram-cutoff      <int>
                      Maximum character n-grams per language profile (default:
                      300)
  --ngrams            <int>
                      Character n-grams to use for language profiles (default:
                      3)
  --train-data        <file_path>
                      Path to training data (default:
                      ./data/wili-2018/x_train.txt)
  --train-labels      <file_path>
                      Path to training labels (default:
                      ./data/wili-2018/y_train.txt)
  -h, --help          show this help message and exit
```

To train a language detection model using our defaults, simply execute:

```
$ python3 -m src.train
```

**Note:** Our default model is already provided in the `./models` directory

</p>
</details>

<details><summary>ii. Evaluation</summary>
<p>

```
usage: evaluate.py [-h] [--logging-level {debug,info,warning,error,critical}]
                   [--model <file_path>] [--models-directory <dir_path>]
                   [--test-data <file_path>] [--test-labels <file_path>]

optional arguments:
  --logging-level     {debug,info,warning,error,critical}
                      Set logging level (default: info)
  --model             <file_path>
                      Path to model JSON file (default:
                      ./models/model_3_300.json)
  --models-directory  <dir_path>
                      Directory to dump models and logs (default: ./models)
  --test-data         <file_path>
                      Path to test data (default: ./data/wili-2018/x_test.txt)
  --test-labels       <file_path>
                      Path to test labels (default:
                      ./data/wili-2018/y_test.txt)
  -h, --help          show this help message and exit
```

To evaluate the default language detection model, simply execute:

```
$ python3 -m src.evaluate
```

This will dump a classification report into the directory specified in `--models-directory`.

**Note:** The classification report for our default model is already provided in the `./models` directory

</p>
</details>

<details><summary>iii. Prediction</summary>
<p>

```
usage: predict.py [-h] --predict-data <file_path>
                  [--logging-level {debug,info,warning,error,critical}]
                  [--model <file_path>]

optional arguments:
  --logging-level  {debug,info,warning,error,critical}
                   Set logging level (default: info)
  --model          <file_path>
                   Path to model JSON file (default:
                   ./models/model_3_300.json)
  -h, --help       show this help message and exit

required arguments:
  --predict-data   <file_path>
                   Path to prediction data (default: None)
```

To predict the language of a document using our default language detection model, simply execute:

```
$ python3 -m src.predict --predict-data /path/to/document
```

**Notes:**

1. This prediction workflow assumes one document per line. If your document is multi-lined, please condense it into a single line

2. The meaning of each output label is expounded in `./data/wili-2018/labels.csv`

</p>
</details>

## Test :microscope:

1. To run a `mypy` typecheck on our source code, execute:

    ```
    $ bash scripts/test_typecheck.sh
    ```

2. To test our default model on some sample data, execute the following:

    ```
    $ python3 -m src.predict --predict-data test/input.txt
    ```

## References :book:

```bibtex
@inproceedings{cavnar1994n,
  title={N-gram-based text categorization},
  author={Cavnar, William B and Trenkle, John M and others},
  booktitle={Proceedings of SDAIR-94, 3rd annual symposium on
  document analysis and information retrieval},
  volume={161175},
  year={1994},
  organization={Citeseer}
}

@article{thoma2018wili,
  title={The WiLI benchmark dataset for written language identification},
  author={Thoma, Martin},
  journal={arXiv preprint arXiv:1801.07779},
  year={2018}
}
```

<!--  LocalWords:  Cavnar Trenkle WiLI neckbeard typecheck
 -->
