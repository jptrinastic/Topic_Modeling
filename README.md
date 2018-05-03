# Bibliometric Topic Modeling for Topic-Normalized Citation Analysis

This code provides a pipeline to perform natural language processing and unsupervised cluster analyses on any set of document texts (corpus).  The original intent of the code was to assign technical category labels to publications and patents, with the goal of normalizing citations by tech category.  However, the code has been written so that its use can be generalized to any type of text analysis requiring document clustering.  The Jupyter notebook included with the code provides a step-by-step example of how to use all code features.  Code to use the Web of Science API to create a corpus is also included, but this will require a Web of Science API username and password.  Any excel file or csv file containing the document corpus can also be loaded to get started with the rest of the code.

## Getting Started

Please refer to the included Jupyter notebook for a general tutorial about how to use all code features.

### Prerequisites

Libraries used to build this package:
numpy 1.13.3

pandas 0.22.0
scipy 1.0.0
matplotlib 2.1.0
sklearn 0.19.1
nltk 3.2.5
re 2.2.1
python-rake (https://pypi.org/project/python-rake/)

## Authors

* **Jonathan Trinastic** - *Initial work* - (https://github.com/jptrinastic)

## License

This project is not licensed.

## Acknowledgments

* Thanks to Dave Rench-McCauley for ideas and inspiration to develop this package!
