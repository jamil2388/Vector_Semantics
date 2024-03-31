# Vector_Semantics
This project studies the comparison of vector semantic methods with gold standard dataset SimLex-999 and reports multiple evaluation results


### Project Setup

```
# create the virtualenv vsm, python3.10 recommended
python3.10 -m virtualenv vsm

# on windows
vsm\Scripts\activate

# on linux
source vsm/bin/activate

# install from requirements.txt
pip install -r requirements.txt

# download the required corpora
python -m nltk.downloader brown

```