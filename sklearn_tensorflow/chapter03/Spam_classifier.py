# _*_ coding: utf-8 _*_
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")


def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH, ham_url=HAM_URL):
    if not os.path.exists(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("han.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            auto_down(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()


def auto_down(url, filename):
    try:
        urllib.request.urlretrieve(url, filename)
    except urllib.error.ContentTooShortError:
        print("Network conditions is not good Reloading")
        auto_down(url, filename)


if __name__ == '__main__':
    fetch_spam_data()
    HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
    SPAM_DIR = os.path.join(SPAM_PATH, "spam")
    ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
    spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]
    print(len(ham_filenames))
    print(len(spam_filenames))
