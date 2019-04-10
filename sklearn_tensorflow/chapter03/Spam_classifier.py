# _*_ coding: utf-8 _*_
import os
import re
import nltk
import email
import tarfile
import urlextract
import numpy as np
import email.parser
import email.policy
from html import unescape
from six.moves import urllib
from collections import Counter
from sklearn.model_selection import train_test_split

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]


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


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = 'spam' if is_spam else 'easy_ham'
    with open(os.path.join(spam_path, directory, filename), 'rb') as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(','.join([get_email_structure(sub_email) for sub_email in payload]))
    else:
        return email.get_content_type()


def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain",'text/html'):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_content())
        if ctype == 'text/plain':
            return content
    if html:
        return html_to_plain_text(html)


if __name__ == '__main__':
    ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
    spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
    # print(structures_counter(ham_emails).most_common())
    # print(structures_counter(spam_emails).most_common())
    # print(ham_emails[1].get_content().strip())
    # print(spam_emails[6].get_content().strip())
    # for header, value in spam_emails[0].items():
    #     print(header, ":", value)
    X = np.array(ham_emails + spam_emails)
    y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    html_spam_emails = [email for email in X_train[y_train == 1]
                        if get_email_structure(email) == "text/html"]
    sample_html_spam = html_spam_emails[7]
    # print(sample_html_spam.get_content().strip()[:1000], "...")
    # print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")