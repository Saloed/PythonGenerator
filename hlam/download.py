import ast
import html
import re
import time
import requests
import _pickle as P

API_URL = "https://api.stackexchange.com/2.2"


def get_code():
    return _fetch_code(fetch_urls())


def get_source(cls, fullname):
    return cls.get_code(fullname)


def fetch_urls():
    page = 0
    result = []
    has_more = True
    while has_more:
        page += 1
        try:
            time.sleep(2)
            ans = requests.get(API_URL + "/search", {
                'page': page,
                'pagesize': 100,
                "order": "desc",
                "sort": "creation",
                "tagged": "python",
                "site": "stackoverflow",
            }).json()
        except Exception as ex:
            print(ex)
            break
        if "items" not in ans or not ans["items"]:
            print('no more answers')
            break
        has_more = ans['has_more']
        result += ans['items']

    print(len(result))
    save_file(result, 'python_posts.pkl')
    return result


def fetch_post(link):
    return requests.get(link)


def load_file(name):
    with open(name, 'rb') as f:
        obj = P.load(f)
    return obj


def save_file(obj, name):
    with open(name, 'wb') as f:
        P.dump(obj, f)


def _fetch_code(post):
    q = requests.get(post['link'])
    return cls._find_code_in_html(q.text)



def find_code_in_html(s):
    answers = re.findall(r'<div id="answer-.*?</table', s, re.DOTALL)  # come get me, Zalgo

    def votecount(x):
        """
        Return the negative number of votes a question has.
        Might return the negative question id instead if its less than 100k. That's a feature.
        """
        r = int(re.search(r"\D(\d{1,5})\D", x).group(1))
        return -r

    for answer in sorted(answers, key=votecount):
        codez = re.finditer(r"<pre[^>]*>[^<]*<code[^>]*>((?:\s|[^<]|<span[^>]*>[^<]+</span>)*)</code></pre>", answer)
        codez = map(lambda x: x.group(1), codez)
        for code in sorted(codez, key=lambda x: -len(x)):  # more code is obviously better
            # don't forget attribution
            author = s
            author = author[author.find(code):]
            author = author[:author.find(">share<")]
            author = author[author.rfind('<a href="') + len('<a href="'):]
            author_link = author[:author.find('"'):]
            author_link = "https://stackoverflow.com" + author_link

            # fetch that code
            code = html.unescape(code)
            code = re.sub(r"<[^>]+>([^<]*)<[^>]*>", "\1", code)
            try:
                ast.parse(code)
                return code, author_link  # it compiled! uhm, parsed!
            except:
                pass
    else:  # https://stackoverflow.com/questions/9979970/why-does-python-use-else-after-for-and-while-loops
        raise ImportError("This question ain't got no good code.")


if __name__ == '__main__':
    urls = fetch_urls()
    urls = load_file('python_posts.pkl')
    posts = [fetch_post(url['link']) for url in urls]
    save_file(posts, 'fetched_posts.pkl')
    posts = load_file('fetched_posts.pkl')
    code = [find_code_in_html(p) for p in posts]
    save_file(code, 'python_code.pkl')
