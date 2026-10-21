"""Two layout fixes applied to every docs/HW*/*.html (idempotent; re-run after regenerating a book).

1. Text uses the full page width: `--measure: 46rem` (about 2/3 of the 67.5rem page) becomes `none`,
   in the pages and in docs/tools/hw*_book/book.css.
2. Code listings show source line numbers: a <style id="ln-style"> + <script id="ln-js"> block is put
   right before </body>. It runs on DOMContentLoaded, so it comes after the book's own (synchronous)
   highlighting and data-hot even when apply_template.py later re-inserts book-js after this block.
   The number of line i is the figcaption's start line (`file.py:53–101`) + i, the same rule data-hot uses.
   The number is drawn by ::before from data-n, so copying the code does not copy the numbers.
"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLOCK = '''<style id="ln-style">
pre.has-ln .ln::before {
  content: attr(data-n); display: inline-block; min-width: var(--ln-w, 2ch); margin-right: 1.4ch;
  text-align: right; color: var(--muted); opacity: 0.6; user-select: none; -webkit-user-select: none;
  font-style: normal; font-weight: 400;
}
</style>
<script id="ln-js">
/* 程式引用的行號：第 i 行 = figcaption 的起始行號（file.py:53–101）+ i，和 data-hot 用同一個規則。
   等 DOMContentLoaded 才跑：book-js 是同步執行的，這樣不論兩段 script 誰在前，行號都加在上色與強調行之後。 */
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('figure.listing').forEach(function (fig) {
    var cap = fig.querySelector('figcaption'), pre = fig.querySelector('pre');
    var m = cap && pre && cap.textContent.match(/:(\\d+)/);
    if (!m || pre.classList.contains('has-ln')) return;
    var target = pre.querySelector('code') || pre, start = parseInt(m[1], 10);
    var lines = target.innerHTML.split('\\n');
    if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop();
    target.innerHTML = lines.map(function (line, i) {
      var ln = '<span class="ln" data-n="' + (start + i) + '"></span>';
      var hot = line.match(/^<span class="hot-line">/);
      return hot ? hot[0] + ln + line.slice(hot[0].length) : ln + line;
    }).join('\\n');
    pre.style.setProperty('--ln-w', String(start + lines.length - 1).length + 'ch');
    pre.classList.add('has-ln');
  });
});
</script>
'''

OLD_BLOCK = re.compile(r'<style id="ln-style">.*?</script>\n', re.S)


def fix_measure(s):
    return s.replace('--measure: 46rem;', '--measure: none;')


def main():
    pages = sorted(glob.glob(os.path.join(ROOT, 'docs', 'HW*', '*.html')))
    for f in pages:
        s = open(f, encoding='utf-8').read()
        t = OLD_BLOCK.sub('', fix_measure(s))
        assert t.count('</body>') == 1, f
        t = t.replace('</body>', BLOCK + '</body>')
        if t != s:
            open(f, 'w', encoding='utf-8').write(t)
    for f in sorted(glob.glob(os.path.join(ROOT, 'docs', 'tools', 'hw*_book', 'book.css'))):
        s = open(f, encoding='utf-8').read()
        if fix_measure(s) != s:
            open(f, 'w', encoding='utf-8').write(fix_measure(s))
    print(len(pages), 'pages')


if __name__ == '__main__':
    main()
