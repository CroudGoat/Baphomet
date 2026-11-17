#!/bin/bash
# findings_ja.md -> findings_ja.pdf (pandoc + weasyprint, Noto Sans CJK JP; fix the font paths in style.css for your machine)
cd "$(dirname "$0")"
pandoc findings_ja.md -s --metadata title=" " -c style.css -o findings_ja.html
sed -i 's#<header id="title-block-header">.*</header>##' findings_ja.html
weasyprint findings_ja.html findings_ja.pdf && rm findings_ja.html
