"""Test markdown rendering for the blog."""
import markdown as md
import bleach

ALLOWED_TAGS = ['a','abbr','b','blockquote','br','caption','cite','code','col','colgroup','dd','del','details','dfn','div','dl','dt','em','figcaption','figure','h1','h2','h3','h4','h5','h6','hr','i','img','ins','kbd','li','mark','ol','p','pre','q','s','samp','small','span','strong','sub','summary','sup','table','tbody','td','tfoot','th','thead','time','tr','u','ul','var']

ALLOWED_ATTRS = {'a':['href','title','rel'],'abbr':['title'],'col':['span','width'],'colgroup':['span','width'],'img':['src','alt','title','width','height'],'ol':['start','type'],'td':['colspan','rowspan','align'],'th':['colspan','rowspan','align','scope'],'time':['datetime'],'li':['value'],'pre':['class'],'code':['class'],'span':['class'],'div':['class'],'table':['class'],'details':['open']}

def render_markdown(text):
    if not text:
        return ""
    html = md.markdown(text, extensions=["fenced_code", "codehilite", "tables", "toc", "sane_lists"])
    return bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)

def strip_html(text):
    return bleach.clean(text, tags=[], strip=True) if text else ""

# Test cases
test_cases = [
    ("# Heading 1", "<h1>Heading 1</h1>"),
    ("**bold** and *italic*", "<p><strong>bold</strong> and <em>italic</em></p>"),
    ("`inline code`", "<p><code>inline code</code></p>"),
    ("- List item", "<ul>\n<li>List item</li>\n</ul>"),
    ("[Link](https://example.com)", '<p><a href="https://example.com">Link</a></p>'),
]

for md_input, expected_html in test_cases:
    result = render_markdown(md_input)
    assert expected_html in result, f"{md_input!r} failed: expected {expected_html!r}, got {result!r}"
    print(f"PASS: {md_input!r} -> {result!r}")

# Test strip_html
assert strip_html("<p><strong>Hello</strong></p>") == "Hello"
print(f"PASS: strip_html works")

# Test empty
assert render_markdown("") == ""
assert render_markdown(None) == ""
print(f"PASS: empty handling works")

print("\nAll tests passed!")
