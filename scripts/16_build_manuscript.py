"""Assemble the manuscript: renumber citations in order of first appearance and build Word.

Reads  manuscript/manuscript_source.md  (text with the provisional citation numbers of the
                                         draft tabs, and a "# References" list "[n] ...")
Writes manuscript/manuscript_build.md   (renumbered Markdown)
       manuscript/manuscript.docx       (pandoc, with figures and equations)

Target journal: Land (MDPI). Style template: the official Land Word template
(manuscript/template/land-template.dot, from mdpi.com) when present. Otherwise the official
Agriculture template is used with the journal name in the running head and footer changed
to Land and the Agriculture logo removed (MDPI templates share the same styles). pandoc
takes page setup, headers and styles from the template; the paragraphs are then given the
template's MDPI_* styles. Without any template, a generic Palatino 10-pt reference
(manuscript/reference_mdpi.docx) is used.

Needs pandoc on the PATH and python-docx.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
SRC = MS / "manuscript_source.md"
BUILD = MS / "manuscript_build.md"
DOCX = MS / "manuscript.docx"
REFDOC = MS / "reference_mdpi.docx"
JOURNAL = "Land"
LAND_DOT = MS / "template" / "land-template.dot"
AGRI_DOT = MS / "template" / "agriculture-template.dot"
TEMPLATE = MS / "template" / "land-template.docx"
BACK_MATTER = ("Supplementary Materials:", "Author Contributions:", "Funding:",
               "Institutional Review Board Statement:", "Informed Consent Statement:",
               "Data Availability Statement:", "Acknowledgments:", "Conflicts of Interest:")

CITE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")


def expand(group):
    out = []
    for part in re.split(r"\s*,\s*", group):
        if re.search(r"[–-]", part):
            a, b = map(int, re.split(r"\s*[–-]\s*", part))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def compress(nums):
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"{nums[i]}–{nums[j]}")
        else:
            parts.extend(str(n) for n in nums[i:j + 1])
        i = j + 1
    return "[" + ",".join(parts) + "]"


def renumber(text):
    body, refs = text.split("# References {#refs}")
    old_refs = {}
    for line in refs.strip().splitlines():
        m = re.match(r"\[(\d+)\]\s+(.*)", line.strip())
        if m:
            old_refs[int(m.group(1))] = m.group(2)
    order = []
    for m in CITE.finditer(body):
        for n in expand(m.group(1)):
            if n not in old_refs:
                sys.exit(f"Citation [{n}] has no reference entry")
            if n not in order:
                order.append(n)
    new = {old: i + 1 for i, old in enumerate(order)}
    body = CITE.sub(lambda m: compress([new[n] for n in expand(m.group(1))]), body)
    unused = sorted(set(old_refs) - set(order))
    if unused:
        print(f"Warning: references never cited, left out: {unused}")
    ref_md = "\n".join(f"{i + 1}. {old_refs[old]}" for i, old in enumerate(order))
    return body + "# References\n\n" + ref_md + "\n", new


def make_reference_docx():
    """pandoc's default reference.docx with MDPI-like fonts."""
    from docx import Document
    from docx.shared import Pt, RGBColor
    raw = subprocess.run(["pandoc", "-o", str(REFDOC), "--print-default-data-file",
                          "reference.docx"], capture_output=True)
    if raw.returncode != 0:
        sys.exit(raw.stderr.decode())
    doc = Document(REFDOC)
    sizes = {"Title": 18, "Heading 1": 10, "Heading 2": 10, "Heading 3": 10,
             "Image Caption": 9, "Table Caption": 9, "Caption": 9}
    for st in doc.styles:
        if not hasattr(st, "font") or st.font is None:
            continue
        st.font.name = "Palatino Linotype"
        st.font.color.rgb = RGBColor(0, 0, 0)
        if st.name in sizes:
            st.font.size = Pt(sizes[st.name])
            st.font.bold = st.name.startswith("Heading") or st.name == "Title"
            st.font.italic = st.name == "Heading 3"
        elif st.name in ("Normal", "Body Text", "First Paragraph", "Compact", "Abstract",
                         "Bibliography", "Block Text", "Author"):
            st.font.size = Pt(10)
    doc.save(REFDOC)


def template_docx():
    """The .dot is a Word 2007+ template package; relabel it as a document for pandoc.

    With only the Agriculture template available, the journal name in the running head and
    footer is changed to Land and the journal logo of the first-page header is removed."""
    import zipfile
    src = LAND_DOT if LAND_DOT.exists() else AGRI_DOT
    if not src.exists():
        return None
    if not TEMPLATE.exists() or TEMPLATE.stat().st_mtime < max(src.stat().st_mtime,
                                                               Path(__file__).stat().st_mtime):
        with zipfile.ZipFile(src) as zi, \
                zipfile.ZipFile(TEMPLATE, "w", zipfile.ZIP_DEFLATED) as zo:
            logo_rids = set()
            if src == AGRI_DOT:
                for name in zi.namelist():
                    if re.fullmatch(r"word/_rels/header\d+\.xml\.rels", name):
                        rels = zi.read(name).decode()
                        logo_rids |= {(name, m) for m in re.findall(
                            r'Id="(rId\d+)"[^>]*Target="media/image3\.png"', rels)}
            for it in zi.infolist():
                data = zi.read(it.filename)
                if it.filename == "[Content_Types].xml":
                    data = data.replace(b"wordprocessingml.template.main+xml",
                                        b"wordprocessingml.document.main+xml")
                if src == AGRI_DOT and re.fullmatch(r"word/(header|footer)\d+\.xml", it.filename):
                    data = re.sub(rb">Agriculture(\s*)<", lambda m: b">" + JOURNAL.encode()
                                  + m.group(1) + b"<", data)
                    data = data.replace(b"<w:t>16</w:t>", b"<w:t>15</w:t>")  # Land 2026 = vol. 15
                    rels = f"word/_rels/{it.filename.split('/')[-1]}.rels"
                    for _, rid in [x for x in logo_rids if x[0] == rels]:
                        # Drop the run that holds the Agriculture logo drawing.
                        data = re.sub(rb"<w:r(?:\s[^>]*)?>(?:(?!</w:r>).)*?r:embed=\"" + rid.encode()
                                      + rb"\"(?:(?!</w:r>).)*?</w:r>", b"", data, flags=re.S)
                zo.writestr(it, data)
    return TEMPLATE


def apply_mdpi_styles(path):
    """Give pandoc's paragraphs the MDPI_* styles of the journal template.

    The template does not define pandoc's generic styles (Title, Heading1, BodyText...),
    so paragraphs are classified by the raw style id pandoc wrote, not by style name."""
    from docx import Document
    from docx.oxml.ns import qn
    doc = Document(path)
    by_name = {s.name: s for s in doc.styles}

    def style_has_numbering(name):
        st = by_name.get(name)
        ppr = st.element.pPr if st is not None else None
        return ppr is not None and ppr.numPr is not None

    def use(p, name, drop_numbering=False):
        if name in by_name:
            p.style = by_name[name]
        ppr = p._p.pPr
        if drop_numbering and ppr is not None and ppr.numPr is not None:
            ppr.remove(ppr.numPr)

    def raw_style(p):
        ppr = p._p.pPr
        return ppr.pStyle.val if ppr is not None and ppr.pStyle is not None else ""

    in_refs, in_front, section, prev_sid = False, True, "", ""
    for p in doc.paragraphs:
        sid, text = raw_style(p), p.text.strip()
        has_math = p._p.find(".//" + qn("m:oMath")) is not None
        numbered = p._p.pPr is not None and p._p.pPr.numPr is not None
        after_title = prev_sid == "Title"
        prev_sid = sid
        if sid == "Title":
            use(p, "MDPI_1.2_title")
        elif sid.startswith("Heading"):
            in_front = False
            level = sid[-1]
            if level == "1":
                section = text
                in_refs = text.startswith("References")
            use(p, {"1": "MDPI_2.1_heading1", "2": "MDPI_2.2_heading2",
                    "3": "MDPI_2.3_heading3"}.get(level, "MDPI_2.3_heading3"))
            p.paragraph_format.keep_with_next = True  # no heading alone at a page foot
        elif in_front and text.startswith("Abstract:"):
            use(p, "MDPI_1.7_abstract")
        elif in_front and text.startswith("Keywords:"):
            use(p, "MDPI_1.8_keywords")
        elif in_front and after_title:
            use(p, "MDPI_1.3_authornames")  # first paragraph after the title
        elif in_front:
            use(p, "MDPI_1.6_affiliation")
        elif in_refs:
            use(p, "MDPI_8.1_references",
                drop_numbering=style_has_numbering("MDPI_8.1_references"))
        elif sid in ("ImageCaption", "Caption"):
            use(p, "MDPI_5.1_figure_caption")
            p.paragraph_format.keep_together = True
        elif sid == "CaptionedFigure" or p._p.find(".//" + qn("w:drawing")) is not None:
            use(p, "MDPI_5.2_figure")
            p.paragraph_format.keep_with_next = True  # figure stays with its caption
        elif sid == "TableCaption":
            use(p, "MDPI_4.1_table_caption")
            p.paragraph_format.keep_together = True
            p.paragraph_format.keep_with_next = True  # title stays with the first row
        elif text.startswith(BACK_MATTER):
            use(p, "MDPI_6.2_back_matter")
        elif numbered:
            name = "MDPI_3.7_itemize" if section.startswith("5.") else "MDPI_3.8_bullet"
            use(p, name, drop_numbering=style_has_numbering(name))
        elif has_math and len(text) < 400:
            use(p, "MDPI_3.9_equation")
        else:
            use(p, "MDPI_3.1_text")
    for t in doc.tables:
        sty = t._tbl.tblPr.find(qn("w:tblStyle"))
        if sty is not None:
            sty.set(qn("w:val"), "MDPI41threelinetable")
        for i, row in enumerate(t.rows):
            # A row never splits across pages; the header row repeats if the table breaks,
            # and the first two rows stay together so a table never starts at a page foot.
            trpr = row._tr.get_or_add_trPr()
            for tag in ("w:cantSplit",) + (("w:tblHeader",) if i == 0 else ()):
                if trpr.find(qn(tag)) is None:
                    trpr.append(trpr.makeelement(qn(tag), {}))
            for cell in row.cells:
                for p in cell.paragraphs:
                    use(p, "MDPI_4.2_table_body")
                    if i == 0:
                        p.paragraph_format.keep_with_next = True
    # The template's section carries a right-to-left flag (<w:bidi/>); Word ignores it
    # for left-to-right text, but other readers reverse table columns. Drop it.
    for sect in doc.element.body.iter(qn("w:sectPr")):
        for b in sect.findall(qn("w:bidi")):
            sect.remove(b)
    # Article type line at the very top, as in the template.
    art = doc.paragraphs[0].insert_paragraph_before("Article")
    use(art, "MDPI_1.1_article_type")
    doc.save(path)


def main():
    text = SRC.read_text(encoding="utf-8")
    built, mapping = renumber(text)
    BUILD.write_text(built, encoding="utf-8")
    print("Citation map (draft -> manuscript): " +
          ", ".join(f"{o}->{n}" for o, n in sorted(mapping.items())))
    ref = template_docx()
    if ref is None:
        make_reference_docx()
        ref = REFDOC
    cmd = ["pandoc", str(BUILD), "-o", str(DOCX), "--reference-doc", str(ref),
           "--resource-path", str(MS), "-f", "markdown+tex_math_dollars+superscript+subscript"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=MS)
    if r.returncode != 0:
        sys.exit(r.stderr)
    if r.stderr.strip():
        print(r.stderr.strip())
    if ref == TEMPLATE:
        apply_mdpi_styles(DOCX)
        print(f"Applied the MDPI_* styles of the {JOURNAL} template")
    print(f"Wrote {BUILD.name} and {DOCX.name} ({DOCX.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
