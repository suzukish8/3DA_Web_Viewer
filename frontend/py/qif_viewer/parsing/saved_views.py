"""Persists ui/viewport.py's named camera "saved views" into the QIF file's
own top-level <Attributes> block, using QIF's own vendor-extension
mechanism (confirmed against real sample files -- the CAD tool that
authored them already stores its own data the same way, e.g.
<AttributeStr name="_kotem.ProblemDimension" value="Spatial3D" />) rather
than a sidecar file. QIF itself has no native concept of a 3D viewer
camera view; this is an app-specific extension, harmless to any other
QIF-consuming tool (which will simply not recognize the attribute name and
ignore it), not part of the standard schema.

Works at the raw XML *text* level, matching how the rest of this app
already treats the QIF source (ui/qif_text_view.py's editable text buffer,
written back out verbatim on File > Save) rather than round-tripping
through a DOM tree, which would reformat/reorder the whole document and
produce a huge, unrelated diff against the user's own file.
"""
from __future__ import annotations

import json
import re
from typing import Optional

SAVED_VIEWS_ATTRIBUTE_NAME = "_qifviewer.SavedViews"

_ATTRIBUTES_BLOCK_RE = re.compile(r'<Attributes\s+n="(\d+)"\s*>(.*?)</Attributes>', re.DOTALL)
_OUR_ATTRIBUTE_RE = re.compile(
    r'[ \t]*<AttributeStr\s+name="' + re.escape(SAVED_VIEWS_ATTRIBUTE_NAME) + r'"\s+value="(.*?)"\s*/>\n?',
    re.DOTALL,
)
_QPID_CLOSE_RE = re.compile(r"</QPId>")
_ROOT_OPEN_TAG_RE = re.compile(r"<QIFDocument\b[^>]*>")


def _escape_xml_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _unescape_xml_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def _find_root_attributes_match(xml_text: str) -> Optional[re.Match]:
    """The document's *own* <Attributes> block -- QIFDocument's direct
    child, holding file-level vendor extensions like _kotem.* -- as
    opposed to any of the many other <Attributes> blocks nested deep
    inside individual Features/Characteristics/etc. elsewhere in the same
    file. Restricting the search to the text before <Header> (confirmed,
    in every real sample file, to be the top-level Attributes block's very
    next sibling) keeps a plain "first <Attributes> anywhere" search from
    matching one of those nested ones instead, in the one case that would
    actually matter: a file with no top-level block of its own at all."""
    boundary = xml_text.find("<Header")
    region_end = boundary if boundary != -1 else min(len(xml_text), 4000)
    return _ATTRIBUTES_BLOCK_RE.search(xml_text, 0, region_end)


def read_saved_views(xml_text: str) -> dict:
    """Returns {} if there's no top-level Attributes block, or no saved-
    views attribute within it, or its value isn't valid JSON (a
    hand-edited or foreign file shouldn't crash the app over this)."""
    match = _find_root_attributes_match(xml_text)
    if match is None:
        return {}
    our_match = _OUR_ATTRIBUTE_RE.search(match.group(2))
    if our_match is None:
        return {}
    try:
        return json.loads(_unescape_xml_attr(our_match.group(1)))
    except (ValueError, TypeError):
        return {}


def write_saved_views(xml_text: str, views: dict) -> str:
    """Returns xml_text with `views` embedded as the saved-views
    AttributeStr inside the document's top-level <Attributes> block --
    updating it in place if already present, adding it (and bumping the
    block's own n="..." count) if the block exists but doesn't have it
    yet, or inserting a brand-new <Attributes> block (right after
    </QPId>, or after the root element's own opening tag if even that
    isn't present) if the document has no top-level Attributes at all.
    `views` empty removes the attribute (and the whole block, if that was
    its only entry) rather than writing out an empty one."""
    encoded = _escape_xml_attr(json.dumps(views, separators=(",", ":")))
    new_attr_line = f'\t\t<AttributeStr name="{SAVED_VIEWS_ATTRIBUTE_NAME}" value="{encoded}" />\n'

    match = _find_root_attributes_match(xml_text)
    if match is None:
        if not views:
            return xml_text
        block = f'<Attributes n="1">\n{new_attr_line}\t</Attributes>\n\t'
        anchor = _QPID_CLOSE_RE.search(xml_text)
        if anchor is not None:
            insert_at = anchor.end()
            return xml_text[:insert_at] + "\n\t" + block.rstrip() + xml_text[insert_at:]
        anchor = _ROOT_OPEN_TAG_RE.search(xml_text)
        if anchor is not None:
            insert_at = anchor.end()
            return xml_text[:insert_at] + "\n\t" + block.rstrip() + xml_text[insert_at:]
        return xml_text  # no recognizable anchor at all -- leave untouched rather than guess

    count = int(match.group(1))
    body = match.group(2)
    existing = _OUR_ATTRIBUTE_RE.search(body)
    if existing is not None:
        if views:
            new_body = _OUR_ATTRIBUTE_RE.sub(new_attr_line, body)
            new_count = count
        else:
            new_body = _OUR_ATTRIBUTE_RE.sub("", body)
            new_count = count - 1
    elif views:
        new_body = body.rstrip() + "\n" + new_attr_line.rstrip("\n") + "\n\t"
        new_count = count + 1
    else:
        return xml_text  # nothing to remove, nothing to add

    if new_count <= 0 and not new_body.strip():
        # That was the block's only entry -- remove the whole (now-empty)
        # block rather than leaving a valid-but-pointless <Attributes n="0">.
        return xml_text[: match.start()] + xml_text[match.end():]
    new_block = f'<Attributes n="{new_count}">{new_body}</Attributes>'
    return xml_text[: match.start()] + new_block + xml_text[match.end():]
