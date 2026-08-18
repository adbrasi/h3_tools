"""Reference parsing, naming, ordinal assignment and @mention substitution.

Stdlib-only on purpose: this module is the testable core of the pack and must
import (and run under pytest) without ComfyUI present.

The ordinal rules mirror the native MiniMaxH3ReferenceToVideo presentation
exactly (see SPEC.md §1.1/§5.3): images first, then videos — a video with
use_soundtrack consumes the next <Audio j> ordinal right before taking its
<Video k> — then standalone audios continue the audio counter.
"""

import json
import re
from dataclasses import dataclass, field

MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3

# an @ not glued to a preceding word character, capturing the name as typed
MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z0-9_]{1,64})")
NAME_RE = re.compile(r"^[a-z0-9_]{1,64}$")
NAME_MAX = 64


class RefError(ValueError):
    """User-facing validation error; the message is shown as-is."""


@dataclass
class Ref:
    name: str
    type: str  # "image" | "video" | "audio"
    file: str  # annotated path relative to the input directory
    use_soundtrack: bool = False
    tag: str = field(default="")            # "<Picture 1>" / "<Video 2>" / "<Audio 3>"
    soundtrack_tag: str = field(default="") # "<Audio j>" when use_soundtrack


def derive_name(filename: str) -> str:
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    name = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    if not name or name[0].isdigit():
        name = "m_" + name
    return name[:NAME_MAX]


def parse_references(raw: str) -> list:
    """Parse and validate the references JSON widget (SPEC.md §4).

    Returns a list of Ref with names and ordinal tags assigned.
    Raises RefError with an actionable message on any user-data problem.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise RefError("references is not valid JSON: %s" % e) from e
    if not isinstance(data, list):
        raise RefError("references must be a JSON array of reference objects")

    refs = []
    explicit = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise RefError("references[%d] must be an object" % i)
        rtype = item.get("type")
        if rtype not in ("image", "video", "audio"):
            raise RefError(
                'references[%d].type must be "image", "video" or "audio" (got %r)'
                % (i, rtype))
        file = item.get("file")
        if not isinstance(file, str) or not file:
            raise RefError(
                "references[%d].file is required (path relative to the input directory)" % i)
        use_soundtrack = item.get("use_soundtrack", False)
        if not isinstance(use_soundtrack, bool):
            raise RefError("references[%d].use_soundtrack must be true or false" % i)
        if use_soundtrack and rtype != "video":
            raise RefError("references[%d]: use_soundtrack is only valid on videos" % i)
        name = item.get("name")
        if name is not None:
            if not isinstance(name, str) or not NAME_RE.match(name):
                raise RefError(
                    "references[%d].name %r is invalid (must match ^[a-z0-9_]{1,64}$)"
                    % (i, name))
            if name in explicit:
                raise RefError('duplicate reference name "%s"' % name)
            explicit.add(name)
        refs.append(Ref(name=name or "", type=rtype, file=file,
                        use_soundtrack=use_soundtrack))

    counts = {"image": 0, "video": 0, "audio": 0}
    for r in refs:
        counts[r.type] += 1
    for rtype, cap in (("image", MAX_IMAGES), ("video", MAX_VIDEOS), ("audio", MAX_AUDIOS)):
        if counts[rtype] > cap:
            raise RefError(
                "too many %s references: %d (MiniMax H3 accepts at most %d)"
                % (rtype, counts[rtype], cap))

    # derived names only ever get collision suffixes; explicit duplicates errored above
    taken = set(explicit)
    for r in refs:
        if r.name:
            continue
        base = derive_name(r.file)
        name, n = base, 2
        while name in taken:
            suffix = "_%d" % n
            name = base[:NAME_MAX - len(suffix)] + suffix
            n += 1
        r.name = name
        taken.add(name)

    return assign_ordinals(refs)


def assign_ordinals(refs: list) -> list:
    """Fill tag/soundtrack_tag mirroring the native presentation order."""
    pictures = videos = audios = 0
    for r in refs:
        if r.type == "image":
            pictures += 1
            r.tag = "<Picture %d>" % pictures
    for r in refs:
        if r.type == "video":
            if r.use_soundtrack:
                audios += 1
                r.soundtrack_tag = "<Audio %d>" % audios
            videos += 1
            r.tag = "<Video %d>" % videos
    for r in refs:
        if r.type == "audio":
            audios += 1
            r.tag = "<Audio %d>" % audios
    return refs


def find_mentions(text: str) -> list:
    return [m.group(1) for m in MENTION_RE.finditer(text)]


def unknown_mentions(text: str, refs: list) -> list:
    """Mentioned names that resolve to no reference (deduped, as first typed)."""
    known = {r.name for r in refs}
    seen, out = set(), []
    for name in find_mentions(text):
        low = name.lower()
        if low not in known and low not in seen:
            seen.add(low)
            out.append(low)
    return out


def substitute_mentions(text: str, refs: list) -> str:
    """Replace resolved @name mentions with their tags; leave everything else."""
    lookup = {r.name: r.tag for r in refs}
    return MENTION_RE.sub(
        lambda m: lookup.get(m.group(1).lower(), m.group(0)), text)


def provenance_lines(refs: list) -> list:
    return ["%s is the synchronized audio track of %s." % (r.soundtrack_tag, r.tag)
            for r in refs if r.type == "video" and r.use_soundtrack]


def build_final_prompt(text: str, refs: list) -> str:
    body = substitute_mentions(text, refs)
    lines = provenance_lines(refs)
    if lines:
        return "\n".join(lines) + "\n\n" + body
    return body


def strip_unknown_mentions(text: str, refs: list):
    """Sanitize LLM output: drop @tokens that match no reference.

    Returns (clean_text, removed_tokens). Runs of spaces left by a removal are
    collapsed — the text is LLM output, not user formatting.
    """
    known = {r.name for r in refs}
    removed = []

    def repl(m):
        if m.group(1).lower() in known:
            return m.group(0)
        removed.append(m.group(0))
        return ""

    out = MENTION_RE.sub(repl, text)
    if removed:
        out = re.sub(r"[ \t]{2,}", " ", out)
    return out, removed


def unmentioned_refs(text: str, refs: list) -> list:
    """Names of references never mentioned in the prompt (they still reach the model)."""
    mentioned = {m.lower() for m in find_mentions(text)}
    return [r.name for r in refs if r.name not in mentioned]
