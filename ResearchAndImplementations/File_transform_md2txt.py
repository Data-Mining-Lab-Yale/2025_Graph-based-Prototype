import os
import sys
import glob

# Optional: 'pip install pypandoc'
import pypandoc

# ---- config ----
DEFAULT_INPUTS = ["(Rimi)Shared_lab_folder_README.md"]  # change or pass via CLI
TARGETS = ["txt", "docx"]  # any of: txt, docx, pdf, html, odt, rtf, pptx, etc.

# Map friendly names to Pandoc's official 'to' formats
FORMAT_MAP = {
    "txt": "plain",
    "text": "plain",
    "docx": "docx",
    "pdf": "pdf",
    "html": "html5",
    "odt": "odt",
    "rtf": "rtf",
    "pptx": "pptx",
    "md": "gfm",           # GitHub-flavored markdown
    "markdown": "gfm",
    "latex": "latex",
}

def ensure_pandoc():
    try:
        pypandoc.get_pandoc_version()
        return True
    except OSError:
        print("Pandoc not found, downloading via pypandoc...", flush=True)
        try:
            pypandoc.download_pandoc()
            pypandoc.get_pandoc_version()
            return True
        except Exception as e:
            print(f"Auto-download failed: {e}")
            return False

def to_pandoc_format(fmt):
    if fmt not in FORMAT_MAP:
        raise RuntimeError(
            f"Unknown output format '{fmt}'. "
            f"Use one of: {', '.join(sorted(FORMAT_MAP.keys()))}"
        )
    return FORMAT_MAP[fmt]

def output_path(input_file, fmt):
    base, _ = os.path.splitext(input_file)
    ext = ".txt" if fmt in ("txt", "text") else f".{fmt}"
    return base + ext

def naive_md_to_txt(md_text):
    """
    Very simple fallback: strips common MD markers.
    Not perfect, but readable if Pandoc is unavailable.
    """
    import re
    text = md_text

    # Remove code fences
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # fenced blocks
    text = re.sub(r"`([^`]*)`", r"\1", text)          # inline code

    # Headings: remove leading #'s
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)

    # Links/images: [text](url) -> text
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links

    # Bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)

    # Blockquotes
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.M)

    # Lists: keep bullets but normalize
    text = re.sub(r"^\s*[-*+]\s+", "- ", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "- ", text, flags=re.M)

    # Horizontal rules
    text = re.sub(r"^\s*([-*_]\s*){3,}\s*$", "", text, flags=re.M)

    # Excess spaces
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip() + "\n"

def convert_one(input_file, out_fmt):
    pandoc_fmt = to_pandoc_format(out_fmt)
    out_path = output_path(input_file, out_fmt)

    # Try with Pandoc
    try:
        pypandoc.convert_file(
            input_file,
            to=pandoc_fmt,
            format="gfm",           # source is GitHub-flavored markdown
            outputfile=out_path,
            extra_args=["--standalone"]
        )
        print(f"✓ {input_file} -> {out_path} [{pandoc_fmt}]")
        return
    except Exception as e:
        # If requested txt and Pandoc failed, use naive fallback
        if out_fmt in ("txt", "text"):
            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    md = f.read()
                txt = naive_md_to_txt(md)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(txt)
                print(f"✓ {input_file} -> {out_path} [naive fallback]")
                return
            except Exception as e2:
                print(f"Failed naive fallback for {input_file}: {e2}")
        # Otherwise raise original
        raise

def convert_many(inputs, targets):
    # Expand globs and filter existing files
    files = []
    for pattern in inputs:
        files.extend(glob.glob(pattern))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("No input files found. Check your paths or patterns.")
        return

    have_pandoc = ensure_pandoc()

    if not have_pandoc:
        print("Pandoc unavailable. Will use naive fallback for txt; other formats will likely fail.")

    for f in files:
        for t in targets:
            try:
                convert_one(f, t)
            except Exception as e:
                print(f"✗ Failed {f} -> {t}: {e}")

if __name__ == "__main__":
    # CLI usage examples:
    #   python File_transform_md2txt.py "(Rimi)Shared_lab_folder_README.md"
    #   python File_transform_md2txt.py "*.md" txt docx
    args = sys.argv[1:]
    if not args:
        inputs = DEFAULT_INPUTS
        targets = TARGETS
    else:
        # Separate inputs and targets by checking for known formats
        known = set(FORMAT_MAP.keys())
        inputs = [a for a in args if a.lower() not in known]
        fmts = [a.lower() for a in args if a.lower() in known]
        targets = fmts if fmts else TARGETS
        if not inputs:
            inputs = DEFAULT_INPUTS
    convert_many(inputs, targets)
