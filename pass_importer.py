"""
MLIR Pass Importer — extracts pass metadata from TableGen definitions.

Supports two modes:
1) regex-based heuristic extraction (fast, no network needed)
2) AI-assisted extraction via local proxy at http://127.0.0.1:8000 (Anthropic protocol)
"""
import json
import gzip
import os
import re
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Tuple


@dataclass
class ImportedPass:
    name: str
    summary: str
    source_dialects: List[str]
    target_dialects: List[str]
    compilation_phase: Optional[str] = None
    predecessor_phase: Optional[str] = None
    generated_side_effects: List[str] = field(default_factory=list)


class AIPassImporter:
    """Imports pass definitions using a local AI proxy (Anthropic protocol)."""

    PROXY_URL = "http://127.0.0.1:8000/v1/messages"
    DEFAULT_MODEL = "deepseek-v4-pro"

    def __init__(self, proxy_url: str = "", model: str = ""):
        self.proxy_url = proxy_url or self.PROXY_URL
        self.model = model or self.DEFAULT_MODEL

    def _call_ai(self, system_prompt: str, user_prompt: str) -> str:
        """Send a request to the local AI proxy and return the text response."""
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.proxy_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept-Encoding": "identity",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw_body = resp.read()
                # Handle gzip-compressed responses
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw_body = gzip.decompress(raw_body)
                body = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"AI proxy request failed: {e}")

        # Extract text from response. The proxy may return content as a list
        # with thinking blocks interspersed — take only "text" type blocks.
        content = body.get("content", "")
        if isinstance(content, list):
            text_blocks = [
                b.get("text", "") for b in content if b.get("type") == "text"
            ]
            content = "\n".join(text_blocks)

        return content

    def import_from_td(
        self, file_path: str, use_ai: bool = False
    ) -> List[ImportedPass]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.import_from_content(content, use_ai=use_ai)

    def import_from_content(
        self, content: str, use_ai: bool = False
    ) -> List[ImportedPass]:
        if use_ai:
            return self._extract_with_ai(content)
        return self._extract_with_regex(content)

    # ---- regex extractor ----

    @staticmethod
    def _extract_with_regex(content: str) -> List[ImportedPass]:
        pattern = re.compile(
            r"def\s+(\w+)\s*:\s*Pass<\s*\"([^\"]+)\"[^>]*>\s*\{(?P<body>.*?)\n\}",
            re.DOTALL,
        )
        imports: List[ImportedPass] = []
        for m in pattern.finditer(content):
            body = m.group("body")
            summary_match = re.search(r'let\s+summary\s*=\s*"([^"]+)"', body)
            summary = summary_match.group(1) if summary_match else ""
            pass_name = m.group(2)
            deps_match = re.search(
                r"let\s+dependentDialects\s*=\s*\[(.*?)\]", body, re.DOTALL
            )
            deps = []
            if deps_match:
                deps = re.findall(
                    r'"([^:"]+)(?:::[^"]+)?"', deps_match.group(1)
                )
            src, tgt = AIPassImporter._infer_dialects(pass_name, summary, deps)
            imports.append(
                ImportedPass(
                    name=pass_name,
                    summary=summary,
                    source_dialects=src,
                    target_dialects=tgt,
                )
            )
        return imports

    @staticmethod
    def _infer_dialects(
        pass_name: str,
        summary: str = "",
        deps: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        text = f"{pass_name} {summary}".lower()
        convert_match = re.search(r"convert-([a-z0-9_]+)-to-([a-z0-9_]+)", text)
        if convert_match:
            return [convert_match.group(1)], [convert_match.group(2)]

        # Try alternative pattern: "tosa-to-linalg", "lower-affine", etc.
        alt_match = re.search(r"([a-z0-9_]+)-to-([a-z0-9_]+)", text)
        if alt_match:
            return [alt_match.group(1)], [alt_match.group(2)]

        src, tgt = [], []
        if deps:
            tgt = [d.lower() for d in deps]
        return src, tgt

    # ---- AI extractor ----

    _AI_SYSTEM_PROMPT = """\
You are an MLIR compiler expert. Extract pass metadata from MLIR TableGen pass \
definitions.
Return a JSON object with a "passes" array. For each pass include:
- name: CLI name (the string in Pass<"...">)
- summary: one-line description
- source_dialects: dialects this pass converts FROM (empty list if not a \
conversion pass)
- target_dialects: dialects this pass converts TO (empty list if not a \
conversion pass)
- compilation_phase: one of "frontend", "optimization", "bufferization", \
"loop-lowering",
  "control-flow-lowering", "arithmetic-lowering", "finalization", "cleanup", \
or null
Output ONLY valid JSON, no markdown fences."""

    def _extract_with_ai(self, content: str) -> List[ImportedPass]:
        # Truncate to avoid token limits
        truncated = content[:15000] if len(content) > 15000 else content
        try:
            raw = self._call_ai(self._AI_SYSTEM_PROMPT, truncated)
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            rows = data.get("passes", data if isinstance(data, list) else [])
            result = []
            for row in rows:
                result.append(
                    ImportedPass(
                        name=row.get("name", ""),
                        summary=row.get("summary", ""),
                        source_dialects=row.get("source_dialects", []),
                        target_dialects=row.get("target_dialects", []),
                        compilation_phase=row.get("compilation_phase"),
                        predecessor_phase=row.get("predecessor_phase"),
                        generated_side_effects=row.get(
                            "generated_side_effects", []
                        ),
                    )
                )
            return result
        except Exception as e:
            print(f"[AI Import] AI extraction failed: {e}, falling back to regex")
            return self._extract_with_regex(content)


def export_passes_to_json(
    imported_passes: List[ImportedPass], output_path: str
):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(p) for p in imported_passes],
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("td_file", help="Path to a .td file")
    ap.add_argument("--ai", action="store_true", help="Use AI extraction")
    ap.add_argument("--json", type=str, help="Export result as JSON")
    args = ap.parse_args()

    importer = AIPassImporter()
    passes = importer.import_from_td(args.td_file, use_ai=args.ai)

    for p in passes:
        print(
            f"  {p.name}: src={p.source_dialects} -> tgt={p.target_dialects}"
            f"  [{p.compilation_phase or '?'}]"
        )

    if args.json:
        export_passes_to_json(passes, args.json)