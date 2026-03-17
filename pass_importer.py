import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional
from urllib import request


@dataclass
class ImportedPass:
    name: str
    summary: str
    source_dialects: List[str]
    target_dialects: List[str]


class PassTableGenImporter:
    """
    从 TableGen 文本中抽取 pass 元信息。
    支持两种模式：
    1) 纯 regex 启发式（默认）
    2) 借助 LLM 的结构化提取（设置 api_key 与 model）
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def import_from_file(self, file_path: str, use_ai: bool = False) -> List[ImportedPass]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.import_from_content(content, use_ai=use_ai)

    def import_from_content(self, content: str, use_ai: bool = False) -> List[ImportedPass]:
        if use_ai and self.api_key:
            ai_result = self._extract_with_ai(content)
            if ai_result:
                return ai_result
        return self._extract_with_regex(content)

    def _extract_with_regex(self, content: str) -> List[ImportedPass]:
        # 简单匹配：def XXX : Pass<"pass-name", "op"> { let summary = "..."; }
        pattern = re.compile(
            r"def\s+(\w+)\s*:\s*Pass<\s*\"([^\"]+)\"[^>]*>\s*\{(?P<body>.*?)\}",
            re.DOTALL,
        )
        imports: List[ImportedPass] = []
        for m in pattern.finditer(content):
            body = m.group("body")
            summary_match = re.search(r'let\s+summary\s*=\s*"([^"]+)"', body)
            summary = summary_match.group(1) if summary_match else ""
            pass_name = m.group(2)
            src, tgt = self._infer_dialects(pass_name, summary)
            imports.append(
                ImportedPass(name=pass_name, summary=summary, source_dialects=src, target_dialects=tgt)
            )
        return imports

    def _infer_dialects(self, pass_name: str, summary: str):
        text = f"{pass_name} {summary}".lower()
        src, tgt = [], []
        convert_match = re.search(r"convert-([a-z0-9_]+)-to-([a-z0-9_]+)", text)
        if convert_match:
            src.append(convert_match.group(1))
            tgt.append(convert_match.group(2))
        return src, tgt

    def _extract_with_ai(self, content: str) -> List[ImportedPass]:
        prompt = (
            "Read this MLIR TableGen snippet and extract pass metadata as JSON list with keys: "
            "name, summary, source_dialects, target_dialects.\n\n"
            f"{content[:12000]}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You extract MLIR pass metadata."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            rows = parsed.get("passes", parsed if isinstance(parsed, list) else [])
            return [ImportedPass(**row) for row in rows]
        except Exception:
            return []


def export_passes_to_json(imported_passes: List[ImportedPass], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in imported_passes], f, ensure_ascii=False, indent=2)
