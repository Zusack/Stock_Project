# src/utils/prompt_utils.py
import os
import re


def _get_section_content(full_text: str, section_name: str) -> str:
    """
    Extract content of a ## Section Name block.

    Returns content from the newline after the header until the next ## header.
    """
    pattern = rf"##\s+{re.escape(section_name)}\s*[:]?\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    content = match.group(1)
    # Stop at the first line that starts with ## (next section)
    lines = []
    for line in content.split("\n"):
        if line.strip().startswith("##"):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_tool_definition_json(raw: str) -> str:
    """
    Extract JSON from Tool Definition section.

    Handles:
    - Fenced code block: ```json ... ``` or ``` ... ```
    - Raw JSON (no fence)
    - Empty or whitespace
    """
    if not raw or not raw.strip():
        return ""
    raw = raw.strip()
    m = re.search(r"^```(?:json)?\s*\n(.*?)\n```\s*$", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw


def extract_prompts_from_folder(folder_path):
    """
    Scans a folder for .md files.
    Parses each file for:
    - Prompt Category (New)
    - Prompt
    - Expected Response
    - Assessment
    - Scoring Criterion
    
    The Filename (minus extension) becomes the Prompt Name.
    """
    prompts_data = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Rubric folder not found at {folder_path}")
        return []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".md"):
            continue
            
        # Old 'Category' is now the Prompt Name (from filename)
        prompt_name = filename.replace(".md", "")
        file_path = os.path.join(folder_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # --- Regex Patterns ---
            
            # 1. Prompt Category (Optional, defaults to Ungrouped)
            cat_match = re.search(r"##\s+Prompt\s+Category[:]?\s*\n(.*?)(?=\n##\s|\Z)", full_text, re.DOTALL | re.IGNORECASE)
            category = cat_match.group(1).strip() if cat_match else "Ungrouped"

            # 2. Prompt
            prompt_match = re.search(r"##\s+Prompt[:]?\s*\n(.*?)(?=\n##\s|\Z)", full_text, re.DOTALL | re.IGNORECASE)
            prompt_text = prompt_match.group(1).strip() if prompt_match else ""
            
            # 3. Expected Response
            expected_match = re.search(r"##\s+(?:Correct|Expected)\s+Response[:]?\s*\n(.*?)(?=\n##\s|\Z)", full_text, re.DOTALL | re.IGNORECASE)
            expected_response = expected_match.group(1).strip() if expected_match else ""
            
            # 4. Assessment
            assessment_match = re.search(r"##\s+Assessment[:]?\s*\n(.*?)(?=\n##\s|\Z)", full_text, re.DOTALL | re.IGNORECASE)
            assessment_text = assessment_match.group(1).strip() if assessment_match else ""
            
            # 5. Scoring Criterion
            scoring_match = re.search(r"##\s+Scoring\s+(?:Criterion|Criteria)[:]?\s*\n(.*?)(?=\n##\s|\Z)", full_text, re.DOTALL | re.IGNORECASE)
            scoring_criteria = scoring_match.group(1).strip() if scoring_match else ""

            # 6. Benchmark Type (text, image_comprehension; legacy vision/ocr/table/chart normalized to image_comprehension)
            bt_match = re.search(r"##\s+Benchmark\s+Type[:]?\s*\n(.*?)(?=\n##\s|\Z)", full_text, re.DOTALL | re.IGNORECASE)
            benchmark_type = (bt_match.group(1).strip().lower() if bt_match else "text").replace(" ", "_") or "text"
            if benchmark_type in ("vision", "ocr_english", "ocr_math", "table", "chart"):
                benchmark_type = "image_comprehension"
            if benchmark_type not in ("text", "image_comprehension", "tool_use", "mcp", "standardized"):
                benchmark_type = "text"

            # 6b. Reference Metric (for standardized benchmarks: bleu, rouge, bert)
            ref_metric_raw = _get_section_content(full_text, "Reference Metric")
            reference_metric = ref_metric_raw.strip().lower() if ref_metric_raw else ""
            if reference_metric not in ("bleu", "rouge", "bert"):
                reference_metric = ""

            # 7. Tool Definition (optional JSON for tool_use/mcp; supports fenced ```json blocks)
            # Extract by section: content between ## Tool Definition and next ## header
            tool_raw = _get_section_content(full_text, "Tool Definition")
            tool_definition_json = _extract_tool_definition_json(tool_raw)

            # 8. Tool Script Path (optional .py path for custom tools, under data/tools/)
            # Use _get_section_content to avoid capturing next section (e.g. "## Prompt") when
            # sections are adjacent with no blank line
            tool_script_path = _get_section_content(full_text, "Tool Script Path")

            if prompt_text:
                prompts_data.append({
                    'name': prompt_name,
                    'category': category,
                    'prompt_text': prompt_text,
                    'expected_response': expected_response,
                    'assessment_text': assessment_text,
                    'scoring_criteria': scoring_criteria,
                    'rubric_text': full_text,
                    'benchmark_type': benchmark_type,
                    'reference_metric': reference_metric,
                    'attachment_paths': '[]',
                    'tool_definition_json': tool_definition_json,
                    'tool_script_path': tool_script_path,
                })

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return prompts_data


def build_rubric_markdown(
    name: str,
    category: str,
    prompt_text: str,
    expected_response: str,
    assessment_text: str,
    scoring_criteria: str,
    benchmark_type: str = "text",
    tool_definition_json: str = "",
    tool_script_path: str = "",
    reference_metric: str = "",
) -> str:
    """
    Build full rubric markdown with consistent section order.

    All rubrics use the same section headers for import/export round-trip.
    """
    parts = [
        f"# {name}",
        "",
        "## Prompt Category",
        category,
        "",
        "## Benchmark Type",
        (benchmark_type or "text").strip(),
        "",
        "## Reference Metric",
        (reference_metric or "").strip(),
        "",
        "## Tool Definition",
        "",
    ]
    tool_json = (tool_definition_json or "").strip()
    if tool_json:
        parts.append("```json")
        parts.append(tool_json)
        parts.append("```")
    parts.extend([
        "",
        "## Tool Script Path",
        (tool_script_path or "").strip(),
        "",
        "## Prompt",
        prompt_text,
        "",
        "## Expected Response",
        expected_response,
        "",
        "## Assessment",
        assessment_text,
        "",
        "## Scoring Criterion",
        "",
        scoring_criteria,
    ])
    return "\n".join(parts)