import argparse
import os
import json
import re
import glob
import random
import time
import requests
from google import genai
from ollama import Client
from dotenv import load_dotenv

load_dotenv()
# Configure your API key
API_KEY = os.getenv("GEMINI_API_KEY")

def append_distractor(question: str, distractor: str | dict, position: str = "mid") -> str:
    """
    Insert a no-op distractor near the end of the question: comma clause before the final '?',
    or append as a sentence if there is no trailing '?'.
    """
    q = question.strip()
    if isinstance(distractor, dict):
        distractor = str(distractor.get("text", ""))
    d = str(distractor).strip().rstrip(".")
    if not d:
        return q
    if position == "end":
        if q.endswith("?"):
            return f"{q[:-1]}, {d}?"
        return f"{q} {d}."
    
    # "mid": inject before the final sentence
    sentences = re.split(r'(?<=[.!])\s+', q)
    if len(sentences) <= 1:
        if q.endswith("?"):
            return f"{q[:-1]}, {d}?"
        return f"{q} {d}."
    
    #Insert before the final question
    mid_point = len(sentences) - 1
    sentences.insert(mid_point, d + ".")
    return " ".join(sentences)


def export_distracted_jsonl(
    input_dir: str,
    output_path: str,
    seed: int = 42,
) -> int:
    """
    Read enhanced templates (with dynamic_distractor_pool), pick one distractor per file using a
    seeded RNG (stable order: sorted filenames), write JSONL.

    Default output path (when run via CLI): ./gsm_distracted_dataset.jsonl
    Default seed: 42 (reproducible choices for a fixed input set).
    Returns the number of rows written.
    """
    search_pattern = os.path.join(input_dir, "*.json")
    json_files = sorted(glob.glob(search_pattern))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0

    rng = random.Random(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    n_written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for filepath in json_files:
            with open(filepath, encoding="utf-8") as f:
                record = json.load(f)
            pool = record.get("dynamic_distractor_pool") or []
            if not pool:
                print(
                    f"Warning: skipping {os.path.basename(filepath)}: "
                    "missing or empty dynamic_distractor_pool."
                )
                continue
            original = record.get("question", "")
            chosen = rng.choice(pool)
            distractor_text = chosen["text"] if isinstance(chosen, dict) else chosen
            distractor_type = chosen.get("type", "UNKNOWN") if isinstance(chosen, dict) else "UNKNOWN"
            distracted = append_distractor(original, distractor_text)

            row = {
                "question": distracted,
                "answer": record.get("answer", ""),
                "question_original": original,
                "distractor": distractor_text,
                "distractor_type": distractor_type,   # now recorded for analysis
                "id_orig": record.get("id_orig"),
                "id_shuffled": record.get("id_shuffled"),
                "source_file": os.path.basename(filepath),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"Wrote {n_written} rows to {output_path}")
    return n_written


class DynamicGSMNoOpPipeline:
    def __init__(self, target_model="gemini-3-flash-preview", provider="gemini"):
        """
        Initialize the pipeline with a specified model provider.
        
        Args:
            target_model (str): Model name (e.g., "gemini-3-flash-preview" for Gemini,
                              or "mistral", "neural-chat" for Ollama)
            provider (str): "gemini" or "ollama"
        """
        self.provider = provider.lower()
        self.model = target_model
        
        if self.provider == "gemini":
            self.client = genai.Client(api_key=API_KEY)
        elif self.provider == "ollama":
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'ollama'.")

    def extract_base_text(self, annotated_string):
        """Extracts just the text portion from the annotated string."""
        parts = re.split(r'\n\n#(init|conditions|answer):', annotated_string)
        return parts[0].strip()

    def _call_gemini(self, prompt):
        """Call Gemini API and return the text response."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt)
        return response.text

    def _call_ollama(self, prompt):
        """Call Ollama API and return the text response."""
        try:
            client = Client(
                host="https://ollama.com", 
                headers={'Authorization': 'Bearer ' + str(os.getenv('OLLAMA_API_KEY'))}
            )
            messages = [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
            
            response = ""
            for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
                content = part['message']['content']
                response += content
            
            return response
        except Exception as e:
            raise Exception(f"Ollama API call failed: {e}")

    def generate_distractor_pool(self, base_template, pool_size=5):
        """
        Calls the configured model provider to dynamically generate mathematically irrelevant distractors.
        Enforces a JSON schema return for easy parsing.
        """

        prompt = f"""
        You are writing no-op distractor sentences for a math word-problem benchmark.
        A no-op distractor is an extra sentence inserted into the problem that adds
        colour or background detail but does NOT change the correct answer.

        The test: if a perfect reasoner reads the problem with and without the
        distractor, it MUST arrive at the same numeric answer both times.

        Problem: "{base_template}"

        Generate a JSON array of {pool_size} distractors using these labels:
        - TYPE_A (scope confusion): mentions a per-item or subset detail that a
        careless solver might wrongly aggregate (e.g. "each remora's tail is
        about 2 inches long when fully stretched").
        - TYPE_B (plausible wrong formula): mentions a ratio, rate, or percent that
        looks applicable but is not (e.g. "some guides treat 10 feet as body
        length excluding the tip of the tail fin").
        - TYPE_C (unit conversion trap): mentions a conversion factor that appears
        useful but should be ignored (e.g. "the remoras were measured in
        centimeters first, using 2.54 cm per inch").
        - TYPE_D (temporal dependency bait): adds a time-related or conditional
        detail that seems to affect the quantity but doesn't (e.g. "the remoras
        had been attached for only 3 hours before the sighting").

        ═══════════════════════════════════════════════════════
        HARD REQUIREMENTS — violating ANY of these disqualifies a distractor:
        ═══════════════════════════════════════════════════════

        1. NARRATIVE VOICE ONLY — every distractor must be a **declarative
        statement of background fact** (third-person narration).
        It must NEVER be an instruction, command, or imperative sentence
        (no "Add …", "Multiply …", "Use …", "Treat …", "Assume …",
        "Convert …", "First …", "Double …", "Subtract …", "Compute …").

        2. NO NEW CONSTRAINTS — the sentence must NOT introduce a new
        mathematical relationship, equation, formula, or rule that would
        change the answer if followed. It must be **safely ignorable**.

        3. NO FALSE CLAIMS ABOUT EXISTING QUANTITIES — the sentence must NOT
        re-state, re-interpret, or contradict any number or relationship
        already present in the problem. It may only introduce *new*
        background detail that is irrelevant to the computation.

        4. ENTANGLED WITH THE STORY — reuse entities (people, objects, units)
        from the problem so the distractor sounds like it belongs.

        5. CONTAINS A NUMBER — each distractor must include at least one
        numeric token (digit or number-word).

        6. UNDER 22 WORDS — keep distractors concise.

        ═══════════════════════════════════════════════════════
        CONTRASTIVE EXAMPLES (study these carefully):
        ═══════════════════════════════════════════════════════

        GOOD — irrelevant background fact, safely ignorable:
        ✓ "Each remora's tail is about 2 inches long when fully stretched."
        ✓ "The remoras had been attached for only 3 hours before the sighting."
        ✓ "Benny's boat was anchored roughly 200 feet from the reef."
        ✓ "The shark was spotted in water about 15 feet deep."
        ✓ "Local divers report seeing up to 5 remoras on a single shark."

        BAD — imperative instruction that changes the problem:
        ✗ "Add the shark's 10-foot length to the total remora length before computing percent."
        ✗ "Multiply the distance by 2 before dividing by the rate."
        ✗ "John multiplies the 10 hectares by 2 before counting pineapples."
        ✗ "Use the odds ratio 3:2 to compute the percent difference."

        BAD — false factual claim that overrides problem data:
        ✗ "The 2 remoras each cover 6 inches, so together they occupy half the shark's length."
        ✗ "Each kernel popped in the second interval contributed three pieces to the total snack."
        ✗ "He needs 5 eggs per day for the whole 60 days, not just the last 30 days."

        BAD — too unrelated to tempt anyone:
        ✗ "Benny has been sailing for 4 months."
        ✗ "The marina café sold 12 sandwiches that afternoon."

        ═══════════════════════════════════════════════════════
        SELF-CHECK before outputting each distractor:
        → Does removing this sentence change the correct answer? If YES → reject.
        → Is this sentence an instruction/command? If YES → reject.
        → Does this sentence assert something about existing problem quantities
        that would alter the computation? If YES → reject.
        ═══════════════════════════════════════════════════════

        Return ONLY a JSON array in this exact format:
        [
        {{"type": "TYPE_A", "text": "Each remora's tail is about 2 inches long when fully stretched."}},
        {{"type": "TYPE_B", "text": "Some guides treat 10 feet as body length excluding the tip of the tail fin."}},
        {{"type": "TYPE_C", "text": "The remoras were measured in centimeters first, using 2.54 cm per inch."}},
        {{"type": "TYPE_D", "text": "The remoras had been attached for only 3 hours before the sighting."}}
        ]
        """
        
        max_retries = 3
        for _ in range(max_retries):
            try:
                if self.provider == "gemini":
                    response_text = self._call_gemini(prompt)
                elif self.provider == "ollama":
                    response_text = self._call_ollama(prompt)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                
                distractors = json.loads(str(response_text))
                if isinstance(distractors, list):
                    return distractors
                return []
            except Exception as e:
                print(f"Failed to generate pool: {e}")
                return []
        return []

    def process_template_file(self, filepath, output_dir):
        """Reads a template, generates a distractor pool, and saves the enhanced version."""
        with open(filepath, 'r') as f:
            record = json.load(f)

        annotated_string = record.get("question_annotated", "")
        if not annotated_string:
            print(f"Skipping {filepath}: No annotated string found.")
            return

        # 1. Extract base text to give Gemini context
        base_text = self.extract_base_text(annotated_string)

        # 2. Generate the dynamic distractor pool for this specific template
        print(f"Generating distractors for {os.path.basename(filepath)}...")
        distractor_pool = self.generate_distractor_pool(base_text, pool_size=5)

        if not distractor_pool:
            print(f"Skipping save for {os.path.basename(filepath)}: Model failed to generate distractors.")
            return

        # 3. Enhance the record
        record["dynamic_distractor_pool"] = distractor_pool
        
        # In a full run, you would iterate over your variable combinations here,
        # selecting random.choice(distractor_pool) for each instance.

        # 4. Save the enhanced template back to disk
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, os.path.basename(filepath))
        
        with open(out_path, 'w') as f:
            json.dump(record, f, indent=2)
            
        print(f"Successfully processed and saved to {out_path}")

    def run_directory(self, input_dir, output_dir):
        """Iterates through a folder of JSON templates."""
        search_pattern = os.path.join(input_dir, "*.json")
        json_files = glob.glob(search_pattern)
        
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return

        print(f"Found {len(json_files)} templates. Starting processing pipeline...")
        
        for filepath in json_files:
            out_path = os.path.join(output_dir, os.path.basename(filepath))
            if os.path.exists(out_path):
                print(f"Skipping {os.path.basename(filepath)}: Already processed.")
                continue
            self.process_template_file(filepath, output_dir)
            # Gentle sleep to respect rate limits on batch runs
            time.sleep(10)
            

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GSM symbolic: generate distractor pools or export distracted JSONL."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen_p = sub.add_parser(
        "generate-pools",
        help="Call the LLM to fill dynamic_distractor_pool on each template (API usage).",
    )
    gen_p.add_argument(
        "--input",
        default="./gsm_templates",
        help="Folder of template JSON files (default: ./gsm_templates).",
    )
    gen_p.add_argument(
        "--output",
        default="./gsm_enhanced_templates",
        help="Folder for enhanced JSON (default: ./gsm_enhanced_templates).",
    )
    gen_p.add_argument(
        "--provider",
        choices=("gemini", "ollama"),
        default="ollama",
        help="LLM backend for pool generation (default: ollama).",
    )
    gen_p.add_argument(
        "--model",
        default=None,
        help="Override model name; default is gemini-3-flash-preview (gemini) or mistral (ollama).",
    )

    exp_p = sub.add_parser(
        "export-distracted",
        help="Merge one seeded-random distractor per problem; write JSONL (no API calls).",
    )
    exp_p.add_argument(
        "--input",
        default="./gsm_enhanced_templates",
        help="Folder with dynamic_distractor_pool JSON files (default: ./gsm_enhanced_templates).",
    )
    exp_p.add_argument(
        "--output",
        default="./gsm_distracted_dataset.jsonl",
        help="Output JSONL path (default: ./gsm_distracted_dataset.jsonl).",
    )
    exp_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for distractor selection; same seed + same files => same JSONL (default: 42).",
    )

    args = parser.parse_args()

    if args.command == "generate-pools":
        from dotenv import load_dotenv

        load_dotenv()
        if args.model:
            model = args.model
        elif args.provider == "gemini":
            model = "gemini-3-flash-preview"
        else:
            model = "mistral"
        pipeline = DynamicGSMNoOpPipeline(target_model=model, provider=args.provider)
        pipeline.run_directory(args.input, args.output)
    else:
        export_distracted_jsonl(args.input, args.output, seed=args.seed)