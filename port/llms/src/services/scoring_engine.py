import warnings
import logging

# Suppress warnings from libraries that might be chatty
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class ScoringEngine:
    def __init__(self):
        self._nltk_loaded = False
        self._bert_loaded = False
        self.scorer_rouge = None
        self.scorer_bert = None

    def _ensure_nltk(self):
        if not self._nltk_loaded:
            import nltk
            # Newer NLTK versions require 'punkt_tab' in addition to 'punkt'
            required_packages = ['punkt', 'punkt_tab']
            
            for pkg in required_packages:
                try:
                    nltk.data.find(f'tokenizers/{pkg}')
                except LookupError:
                    print(f"[SCORING] Downloading NLTK resource: {pkg}...")
                    nltk.download(pkg, quiet=True)
            
            self._nltk_loaded = True

    def _ensure_rouge(self):
        if not self.scorer_rouge:
            from rouge_score import rouge_scorer
            # We use ROUGE-L (Longest Common Subsequence) as it's best for prose
            self.scorer_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def _ensure_bert(self):
        if not self._bert_loaded:
            print("[SCORING] Initializing BERTScore model (this may take a moment)...")
            # We perform a dummy import/load here. 
            self._bert_loaded = True

    def calculate_scores(self, response_text: str, reference_text: str, metrics: list[str] | None = None) -> dict:
        """
        Calculates BLEU, ROUGE-L, and/or BERTScore.
        Returns dictionary of floats (0.0 - 1.0).
        When metrics is provided (e.g. ['bleu']), compute only those; otherwise compute all three.
        """
        if not response_text or not reference_text:
            return {}

        scores = {}
        compute_all = metrics is None or len(metrics) == 0
        do_bleu = compute_all or 'bleu' in (metrics or [])
        do_rouge = compute_all or 'rouge' in (metrics or [])
        do_bert = compute_all or 'bert' in (metrics or [])

        # 1. BLEU Score (NLTK)
        if do_bleu:
            try:
                self._ensure_nltk()
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                from nltk.tokenize import word_tokenize

                ref_tokens = word_tokenize(reference_text.lower())
                cand_tokens = word_tokenize(response_text.lower())

                cc = SmoothingFunction()
                bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=cc.method1)
                scores['bleu_score'] = round(bleu, 4)
            except Exception as e:
                print(f"[SCORING] BLEU Error: {e}")
                scores['bleu_score'] = 0.0

        # 2. ROUGE-L (Google Research)
        if do_rouge:
            try:
                self._ensure_rouge()
                rouge_res = self.scorer_rouge.score(reference_text, response_text)
                scores['rouge_score'] = round(rouge_res['rougeL'].fmeasure, 4)
            except Exception as e:
                print(f"[SCORING] ROUGE Error: {e}")
                scores['rouge_score'] = 0.0

        # 3. BERTScore (Contextual Embeddings)
        if do_bert:
            try:
                self._ensure_bert()
                from bert_score import score

                if response_text.strip() and reference_text.strip():
                    # SAFETY FIX: Force device='cpu' to prevent OOM crashes on the GPU
                    # while a Large Language Model is already loaded.
                    P, R, F1 = score(
                        [response_text],
                        [reference_text],
                        lang="en",
                        verbose=False,
                        device='cpu'
                    )
                    scores['bert_score'] = round(F1.mean().item(), 4)
                else:
                    scores['bert_score'] = 0.0
            except Exception as e:
                print(f"[SCORING] BERTScore Error: {e}")
                scores['bert_score'] = 0.0

        return scores