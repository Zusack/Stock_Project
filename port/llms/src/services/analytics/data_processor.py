"""
Data processing and transformation for analytics.
Handles calculations, aggregations, and statistical analysis.
"""
import statistics
import math
import warnings

from src.utils.model_display_label import raw_data_model_key

from . import model_analyzer_stats as _model_analyzer_stats
from . import prompt_analyzer_stats as _prompt_analyzer_stats

# --- IMPORT CHECK ---
try:
    from nltk.metrics import agreement
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


def _load_time_score(mean_ttft_sec: float) -> float:
    """Map mean TTFT (seconds) to 0..10 for stacked bar; lower latency = higher score."""
    if mean_ttft_sec <= 0:
        return 0.0
    cap = 2.5
    return min(10.0, max(0.0, 10.0 * (1.0 - min(float(mean_ttft_sec), cap) / cap)))


_TOKEN_EFFICIENCY_CATEGORY = "Token Efficiency (Q-Adj)"


def _clamp_0_10(value: float) -> float:
    return min(10.0, max(0.0, float(value)))


def _extract_tokens(row: dict) -> float | None:
    """Return generated token count from either reader or test row naming."""
    value = row.get("Tokens")
    if value is None:
        value = row.get("tokens_generated")
    if value is None:
        return None
    try:
        tokens = float(value)
    except (TypeError, ValueError):
        return None
    if tokens <= 0:
        return None
    return tokens


class AnalyticsDataProcessor:
    """Processes and transforms raw data for analytics"""
    
    def get_leaderboard_data(
        self,
        raw_data,
        include_speed=True,
        include_win_rate=True,
        include_load_time=True,
        include_token_efficiency=True,
    ):
        """
        Process raw data into leaderboard format.
        
        Args:
            raw_data: List of dictionaries from get_raw_data()
            include_speed: Include speed metrics
            include_win_rate: Include win rate metrics
            include_load_time: Include load latency (mean TTFT) as an augmented bar segment
            include_token_efficiency: Include quality-adjusted token efficiency
            
        Returns:
            dict with ``model_data``, ``all_categories`` (prompts then augmented, flat),
            ``prompt_categories``, ``augmented_categories``, and ``category_prompt_counts``.
        """
        if not raw_data:
            return None
        
        # 1. Group by Model
        models = {}
        category_averages = {}
        
        # First pass: Aggregate scores per Category
        for r in raw_data:
            cat = r['Category']
            s = r['Score']
            if s is None or s < 0:
                continue
            
            if cat not in category_averages:
                category_averages[cat] = []
            category_averages[cat].append(s)
        
        category_means = {cat: statistics.mean(scores) for cat, scores in category_averages.items()}
        all_display_categories = set(category_means.keys())

        # Prompt-local token efficiency storage:
        # We only compare token counts within the same prompt to avoid invalid cross-prompt comparisons.
        # See docs/token_efficiency_leaderboard_metric.md for formula details and rationale.
        prompt_token_entries: dict[tuple[str, str], list[dict]] = {}

        # Distinct prompt names per category (valid scores only) — for UI e.g. Leaderboard legend.
        prompts_per_category: dict[str, set[str]] = {}
        for r in raw_data:
            cat = r.get("Category")
            if not cat:
                continue
            score = r.get("Score")
            if score is None or score < 0:
                continue
            label = (r.get("PromptName") or "").strip() or "—"
            prompts_per_category.setdefault(cat, set()).add(label)
        category_prompt_counts = {c: len(prompts_per_category.get(c, ())) for c in sorted(all_display_categories)}
        
        # Second Pass: Model Metrics (group by display model name only).
        # Context-specific rows are aggregated together so analytics views do not
        # show duplicate model entries like "Model" and "Model (8192)".
        for r in raw_data:
            m = raw_data_model_key(r)
            if m not in models:
                models[m] = {
                    'raw_scores': [],
                    'speeds': [],
                    'ttfts': [],
                    'wins': 0,
                    'total_comparisons': 0,
                    'categories': {}
                }
            
            if r['GenTPS']:
                models[m]['speeds'].append(r['GenTPS'])
            ttft = r.get('TTFT')
            if ttft is not None and float(ttft) > 0:
                models[m]['ttfts'].append(float(ttft))
            
            score = r['Score']
            if score is not None and score >= 0:
                models[m]['raw_scores'].append(score)
                
                cat = r['Category']
                prompt_name = r['PromptName']
                
                if cat not in models[m]['categories']:
                    models[m]['categories'][cat] = {'scores': [], 'details': []}
                
                models[m]['categories'][cat]['scores'].append(score)
                models[m]['categories'][cat]['details'].append(f"{prompt_name}: {score}")
                
                # Win Rate calculation based on beating the CATEGORY average
                if cat in category_means:
                    models[m]['total_comparisons'] += 1
                    if score > category_means[cat]:
                        models[m]['wins'] += 1

                prompt_name = (r.get("PromptName") or "").strip() or "—"
                tokens = _extract_tokens(r)
                if tokens is not None:
                    prompt_key = (cat, prompt_name)
                    prompt_token_entries.setdefault(prompt_key, []).append({
                        "model": m,
                        "category": cat,
                        "prompt": prompt_name,
                        "tokens": tokens,
                        "score": float(score),
                    })

        # Per-model token efficiency (hierarchical):
        # response -> model+prompt mean -> model+category mean -> model mean across categories.
        # This prevents categories/prompts with more runs from dominating.
        model_category_prompt_scores: dict[str, dict[str, dict[str, list[float]]]] = {}
        model_category_prompt_components: dict[str, dict[str, dict[str, list[tuple[float, float, float]]]]] = {}
        for prompt_entries in prompt_token_entries.values():
            if not prompt_entries:
                continue
            token_values = [entry["tokens"] for entry in prompt_entries]
            t_min = min(token_values)
            t_max = max(token_values)
            same_tokens = math.isclose(t_max, t_min)
            for entry in prompt_entries:
                token_eff = 5.0 if same_tokens else (10.0 * (t_max - entry["tokens"]) / (t_max - t_min))
                token_eff = _clamp_0_10(token_eff)
                quality_factor = _clamp_0_10(entry["score"]) / 10.0
                q_adj_eff = _clamp_0_10(token_eff * quality_factor)

                m = entry["model"]
                cat = entry["category"]
                prompt_name = entry["prompt"]
                model_category_prompt_scores.setdefault(m, {}).setdefault(cat, {}).setdefault(prompt_name, []).append(q_adj_eff)
                model_category_prompt_components.setdefault(m, {}).setdefault(cat, {}).setdefault(prompt_name, []).append(
                    (token_eff, quality_factor, q_adj_eff)
                )
        
        # Final Calculation
        prompt_categories_sorted = sorted(list(all_display_categories))
        leaderboard_data = {
            "model_data": {},
            "all_categories": list(prompt_categories_sorted),
            "prompt_categories": list(prompt_categories_sorted),
            "augmented_categories": [],
            "category_prompt_counts": category_prompt_counts,
        }
        
        for display_name, data in models.items():
            # Skip empty groups that have neither quality nor speed data.
            if not data['raw_scores'] and not data['speeds']:
                continue
            avg_qual = statistics.mean(data['raw_scores']) if data['raw_scores'] else 0
            avg_speed = statistics.mean(data['speeds']) if data['speeds'] else 0
            ttfts = data.get('ttfts') or []
            avg_ttft = statistics.mean(ttfts) if ttfts else 0.0
            
            win_rate_pct = 0
            if data['total_comparisons'] > 0:
                win_rate_pct = (data['wins'] / data['total_comparisons']) * 100

            token_by_cat = {}
            token_component_lines = []
            model_token_prompt_count = 0
            cat_prompt_scores = model_category_prompt_scores.get(display_name, {})
            cat_prompt_components = model_category_prompt_components.get(display_name, {})
            for cat, prompt_scores in cat_prompt_scores.items():
                prompt_means = [statistics.mean(values) for values in prompt_scores.values() if values]
                if not prompt_means:
                    continue
                token_by_cat[cat] = statistics.mean(prompt_means)
                model_token_prompt_count += len(prompt_means)

                prompt_components = cat_prompt_components.get(cat, {})
                cat_token_raw = []
                cat_quality = []
                for values in prompt_components.values():
                    for token_raw, q_factor, _q_adj in values:
                        cat_token_raw.append(token_raw)
                        cat_quality.append(q_factor)
                if cat_token_raw and cat_quality:
                    token_component_lines.append(
                        f"{cat}: Final {token_by_cat[cat]:.2f} | Token {statistics.mean(cat_token_raw):.2f} | Quality x{statistics.mean(cat_quality):.2f}"
                    )

            token_efficiency = statistics.mean(token_by_cat.values()) if token_by_cat else 0.0
            
            # Scores for Ranking
            speed_score = min(10, avg_speed / 5)
            win_score = win_rate_pct / 10
            smart_score = (
                (avg_qual * 0.75)
                + (speed_score * 0.10)
                + (win_score * 0.05)
                + (token_efficiency * 0.10)
            )
            
            # Build Segments for Stacked Bar
            segments = []
            
            # 1. Category averages (prompt categories only — same order as prompt_categories)
            for cat in leaderboard_data['prompt_categories']:
                if cat in data['categories']:
                    scores = data['categories'][cat]['scores']
                    cat_avg = statistics.mean(scores)
                    detail_str = "\n".join(data['categories'][cat]['details'])
                    
                    segments.append({
                        'category': cat,
                        'score': cat_avg,
                        'details': detail_str
                    })
            
            # 2. Optional Metrics
            if include_speed:
                segments.append({
                    'category': "Speed (TPS)",
                    'score': speed_score,
                    'details': f"Avg: {avg_speed:.1f} TPS"
                })
            
            if include_load_time:
                load_score = _load_time_score(avg_ttft)
                segments.append({
                    'category': "Load Time",
                    'score': load_score,
                    'details': f"Avg TTFT: {avg_ttft:.3f}s" if ttfts else "No TTFT samples",
                })
            
            if include_win_rate:
                segments.append({
                    'category': "Win Rate",
                    'score': win_score,
                    'details': f"{win_rate_pct:.0f}% Win Rate"
                })
            if include_token_efficiency:
                segments.append({
                    'category': _TOKEN_EFFICIENCY_CATEGORY,
                    'score': token_efficiency,
                    'details': "\n".join(token_component_lines) if token_component_lines else "No token samples",
                })
            
            leaderboard_data['model_data'][display_name] = {
                'display_name': display_name,
                'smart_score': smart_score,
                'quality': avg_qual,
                'speed': avg_speed,
                'avg_ttft': avg_ttft,
                'win_rate': win_rate_pct,
                'token_efficiency': token_efficiency,
                'token_efficiency_by_category': token_by_cat,
                'token_efficiency_prompt_count': model_token_prompt_count,
                'count': len(data['raw_scores']),
                'segments': segments
            }
        
        augmented: list[str] = []
        if include_speed:
            leaderboard_data['all_categories'].append("Speed (TPS)")
            augmented.append("Speed (TPS)")
        if include_load_time:
            leaderboard_data['all_categories'].append("Load Time")
            augmented.append("Load Time")
        if include_win_rate:
            leaderboard_data['all_categories'].append("Win Rate")
            augmented.append("Win Rate")
        if include_token_efficiency:
            leaderboard_data['all_categories'].append(_TOKEN_EFFICIENCY_CATEGORY)
            augmented.append(_TOKEN_EFFICIENCY_CATEGORY)
        leaderboard_data['augmented_categories'] = augmented
        
        return leaderboard_data
    
    def calculate_krippendorff_alpha(self, raw_matrix):
        """
        Calculate Krippendorff's alpha for inter-rater reliability.
        
        Args:
            raw_matrix: List of (response_id, evaluator_id, rating) tuples
            
        Returns:
            tuple: (alpha_value, interpretation_string) or (None, error_message)
        """
        if not HAS_NLTK:
            return None, "Library 'nltk' missing."
        
        if not raw_matrix:
            return None, "No evaluation data found."
        
        formatted_data = []
        coders = set()
        labels = set()
        
        for row in raw_matrix:
            response_id, evaluator_id, rating = row
            formatted_data.append((evaluator_id, response_id, rating))
            coders.add(evaluator_id)
            labels.add(rating)
        
        if len(coders) < 2:
            return None, "Need at least 2 Evaluators."
        if len(labels) <= 1:
            return 1.0, "Perfect Agreement"
        
        try:
            def interval_distance(label1, label2):
                return (label1 - label2) ** 2
            
            task = agreement.AnnotationTask(data=formatted_data, distance=interval_distance)
            alpha = task.alpha()
            return round(alpha, 3), self._interpret_alpha(alpha)
        except Exception as e:
            return None, f"Calc Error: {e}"
    
    def _interpret_alpha(self, alpha):
        """Interpret Krippendorff's alpha value"""
        if alpha >= 0.8:
            return "High Reliability"
        if alpha >= 0.667:
            return "Tentative Reliability"
        return "Low Reliability"
    
    def get_conflict_items(self, raw_evaluations):
        """
        Find evaluation conflicts (high variance in scores).
        Returns the highest disagreement per category (each category appears at most once).
        
        Args:
            raw_evaluations: List of evaluation dictionaries
            
        Returns:
            List of conflict items sorted by variance, max one per category
        """
        if not raw_evaluations:
            return []
        
        # Group by response_id
        grouped = {}
        for r in raw_evaluations:
            rid = r['response_id']
            if rid not in grouped:
                grouped[rid] = {
                    'category': r['category'],
                    'respondent': r['respondent'],
                    'scores': [],
                    'evals': []
                }
            grouped[rid]['scores'].append(r['rating'])
            grouped[rid]['evals'].append({
                'judge': r['evaluator'],
                'score': r['rating'],
                'rationale': r['rationale']
            })
        
        # Calculate variance for each response
        conflicts = []
        for i in grouped.values():
            if len(i['scores']) < 2:
                continue
            var = statistics.stdev(i['scores'])
            if var > 0.5:
                i['variance'] = var
                conflicts.append(i)
        
        # Group by category and keep only the highest variance per category
        category_max = {}
        for conflict in conflicts:
            cat = conflict['category']
            if cat not in category_max or conflict['variance'] > category_max[cat]['variance']:
                category_max[cat] = conflict
        
        # Sort by variance and return top 5
        result = list(category_max.values())
        result.sort(key=lambda x: x['variance'], reverse=True)
        return result[:5]
    
    def get_consistency_stats(self, raw_data):
        """Calculate consistency statistics per model."""
        stats = {}
        for r in raw_data:
            m = raw_data_model_key(r)
            s = r['Score']
            if s is None or s < 0:
                continue
            if m not in stats:
                stats[m] = []
            stats[m].append(s)
        
        res = []
        for display_name, s in stats.items():
            res.append({
                'model': display_name,
                'mean': statistics.mean(s),
                'median': statistics.median(s),
                'stdev': statistics.stdev(s) if len(s) > 1 else 0.0,
                'perfect_rate': (s.count(10) / len(s)) * 100
            })
        return sorted(res, key=lambda x: x['model'])
    
    def get_category_trends(self, raw_data):
        """Calculate category trends across models."""
        chart = {}
        totals = {}
        cats = set()
        
        for r in raw_data:
            m = raw_data_model_key(r)
            c = r['Category']
            s = r['Score']
            if s is None or s < 0:
                continue
            if m not in chart:
                chart[m] = {}
            if c not in chart[m]:
                chart[m][c] = []
            chart[m][c].append(s)
            if c not in totals:
                totals[c] = []
            totals[c].append(s)
            cats.add(c)
        
        sorted_cats = sorted(list(cats))
        model_avgs = {}
        for display_name in chart:
            model_avgs[display_name] = [
                statistics.mean(chart[display_name].get(c, [])) if chart[display_name].get(c) else 0
                for c in sorted_cats
            ]
        return {
            "categories": sorted_cats,
            "global_avgs": [
                statistics.mean(totals.get(c, [])) if totals.get(c) else 0
                for c in sorted_cats
            ],
            "model_avgs": model_avgs
        }
    
    def get_hardware_stats(self, raw_data):
        """Calculate hardware statistics per model."""
        stats = {}
        for r in raw_data:
            m = raw_data_model_key(r)
            if m not in stats:
                stats[m] = {
                    'peak_vram': [],
                    'peak_ram': [],
                    'therm_throttles': 0,
                    'pwr_throttles': 0,
                    'context_eff': []
                }
            if r.get('peak_vram_usage_gb'):
                stats[m]['peak_vram'].append(r['peak_vram_usage_gb'])
            if r.get('peak_ram_usage_gb'):
                stats[m]['peak_ram'].append(r['peak_ram_usage_gb'])
            if r.get('thermal_throttling_count'):
                stats[m]['therm_throttles'] += r['thermal_throttling_count']
            if r.get('power_throttling_count'):
                stats[m]['pwr_throttles'] += r['power_throttling_count']
            ctx_len = r.get('Context') or r.get('context_length')
            if ctx_len and r.get('tokens_generated'):
                ctx_pct = (r['tokens_generated'] / ctx_len) * 100
                stats[m]['context_eff'].append((ctx_pct, r['GenTPS']))
        
        res = []
        for display_name, d in stats.items():
            res.append({
                'model': display_name,
                'max_vram': max(d['peak_vram']) if d['peak_vram'] else 0,
                'max_ram': max(d['peak_ram']) if d['peak_ram'] else 0,
                'therm_throttles': d['therm_throttles'],
                'pwr_throttles': d['pwr_throttles'],
                'context_data': d['context_eff']
            })
        return res
    
    def get_best_in_class(self, raw_data, category):
        """Find best performing model in a category."""
        filtered = [r for r in raw_data if r['Category'] == category]
        if not filtered:
            return None, []
        
        model_stats = {}
        for r in filtered:
            m = raw_data_model_key(r)
            s = r['Score']
            if s is None or s < 0:
                continue
            if m not in model_stats:
                model_stats[m] = []
            model_stats[m].append(s)
        
        results = []
        for display_name, scores in model_stats.items():
            results.append({
                'model': display_name,
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'stdev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                'count': len(scores),
                'ratio': (scores.count(10) / len(scores)) * 100
            })
        
        results.sort(key=lambda x: (-x['mean'], x['stdev']))
        if not results:
            return None, []
        return results[0], results
    
    def get_performance_scatter_data(self, raw_data):
        """Get performance scatter plot data."""
        model_perf = {}
        for row in raw_data:
            m = raw_data_model_key(row)
            if m not in model_perf:
                model_perf[m] = {
                    'gen_tps': [],
                    'scores': [],
                    'ttft': [],
                    'peak_gpu': []
                }
            
            if row['GenTPS']:
                model_perf[m]['gen_tps'].append(row['GenTPS'])
            if row['TTFT']:
                model_perf[m]['ttft'].append(row['TTFT'])
            if row['PeakGPU']:
                model_perf[m]['peak_gpu'].append(row['PeakGPU'])
            if row['Score'] is not None and row['Score'] >= 0:
                model_perf[m]['scores'].append(row['Score'])
        
        results = []
        for display_name, stats in model_perf.items():
            def safe_avg(lst):
                return statistics.mean(lst) if lst else 0.0
            
            avg_gen = safe_avg(stats['gen_tps'])
            avg_score = safe_avg(stats['scores'])
            
            if avg_gen > 0 and avg_score > 0:
                results.append({
                    "model": display_name,
                    "avg_gen_tps": avg_gen,
                    "avg_score": avg_score
                })
        return results
    
    def process_comparison_data(self, raw_data, system_ids, system_map, global_stats, strict_mode=True):
        """
        Process comparison data for multiple systems.
        Groups by (llm_identifier, context_length) so only same-context runs are compared.
        
        Args:
            raw_data: List of comparison data dictionaries (with context_length)
            system_ids: List of system IDs being compared
            system_map: Dict mapping system_id -> system_name
            global_stats: Global statistics dict
            strict_mode: If True, filter out models with >5% size variance
            
        Returns:
            dict with 'systems', 'global_stats', and 'models'
        """
        # Process Models (group by llm_identifier + context_length for valid comparisons)
        models_grouped = {}
        for row in raw_data:
            m_id = row['llm_identifier']
            ctx = row.get('context_length') or 0
            m_key = (m_id, ctx)
            if m_key not in models_grouped:
                models_grouped[m_key] = {
                    'name': row['display_name'] or m_id.split('/')[-1],
                    'sizes': [],
                    'data': {}
                }
            
            sys_id = row['system_id']
            models_grouped[m_key]['data'][sys_id] = dict(row)
            models_grouped[m_key]['sizes'].append(row['size_bytes'] or 0)
        
        # Apply Filtering & Logic
        final_models = {}
        
        for m_key, content in models_grouped.items():
            m_id, ctx = m_key
            # 1. Intersection Check: Must have data for ALL selected systems
            present_systems = content['data'].keys()
            if not all(sid in present_systems for sid in system_ids):
                continue
            
            sizes = [s for s in content['sizes'] if s > 0]
            warning = None
            
            # 2. Size Variance Check
            if sizes:
                min_size = min(sizes)
                max_size = max(sizes)
                variance = (max_size - min_size) / min_size if min_size > 0 else 0
                
                if variance > 0.05:
                    if strict_mode:
                        continue  # Skip this model entirely
                    else:
                        warning = f"Size Mismatch: Variance {int(variance*100)}%"
            
            # Use display key that includes context for uniqueness
            display_key = f"{m_id} ({ctx})" if ctx else m_id
            final_models[display_key] = {
                'name': f"{content['name']} ({ctx})" if ctx else content['name'],
                'data': content['data'],
                'warning': warning
            }
        
        return {
            'systems': system_map,
            'global_stats': global_stats,
            'models': final_models
        }

    def model_analyzer_ordered_models(self, raw_data, leaderboard):
        """Display names sorted by leaderboard smart_score (see model_analyzer_stats)."""
        return _model_analyzer_stats.ordered_model_names(leaderboard, raw_data)

    def prompt_analyzer_categories(self, raw_data, hidden_categories=None):
        return _prompt_analyzer_stats.categories_visible(raw_data, hidden_categories)

    def prompt_analyzer_sorted_rows(
        self, raw_data, category: str, hidden_categories=None, sort_mode: str = "name"
    ):
        rows = _prompt_analyzer_stats.prompt_summaries_for_category(
            raw_data, category, hidden_categories
        )
        return _prompt_analyzer_stats.sort_prompt_rows(rows, sort_mode)

