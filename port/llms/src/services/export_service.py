import os
import re
import datetime

from src.database.manager import ROOT_DIR
from src.utils.model_display_label import model_card_label

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import openpyxl
    from openpyxl.utils.exceptions import IllegalCharacterError
except ImportError:
    openpyxl = None

class ExportService:
    def _sanitize_for_excel(self, val):
        """
        Removes characters that are illegal in Excel cells.
        """
        if isinstance(val, str):
            # Excel allows: \t, \n, \r and characters from range [0x20, 0xD7FF], [0xE000, 0xFFFD], [0x10000, 0x10FFFF]
            # We remove everything else (control characters).
            return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', val)
        return val

    def export_to_excel(self, raw_data, filename="benchmark_report"):
        if not pd:
            raise ImportError("Pandas is not installed. Run 'pip install pandas openpyxl'")
        
        if not openpyxl:
            raise ImportError("Library 'openpyxl' is missing. Install it with: pip install openpyxl")

        if not raw_data:
            raise ValueError("No data available to export.")

        # --- FIX: Add Datetime Stamp ---
        # Format: benchmark_report_YYYY-MM-DD_HH-MM-SS.xlsx
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Strip extension if present to append timestamp correctly
        base_name = filename.replace(".xlsx", "")
        final_filename = f"{base_name}_{timestamp}.xlsx"

        reports_dir = os.path.join(ROOT_DIR, "exports", "reports")
        
        if not os.path.exists(reports_dir):
            try:
                os.makedirs(reports_dir)
            except OSError:
                # Fallback to current directory if permission denied
                reports_dir = os.getcwd()

        full_path = os.path.join(reports_dir, final_filename)
        print(f"[EXPORT] Preparing to save to: {full_path}")
        
        # Prepare DataFrame
        df = pd.DataFrame(raw_data)
        
        # Sanitize all string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].apply(self._sanitize_for_excel)

        # Safe column access — include provider when BackendType is present
        if 'DisplayName' in df.columns and 'Model' in df.columns:
            if 'BackendType' in df.columns:
                df['FinalName'] = df.apply(
                    lambda r: model_card_label(r.get('DisplayName'), r.get('Model'), r.get('BackendType')),
                    axis=1,
                )
            else:
                df['FinalName'] = df['DisplayName'].fillna(df['Model'])
        elif 'Model' in df.columns:
            df['FinalName'] = df['Model']
        else:
            df['FinalName'] = "Unknown"

        # 1. Summary Leaderboard
        agg_dict = {
            'Score': ['mean', 'count', 'std']
        }
        
        # Add metrics if they exist
        if 'GenTPS' in df.columns: agg_dict['GenTPS'] = 'mean'
        if 'TotalTPS' in df.columns: agg_dict['TotalTPS'] = 'mean'
        if 'TTFT' in df.columns: agg_dict['TTFT'] = 'mean'
        if 'PeakGPU' in df.columns: agg_dict['PeakGPU'] = 'max'
        
        # Only include ref metrics if any values exist (standardized prompts only)
        if 'BLEU' in df.columns and df['BLEU'].notna().any(): agg_dict['BLEU'] = 'mean'
        if 'ROUGE' in df.columns and df['ROUGE'].notna().any(): agg_dict['ROUGE'] = 'mean'
        if 'BERT' in df.columns and df['BERT'].notna().any(): agg_dict['BERT'] = 'mean'

        summary_df = df.groupby('FinalName').agg(agg_dict).round(2)
        
        # 2. Quality Matrix (Pivot)
        if 'Category' in df.columns and 'Score' in df.columns:
            quality_df = df.pivot_table(index='FinalName', columns='Category', values='Score', aggfunc='mean').round(1)
        else:
            quality_df = pd.DataFrame(["No Category Data"])

        # 3. Performance Stats
        perf_cols = {}
        for col in ['GenTPS', 'TotalTPS', 'TTFT', 'TotalTime']:
            if col in df.columns: perf_cols[col] = 'mean'
        for col in ['PeakGPU', 'PeakCPU']:
            if col in df.columns: perf_cols[col] = 'max'
            
        if perf_cols:
            perf_df = df.groupby('FinalName').agg(perf_cols).round(2)
        else:
            perf_df = pd.DataFrame(["No Perf Data"])

        # Write to Excel
        try:
            with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name="Summary Leaderboard")
                quality_df.to_excel(writer, sheet_name="Quality Matrix")
                perf_df.to_excel(writer, sheet_name="Performance")
                df.to_excel(writer, sheet_name="Raw Data", index=False)
            print(f"[EXPORT] File successfully written.")
            return full_path
        except PermissionError:
            raise PermissionError(f"Permission denied. Close '{final_filename}' if it is open in Excel.")
        except Exception as e:
            # Re-raise with detail so the UI can capture it
            raise RuntimeError(f"Pandas Write Error: {e}")