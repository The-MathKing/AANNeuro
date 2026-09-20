"""
Robust GEO Series Matrix & Platform Annotation Parser.
Handles header metadata extraction, probe expression matrices, and GPL annotation mapping.
"""

import gzip
import os
import io
import pandas as pd
import numpy as np


def parse_series_matrix(filepath, load_expression=True):
    """
    Parse a GEO series matrix file (.txt or .txt.gz).
    Returns (metadata_df, expression_df).
    metadata_df has index = GEO accession (e.g. GSM...).
    expression_df has rows = Probes/Genes, columns = GEO accessions.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GEO file not found: {filepath}")

    open_fn = gzip.open if filepath.endswith('.gz') else open
    
    meta_headers = {}
    sample_ids = []
    matrix_start_pos = 0

    with open_fn(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            
            line_str = line.strip()
            if line_str.startswith('!Sample_geo_accession'):
                parts = [p.strip('"\t ') for p in line.strip().split('\t')]
                sample_ids = parts[1:]
                meta_headers['geo_accession'] = sample_ids
            elif line_str.startswith('!Sample_'):
                parts = [p.strip('"\t ') for p in line.strip().split('\t')]
                key = parts[0].replace('!Sample_', '')
                if key not in meta_headers:
                    meta_headers[key] = []
                meta_headers[key].append(parts[1:])
            elif line_str.startswith('!series_matrix_table_begin'):
                matrix_start_pos = f.tell()
                break

    if not sample_ids:
        raise ValueError(f"Could not find !Sample_geo_accession in {filepath}")

    # Build metadata dataframe
    metadata = {sid: {} for sid in sample_ids}
    for key, val in meta_headers.items():
        if key == 'geo_accession':
            for sid in sample_ids:
                metadata[sid]['geo_accession'] = sid
        elif key == 'characteristics_ch1':
            # List of characteristic lines
            for row in val:
                for sid, v in zip(sample_ids, row):
                    if ':' in v:
                        k, item_val = v.split(':', 1)
                        metadata[sid][k.strip()] = item_val.strip()
                    else:
                        metadata[sid][f'char_{len(metadata[sid])}'] = v.strip()
        else:
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
                # Multiple lines with same key, take first or join
                for sid, v in zip(sample_ids, val[0]):
                    metadata[sid][key] = v
            elif isinstance(val, list):
                for sid, v in zip(sample_ids, val):
                    metadata[sid][key] = v

    meta_df = pd.DataFrame.from_dict(metadata, orient='index')

    expr_df = None
    if load_expression and matrix_start_pos > 0:
        with open_fn(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            f.seek(matrix_start_pos)
            expr_lines = []
            for line in f:
                if line.startswith('!series_matrix_table_end'):
                    break
                expr_lines.append(line)
        
        expr_buf = io.StringIO(''.join(expr_lines))
        expr_df = pd.read_csv(expr_buf, sep='\t', index_col=0, low_memory=False)
        expr_df.index = expr_df.index.astype(str).str.strip('"\t ')
        expr_df.columns = expr_df.columns.astype(str).str.strip('"\t ')
        # Drop rows/columns that might be empty
        expr_df = expr_df.dropna(how='all')
        expr_df = expr_df.apply(pd.to_numeric, errors='coerce')

    return meta_df, expr_df


def parse_platform_annotation(filepath):
    """
    Parse a GEO platform table or annotation file.
    Returns DataFrame mapping Probe_ID -> Gene_Symbol / Symbol.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Platform annotation file not found: {filepath}")

    open_fn = gzip.open if filepath.endswith('.gz') else open
    table_lines = []
    in_table = False

    with open_fn(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith('!platform_table_begin') or line_str.startswith('#'):
                in_table = True
                continue
            elif line_str.startswith('!platform_table_end'):
                break
            elif in_table or '\t' in line:
                # If soft table format or standard TSV
                if line.startswith('^') or line.startswith('!'):
                    continue
                table_lines.append(line)

    if not table_lines:
        # Try direct TSV read
        annot_df = pd.read_csv(filepath, sep='\t', comment='#', low_memory=False)
    else:
        buf = io.StringIO(''.join(table_lines))
        annot_df = pd.read_csv(buf, sep='\t', low_memory=False)

    # Standardize column names
    # Deduplicate columns if any
    cols = list(annot_df.columns)
    new_cols = []
    seen = {}
    for c in cols:
        cl = c.lower().strip()
        if cl in seen:
            seen[cl] += 1
            new_cols.append(f"{cl}_{seen[cl]}")
        else:
            seen[cl] = 1
            new_cols.append(cl)
    annot_df.columns = new_cols

    probe_col = None
    symbol_col = None

    for col in annot_df.columns:
        if col in ['id', 'probe_id', 'id_ref', 'spot_id', 'probeset_id', 'id_1']:
            if probe_col is None:
                probe_col = col
        elif col in ['gene symbol', 'genesymbol', 'symbol', 'gene_symbol', 'gene_assignment', 'ilmn_gene', 'gene symbol_1', 'gene_symbol_1']:
            if symbol_col is None:
                symbol_col = col

    if probe_col is None and len(annot_df.columns) > 0:
        probe_col = annot_df.columns[0]

    if symbol_col is None:
        for col in annot_df.columns:
            if 'symbol' in col and 'description' not in col:
                symbol_col = col
                break
            elif 'gene' in col and 'title' not in col and 'id' not in col and 'description' not in col:
                symbol_col = col
                break

    if probe_col is not None and symbol_col is not None:
        probe_series = annot_df[probe_col].astype(str).str.strip('"\t ')
        symbol_series = annot_df[symbol_col].astype(str).str.strip('"\t ')
        cleaned_df = pd.DataFrame({'probe_id': probe_series, 'gene_symbol': symbol_series})
        cleaned_df['gene_symbol'] = cleaned_df['gene_symbol'].apply(
            lambda s: s.split('///')[0].split('//')[0].strip() if pd.notna(s) and s != 'nan' and s != '---' and s != '' else np.nan
        )
        cleaned_df = cleaned_df.dropna(subset=['probe_id', 'gene_symbol'])
        cleaned_df = cleaned_df[cleaned_df['gene_symbol'] != '']
        return cleaned_df.drop_duplicates()

    return annot_df
