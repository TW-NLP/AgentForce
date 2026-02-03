#!/usr/bin/env python3
"""
Extract tables from PDF files
Part of the pdf-processing Agent Skill
"""

import argparse
import sys
import json
from pathlib import Path

def extract_tables(pdf_path, format='excel', output=None):
    """Extract tables from PDF"""
    try:
        import pdfplumber
        import pandas as pd
    except ImportError as e:
        print(f"❌ Error: Missing dependency: {e}")
        print("   Install with: pip install pdfplumber pandas openpyxl")
        return 1
    
    try:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            print(f"❌ Error: File not found: {pdf_path}")
            return 1
        
        all_tables = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table_num, table in enumerate(tables, 1):
                    if table and len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        all_tables.append({
                            'page': page_num,
                            'table': table_num,
                            'dataframe': df,
                            'shape': df.shape
                        })
        
        if not all_tables:
            print("ℹ️  No tables found in PDF")
            print("   Tip: If this is a scanned PDF, try using ocr_pdf.py first")
            return 1
        
        # Generate output path
        if not output:
            base_name = pdf_file.stem
            if format == 'excel':
                output = f"{base_name}_tables.xlsx"
            elif format == 'csv':
                output = f"{base_name}_tables.csv"
            else:
                output = f"{base_name}_tables.json"
        
        output_file = Path(output)
        
        # Save based on format
        if format == 'excel':
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for t in all_tables:
                    sheet_name = f"P{t['page']}_T{t['table']}"[:31]
                    t['dataframe'].to_excel(writer, sheet_name=sheet_name, index=False)
        
        elif format == 'csv':
            combined = pd.concat([t['dataframe'] for t in all_tables], ignore_index=True)
            combined.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        elif format == 'json':
            json_data = []
            for t in all_tables:
                json_data.append({
                    'page': t['page'],
                    'table': t['table'],
                    'rows': t['shape'][0],
                    'columns': t['shape'][1],
                    'data': t['dataframe'].to_dict(orient='records')
                })
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Success message
        print(f"✅ Extracted {len(all_tables)} tables:")
        for t in all_tables:
            print(f"   • Page {t['page']}, Table {t['table']}: {t['shape'][0]} rows × {t['shape'][1]} columns")
        print(f"\n📁 Saved to: {output_file}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='Extract tables from PDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract to Excel (default)
  %(prog)s financial_report.pdf
  
  # Extract to CSV
  %(prog)s data.pdf --format csv
  
  # Custom output path
  %(prog)s data.pdf --output results/tables.xlsx
        """
    )
    
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--format', choices=['excel', 'csv', 'json'], 
                       default='excel', help='Output format (default: excel)')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    sys.exit(extract_tables(args.pdf_path, args.format, args.output))

if __name__ == '__main__':
    main()