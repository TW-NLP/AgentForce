#!/usr/bin/env python3
"""
Extract text from PDF files
Part of the pdf-processing Agent Skill
"""

import argparse
import sys
from pathlib import Path

def extract_text(pdf_path, pages=None, output=None):
    """Extract text from PDF"""
    try:
        import pdfplumber
    except ImportError:
        print("❌ Error: pdfplumber not installed")
        print("   Install with: pip install pdfplumber")
        return 1
    
    try:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            print(f"❌ Error: File not found: {pdf_path}")
            return 1
        
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)
            
            # Parse page range
            if pages:
                if '-' in pages:
                    start, end = map(int, pages.split('-'))
                    page_nums = list(range(start-1, end))
                elif ',' in pages:
                    page_nums = [int(p.strip())-1 for p in pages.split(',')]
                else:
                    page_nums = [int(pages)-1]
            else:
                page_nums = list(range(total_pages))
            
            # Extract text
            texts = []
            for idx in page_nums:
                if 0 <= idx < total_pages:
                    page_text = pdf.pages[idx].extract_text()
                    if page_text:
                        texts.append(f"=== Page {idx+1} ===\n{page_text}")
            
            if not texts:
                print("⚠️  Warning: No text extracted")
                return 1
            
            result = "\n\n".join(texts)
            
            # Output
            if output:
                output_file = Path(output)
                output_file.write_text(result, encoding='utf-8')
                print(f"✅ Extracted text from {len(texts)} pages → {output_file}")
            else:
                # Print to stdout (for agent to capture)
                print(f"✅ Extracted text from {len(texts)} pages:")
                print()
                # Limit output for readability
                if len(result) > 2000:
                    print(result[:2000])
                    print(f"\n... (showing first 2000 of {len(result)} characters)")
                else:
                    print(result)
            
            return 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='Extract text from PDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all text
  %(prog)s report.pdf
  
  # Extract specific pages
  %(prog)s report.pdf --pages "1-5"
  
  # Save to file
  %(prog)s report.pdf --output extracted.txt
        """
    )
    
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--pages', help='Page range (e.g., "1-5" or "1,3,5")')
    parser.add_argument('--output', help='Output file path (default: stdout)')
    
    args = parser.parse_args()
    
    sys.exit(extract_text(args.pdf_path, args.pages, args.output))

if __name__ == '__main__':
    main()