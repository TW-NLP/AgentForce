#!/usr/bin/env python3
"""
Crawl content from a URL using Firecrawl API
Part of the web-search Agent Skill
"""

import argparse
import sys
import json
import os
from pathlib import Path
from src.services.base import BaseConfigurableService

class CrawlSearch(BaseConfigurableService):
    """Crawl Search Skill"""
    
    def __init__(self):
        super().__init__()
    def build_instance():
        pass
    def crawl_url(self,url, max_length=10000, output=None):
        """Crawl a single URL"""
        try:
            from firecrawl import FirecrawlApp
        except ImportError as e:
            print(f"❌ Error: Missing dependency: {e}")
            print("   Install with: pip install firecrawl-py")
            return 1
        
        try:
            # Get API key from environment
            api_key = self.settings.FIRECRAWL_API_KEY
            if not api_key:
                print("⚠️  Warning: FIRECRAWL_API_KEY not set, crawling may be limited")
                print("   Set it with: export FIRECRAWL_API_KEY='your_key_here'")
                return 1
            
            # Initialize client
            client = FirecrawlApp(api_key=api_key)
            
            print(f"🕷️  Crawling: {url}")
            
            # Scrape the URL
            result = client.scrape(url)
            
            # Extract markdown content
            if hasattr(result, 'markdown') and result.markdown:
                content = result.markdown[:max_length]
            else:
                print("ℹ️  No content extracted")
                return 1
            
            # Prepare structured output for LLM
            structured_output = {
                "url": url,
                "content_length": len(content),
                "content": content,
                "success": True
            }
            
            # Save to file if output specified
            if output:
                output_file = Path(output)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # ALWAYS output JSON to stdout for LLM to parse
            print(json.dumps(structured_output, ensure_ascii=False, indent=2))
            
            return 0
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

def main():
    craw_enginer=CrawlSearch()

    parser = argparse.ArgumentParser(
        description='Crawl content from a URL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl a webpage
  %(prog)s "https://example.com/article"
  
  # Limit content length
  %(prog)s "https://example.com" --max-length 5000
  
  # Save to file
  %(prog)s "https://example.com" --output content.md
        """
    )
    
    parser.add_argument('url', help='URL to crawl')
    parser.add_argument('--max-length', type=int, default=10000,
                       help='Maximum content length (default: 10000)')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    sys.exit(craw_enginer.crawl_url(args.url, args.max_length, args.output))

if __name__ == '__main__':
    main()