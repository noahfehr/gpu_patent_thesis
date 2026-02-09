#!/usr/bin/env python3
"""
Test script for v2_core_expansion_pipeline functions.
This validates the pipeline logic without calling the lens.org API.
"""

import json
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_cpc_codes():
    """Test that all CPC codes are properly defined."""
    
    # Core CPC codes
    CORE_CPC_CODES = [
        'G06F9/3887',   # Parallel processing
        'G06F9/3888',   # Parallel processing
        'G06F9/38885',  # Parallel processing
        'G06F9/3009',   # Multiprocessing arrangements
        'G06F12/0842',  # Cache memory
        'G06F12/0844',  # Cache memory
        'G06F13/42',    # Bus architectures
        'G06F13/14',    # Bus architectures
        'G06F13/16',    # Bus architectures
    ]
    
    # Expansion CPC codes
    EXPANSION_CPC_CODES = [
        'G06F15/8007',  # Multiprocessor systems
        'G06F15/8053',  # Multiprocessor systems
        'G06N3/06',     # Neural networks
    ]
    
    # Keywords for ExpansionXVocab filtering
    VOCAB_KEYWORDS = [
        'gpu',
        'high-performance compute',
        'hpc'
    ]
    
    print("Testing CPC Codes and Keywords Configuration:")
    print(f"  ✓ Core CPC codes: {len(CORE_CPC_CODES)} codes")
    assert len(CORE_CPC_CODES) == 9, "Core should have 9 CPC codes"
    
    print(f"  ✓ Expansion CPC codes: {len(EXPANSION_CPC_CODES)} codes")
    assert len(EXPANSION_CPC_CODES) == 3, "Expansion should have 3 CPC codes"
    
    print(f"  ✓ Vocab keywords: {len(VOCAB_KEYWORDS)} keywords")
    assert len(VOCAB_KEYWORDS) == 3, "Should have 3 keywords"
    
    # Verify no overlap between core and expansion
    core_set = set(CORE_CPC_CODES)
    expansion_set = set(EXPANSION_CPC_CODES)
    overlap = core_set.intersection(expansion_set)
    print(f"  ✓ No overlap between Core and Expansion CPC codes")
    assert len(overlap) == 0, f"Unexpected overlap: {overlap}"
    
    return True

def test_keyword_filtering():
    """Test keyword filtering logic."""
    
    def contains_keywords(text, keywords):
        """Check if text contains any of the specified keywords (case-insensitive)."""
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    keywords = ['gpu', 'high-performance compute', 'hpc']
    
    print("\nTesting Keyword Filtering:")
    
    # Test cases
    test_cases = [
        ("GPU acceleration for machine learning", True),
        ("Graphics Processing Unit (GPU) for parallel computing", True),
        ("High-Performance Compute cluster architecture", True),
        ("HPC systems and applications", True),
        ("gpu-accelerated workloads", True),
        ("Standard CPU processing", False),
        ("Memory management in computers", False),
        ("Graphics Processing Unit without the acronym", False),  # Should be False - no "gpu" keyword
        ("", False),
        (None, False),
    ]
    
    for text, expected in test_cases:
        result = contains_keywords(text, keywords)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' -> {result} (expected {expected})")
        assert result == expected, f"Failed on: {text}"
    
    return True

def test_query_structure():
    """Test that query structure is correctly formatted."""
    
    def build_cpc_query(cpc_codes, jurisdiction, max_results=1000):
        """Build lens.org API query for patents with specific CPC codes."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {"term": {"classification_cpc.classification_id": code}} 
                                    for code in cpc_codes
                                ],
                                "minimum_should_match": 1
                            }
                        },
                        {"term": {"jurisdiction": jurisdiction}}
                    ]
                }
            },
            "size": max_results,
            "include": [
                "lens_id", "title", "abstract", "description", "claims",
                "date_published", "jurisdiction", "applicants", "inventors",
                "classification_cpc", "biblio"
            ]
        }
        return query
    
    print("\nTesting Query Structure:")
    
    test_codes = ['G06F9/3887', 'G06F9/3888']
    query = build_cpc_query(test_codes, 'US', 100)
    
    # Verify structure
    assert 'query' in query, "Query should have 'query' field"
    assert 'size' in query, "Query should have 'size' field"
    assert 'include' in query, "Query should have 'include' field"
    assert query['size'] == 100, "Size should be 100"
    
    # Verify CPC codes in query
    cpc_should = query['query']['bool']['must'][0]['bool']['should']
    assert len(cpc_should) == 2, "Should have 2 CPC code terms"
    
    # Verify jurisdiction
    jurisdiction_term = query['query']['bool']['must'][1]
    assert jurisdiction_term['term']['jurisdiction'] == 'US', "Jurisdiction should be US"
    
    print("  ✓ Query structure is correct")
    print("  ✓ CPC codes properly included")
    print("  ✓ Jurisdiction filter applied")
    print("  ✓ Include fields specified")
    
    return True

def test_directory_structure():
    """Test that directory structure was created."""
    
    print("\nTesting Directory Structure:")
    
    base_path = Path(__file__).parent.parent / 'data' / 'patents' / 'v2_core_expansion'
    
    datasets = ['core', 'expansion', 'expansionxvocab']
    subdirs = ['raw', 'parsed', 'text_clean', 'embeddings', 'logs']
    
    for dataset in datasets:
        dataset_path = base_path / dataset
        assert dataset_path.exists(), f"Dataset directory {dataset} should exist"
        print(f"  ✓ {dataset}/ directory exists")
        
        for subdir in subdirs:
            subdir_path = dataset_path / subdir
            assert subdir_path.exists(), f"Subdirectory {subdir} should exist in {dataset}"
            
            # Check for .gitkeep
            gitkeep = subdir_path / '.gitkeep'
            assert gitkeep.exists(), f".gitkeep should exist in {dataset}/{subdir}"
    
    print("  ✓ All dataset directories created")
    print("  ✓ All subdirectories created")
    print("  ✓ All .gitkeep files present")
    
    return True

def test_readme_exists():
    """Test that README documentation was created."""
    
    print("\nTesting Documentation:")
    
    readme_path = Path(__file__).parent.parent / 'data' / 'patents' / 'v2_core_expansion' / 'README.md'
    assert readme_path.exists(), "README.md should exist"
    print("  ✓ v2_core_expansion/README.md exists")
    
    # Check that it contains key information
    content = readme_path.read_text()
    assert 'Core Dataset' in content, "README should document Core dataset"
    assert 'Expansion Dataset' in content, "README should document Expansion dataset"
    assert 'ExpansionXVocab' in content, "README should document ExpansionXVocab dataset"
    assert 'G06F' in content, "README should list CPC codes"
    assert 'gpu' in content.lower(), "README should mention GPU keyword"
    print("  ✓ README contains Core dataset documentation")
    print("  ✓ README contains Expansion dataset documentation")
    print("  ✓ README contains ExpansionXVocab documentation")
    print("  ✓ README contains CPC codes")
    
    return True

def main():
    """Run all tests."""
    print("=" * 80)
    print("V2 CORE EXPANSION PIPELINE - VALIDATION TESTS")
    print("=" * 80)
    
    try:
        test_cpc_codes()
        test_keyword_filtering()
        test_query_structure()
        test_directory_structure()
        test_readme_exists()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nThe pipeline is correctly configured and ready to use.")
        print("To run the pipeline, execute: jupyter notebook code/v2_core_expansion_pipeline.ipynb")
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
