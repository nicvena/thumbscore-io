#!/usr/bin/env python3
"""
Comprehensive system test for Thumbscore.io
Tests: Server health, FAISS, Brain, Scoring pipeline, Multi-thumbnail comparison
"""
import requests
import json

print('='*70)
print('🧪 COMPREHENSIVE SYSTEM TEST - Thumbscore.io')
print('='*70)

# Test 1: Health check
print('\n[1/5] Testing server health...')
try:
    response = requests.get('http://localhost:8000/', timeout=5)
    if response.status_code == 200:
        print('✓ Server is running')
        health_data = response.json()
        print(f'  Service: {health_data.get("service")}')
        print(f'  Version: {health_data.get("version")}')
        print(f'  Status: {health_data.get("status")}')
    else:
        print(f'✗ Server health check failed: {response.status_code}')
except Exception as e:
    print(f'✗ Server not accessible: {e}')
    exit(1)

# Test 2: FAISS status
print('\n[2/5] Testing FAISS similarity system...')
try:
    response = requests.get('http://localhost:8000/internal/faiss-status', timeout=10)
    if response.status_code == 200:
        faiss_data = response.json()
        print(f'✓ FAISS is operational')
        print(f'  Cache ready: {faiss_data.get("cache_ready")}')
        print(f'  Total niches: {faiss_data.get("cache_stats", {}).get("total_niches", 0)}')
        print(f'  Total items: {faiss_data.get("cache_stats", {}).get("total_items", 0)}')
        print(f'  Memory usage: {faiss_data.get("cache_stats", {}).get("memory_usage_mb", 0):.1f} MB')
    else:
        print(f'⚠ FAISS status check returned {response.status_code}')
except Exception as e:
    print(f'⚠ FAISS status check failed: {e}')

# Test 3: Brain status
print('\n[3/5] Testing YouTube Intelligence Brain...')
try:
    response = requests.get('http://localhost:8000/internal/brain-status', timeout=10)
    if response.status_code == 200:
        brain_data = response.json()
        print(f'✓ Brain is initialized')
        print(f'  Data collector: {brain_data.get("data_collector_ready")}')
        print(f'  Pattern miner: {brain_data.get("pattern_miner_ready")}')
        print(f'  Niche models: {brain_data.get("niche_models_ready")}')
    else:
        print(f'⚠ Brain status check returned {response.status_code}')
except Exception as e:
    print(f'⚠ Brain status check failed: {e}')

# Test 4: Scoring with real image
print('\n[4/5] Testing full scoring pipeline...')
try:
    url = 'http://localhost:8000/v1/score'
    data = {
        'title': 'How to Build AI Apps in 2025',
        'thumbnails': [
            {'id': 'test_thumb_1', 'url': 'https://httpbin.org/image/jpeg'}
        ],
        'category': 'tech'
    }
    response = requests.post(url, json=data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print('✓ Scoring completed successfully')
        
        thumb = result['results'][0]
        print(f'\n  📊 Results:')
        print(f'    CTR Score: {thumb["ctr_score"]}/100')
        print(f'    Winner: {result["winner_id"]}')
        print(f'    Explanation: {result["explanation"]}')
        
        print(f'\n  🎯 Sub-scores:')
        subscores = thumb['subscores']
        for key, value in subscores.items():
            print(f'    {key}: {value}/100')
        
        print(f'\n  💡 Insights ({len(thumb.get("insights", []))}):"')
        for i, insight in enumerate(thumb.get('insights', [])[:3], 1):
            print(f'    {i}. {insight}')
            
    else:
        print(f'✗ Scoring failed: {response.status_code}')
        print(f'  Error: {response.text}')
except Exception as e:
    print(f'✗ Scoring test failed: {e}')

# Test 5: Multiple thumbnails comparison
print('\n[5/5] Testing multi-thumbnail comparison...')
try:
    url = 'http://localhost:8000/v1/score'
    data = {
        'title': 'Amazing Tech Review',
        'thumbnails': [
            {'id': 'thumb_a', 'url': 'https://httpbin.org/image/jpeg'},
            {'id': 'thumb_b', 'url': 'https://httpbin.org/image/png'}
        ],
        'category': 'tech'
    }
    response = requests.post(url, json=data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print('✓ Multi-thumbnail comparison successful')
        print(f'  Compared: {len(result["results"])} thumbnails')
        print(f'  Winner: {result["winner_id"]}')
        print(f'  Rankings:')
        for i, thumb in enumerate(result['results'], 1):
            print(f'    {i}. {thumb["id"]}: {thumb["ctr_score"]}/100')
    else:
        print(f'✗ Multi-thumbnail test failed: {response.status_code}')
except Exception as e:
    print(f'✗ Multi-thumbnail test failed: {e}')

print('\n' + '='*70)
print('🎉 SYSTEM TEST COMPLETE')
print('='*70)

