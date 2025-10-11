#!/usr/bin/env python3
"""
Sample Dataset Generator for GenAI RAG Application
Generates a sample .jsonl file with academic abstracts for testing
"""

import json
import random
from datetime import datetime, timedelta

# Sample data for generating realistic abstracts
TOPICS = [
    "transformers", "neural networks", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "attention mechanisms", "BERT",
    "GPT", "machine translation", "image classification", "object detection",
    "semantic segmentation", "generative models", "adversarial networks", "convolutional networks",
    "recurrent networks", "transfer learning", "few-shot learning", "meta-learning",
    "graph neural networks", "quantum computing", "federated learning", "explainable AI"
]

METHODS = [
    "novel approach", "improved method", "efficient algorithm", "state-of-the-art",
    "innovative technique", "scalable solution", "robust framework", "unified model",
    "end-to-end system", "self-supervised", "attention-based", "transformer-based"
]

IMPROVEMENTS = [
    "significant improvement", "better performance", "reduced complexity",
    "enhanced accuracy", "faster training", "lower memory usage", "improved generalization",
    "state-of-the-art results", "competitive performance", "superior quality"
]

APPLICATIONS = [
    "natural language understanding", "text generation", "sentiment analysis",
    "question answering", "image recognition", "video analysis", "speech recognition",
    "recommendation systems", "autonomous driving", "medical diagnosis", "drug discovery",
    "financial forecasting", "climate modeling", "robotics", "game playing"
]

def generate_abstract(topic, paper_id):
    """Generate a realistic-looking abstract."""
    method = random.choice(METHODS)
    improvement = random.choice(IMPROVEMENTS)
    application = random.choice(APPLICATIONS)
    
    templates = [
        f"We propose a {method} for {topic} that achieves {improvement}. "
        f"Our approach is based on {random.choice(TOPICS)} and demonstrates "
        f"strong results on {application}. Experiments show that our method "
        f"outperforms previous baselines by a significant margin. "
        f"We provide theoretical analysis and empirical validation on standard benchmarks.",
        
        f"This paper introduces a {method} for {topic}. "
        f"Unlike previous work, our approach achieves {improvement} while maintaining "
        f"computational efficiency. We evaluate our method on {application} tasks "
        f"and demonstrate its effectiveness across multiple datasets. "
        f"Our contributions include theoretical insights and practical implementations.",
        
        f"Recent advances in {topic} have shown promising results, but challenges remain. "
        f"We present a {method} that addresses these limitations and achieves {improvement}. "
        f"Our approach combines ideas from {random.choice(TOPICS)} and {random.choice(TOPICS)}. "
        f"Extensive experiments on {application} validate our approach and show "
        f"that it sets new state-of-the-art performance on several benchmarks.",
        
        f"In this work, we investigate {topic} and propose a {method}. "
        f"Our key insight is that incorporating {random.choice(TOPICS)} can lead to {improvement}. "
        f"We conduct comprehensive experiments on {application} and provide "
        f"both quantitative and qualitative analysis. The results demonstrate "
        f"the effectiveness and efficiency of our proposed approach.",
    ]
    
    return random.choice(templates)

def generate_title(topic):
    """Generate a paper title."""
    templates = [
        f"{random.choice(METHODS).title()} for {topic.title()}",
        f"{topic.title()}: A {random.choice(METHODS).title()}",
        f"Towards {random.choice(IMPROVEMENTS).title()} in {topic.title()}",
        f"{random.choice(METHODS).title()}: Advances in {topic.title()}",
        f"Efficient {topic.title()} via {random.choice(METHODS).title()}",
    ]
    return random.choice(templates)

def generate_authors():
    """Generate author names."""
    first_names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
        "Iris", "Jack", "Kate", "Leo", "Maria", "Nathan", "Olivia", "Peter",
        "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
    ]
    
    num_authors = random.randint(1, 5)
    authors = []
    for _ in range(num_authors):
        first = random.choice(first_names)
        last = random.choice(last_names)
        authors.append(f"{first} {last}")
    
    return ", ".join(authors)

def generate_categories():
    """Generate paper categories."""
    categories = [
        "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "stat.ML",
        "cs.RO", "cs.HC", "cs.IR", "cs.CR", "cs.DC", "cs.DB"
    ]
    
    num_cats = random.randint(1, 3)
    return ", ".join(random.sample(categories, num_cats))

def generate_paper_id():
    """Generate a realistic arXiv-style paper ID."""
    year = random.randint(2020, 2024)
    month = random.randint(1, 12)
    number = random.randint(1000, 9999)
    return f"{year}{month:02d}.{number:05d}"

def generate_dataset(output_file, num_papers=100):
    """Generate a complete dataset."""
    print(f"Generating {num_papers} sample papers...")
    
    papers = []
    used_ids = set()
    
    for i in range(num_papers):
        # Generate unique paper ID
        paper_id = generate_paper_id()
        while paper_id in used_ids:
            paper_id = generate_paper_id()
        used_ids.add(paper_id)
        
        # Select topic
        topic = random.choice(TOPICS)
        
        # Generate paper data
        paper = {
            "id": paper_id,
            "title": generate_title(topic),
            "authors": generate_authors(),
            "categories": generate_categories(),
            "abstract": generate_abstract(topic, paper_id)
        }
        
        papers.append(paper)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_papers} papers...")
    
    # Sort by ID
    papers.sort(key=lambda x: x['id'])
    
    # Write to file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper) + '\n')
    
    print(f"✅ Successfully generated {len(papers)} papers!")
    print(f"   Output: {output_file}")
    print(f"   Size: {len(papers)} records")
    print(f"\nYou can now use this file with the GenAI RAG application:")
    print(f"   docker run -p 8080:8080 \\")
    print(f"     -v $(pwd)/{output_file}:/data/{output_file}:ro \\")
    print(f"     -e DATA_PATH=/data/{output_file} \\")
    print(f"     yourname/genai-app:latest")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate sample academic dataset for testing"
    )
    parser.add_argument(
        '-n', '--num-papers',
        type=int,
        default=100,
        help='Number of papers to generate (default: 100)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='sample_dataset.jsonl',
        help='Output file name (default: sample_dataset.jsonl)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  Sample Dataset Generator for GenAI RAG Application")
    print("="*70)
    print()
    
    generate_dataset(args.output, args.num_papers)

if __name__ == "__main__":
    main()
