import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import streamlit as st
from llm import LLM

load_dotenv()

def call_llm(context, query, content):
    """Call LLM with given query and content using the LLM class."""
    llm = LLM()  # Provider and model will be chosen randomly
    
    # Combine context and query for the system prompt
    system_prompt = context
    user_prompt = f"Content:\n{content}\n\nQuery: {query}"
    
    result = llm.ask(system_prompt=system_prompt, user_prompt=user_prompt)
    return result['text']

def split_chunks_by_size(chunks, max_size=12000):
    """Group chunks such that each group's total text size is under max_size."""
    groups = []
    current_group = []
    current_length = 0

    for chunk in chunks:
        chunk_text = chunk["text"]
        if current_length + len(chunk_text) > max_size:
            groups.append(current_group)
            current_group = [chunk]
            current_length = len(chunk_text)
        else:
            current_group.append(chunk)
            current_length += len(chunk_text)

    if current_group:
        groups.append(current_group)

    return groups

def format_chunk_for_prompt(chunk: Dict[str, Any]) -> str:
    """Format a chunk for inclusion in prompt"""
    formatted = f"Case Numbers: {', '.join(chunk.get('case_numbers', []))}\n"
    formatted += f"Party Names: {', '.join(chunk.get('party_names', []))}\n"
    formatted += f"Page: {chunk.get('page_number', 'N/A')}\n"
    formatted += f"Content:\n{chunk.get('text', '')}\n"
    formatted += "-" * 80 + "\n"
    return formatted

def generate_final_response(query, chunks):
    """Generate final response by querying LLM with grouped chunks and summarizing."""
    # Step 1: Group chunks
    chunk_groups = split_chunks_by_size(chunks)

    # Step 2: Get intermediate answers
    intermediate_answers = []
    for group in chunk_groups:
        combined_text = "\n".join(format_chunk_for_prompt(chunk) for chunk in group)
        intermediate_context = '''
        Based on the following court judgment excerpts, provide key findings and relevant information for the query.
        Focus on extracting the most relevant information like who won, legal principles, case details, 
        and factual information that directly addresses the query. Keep it concise. Also include the page number and filename mentioned in the context in your response.
        '''
        answer = call_llm(intermediate_context, query, combined_text)
        intermediate_answers.append(answer)

    # Step 3: Get final answer
    summary_input = "\n\n".join(
        f"Intermediate Answer {i+1}:\n{ans}" for i, ans in enumerate(intermediate_answers)
    )
    final_context = '''
    Based on the following intermediate answers generated from the chunks of judgement text, generate a final answer
    to the query. Keep the answer concise and include the relevant highlights at the begining of the response.
    Please provide a comprehensive answer. Include:
    1. Relevant case numbers and party names
    2. Key legal principles or findings
    3. Direct quotes from judgments where applicable
    4. Page number references for important information
    '''
    final_answer = call_llm(final_context, query, summary_input)

    return final_answer