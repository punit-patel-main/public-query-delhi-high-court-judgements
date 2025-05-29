import streamlit as st
import requests
import json
import os
from pinecone import Pinecone
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from typing import List, Dict, Any
from _4metadataFilter import identify_pdf_names_from_party_or_case
from _5finalResponse import generate_final_response
from llm import LLM
import re

def classify_query_type(query):
    '''
    Ask LLM to fill the following variables based on the query:
    single_or_multiple_cases, doc_or_span_level, mention_of_specifics, explanation
    '''
    
    # Initialize LLM without specifying provider or model (will be chosen randomly)
    llm = LLM()
    
    # Construct the prompt for the LLM
    system_prompt = """You are analyzing legal queries for a RAG system that searches through 400+ court judgments. 
    
    Classify the given query based on three characteristics and respond ONLY with a valid JSON object and
    do not mention anything like ```json before or after the json:

    {
        "single_or_multiple_cases": "Single" or "Multiple",
        "doc_or_span_level": "doc" or "span", 
        "mention_of_specifics": "Yes" or "No",
        "explanation": ""
    }

    Classification rules:
    1. single_or_multiple_cases:
       - "Single": Query asks about ONE specific case/judgment
       - "Multiple": Query seeks information across MULTIPLE cases/judgments
    
    2. doc_or_span_level:
       - "doc": Query needs information from entire text of relevant judgements (like summary of a judgement or summary of few judgements)
       - "span": The answer to the query might need information from few-few chunks of text from the judgements rather than whole text from the judgement (like particular evidence, specific arguments, citations, key principles in a specific category of cases)
    
    3. mention_of_specifics:
       - "Yes": Query mentions case numbers (like "LPA/731/2023") OR party names (like "DELHI TECHNOLOGICAL UNIVERSITY Vs DR JAI GOPAL SHARMA")
       - "No": Query doesn't mention specific case identifiers or party names

    4. explanation:
        - explain why you choose the values you did in previous classifications

    Respond with ONLY the JSON object, no other text. Do not include any preceeding or succeeding strings like ```json."""
    
    try:
        # Make the LLM call using the ask method
        result = llm.ask(
            system_prompt=system_prompt,
            user_prompt=f"Classify this query: {query}."
        )
        
        # Get the response text
        llm_response = result['text'].strip()

        # print('\n\n')
        # print(llm_response)
        # print('\n\n')

        # Extract the JSON block from the response
        match = re.search(r'\{.*?\}', llm_response, re.DOTALL)
        if match:
            classification = json.loads(match.group())
        else:
            raise ValueError("No valid JSON object found in the response.1")
        
        # # Parse the JSON response from LLM
        # classification = json.loads(llm_response)
        
        # Extract the three variables
        single_or_multiple_cases = classification.get('single_or_multiple_cases')
        doc_or_span_level = classification.get('doc_or_span_level') 
        mention_of_specifics = classification.get('mention_of_specifics')
        explanation = classification.get('explanation')
        
        # Validate the responses
        if single_or_multiple_cases not in ['Single', 'Multiple']:
            raise ValueError(f"Invalid single_or_multiple_cases value: {single_or_multiple_cases}")
        if doc_or_span_level not in ['doc', 'span']:
            raise ValueError(f"Invalid doc_or_span_level value: {doc_or_span_level}")
        if mention_of_specifics not in ['Yes', 'No']:
            raise ValueError(f"Invalid mention_of_specifics value: {mention_of_specifics}")
            
        return single_or_multiple_cases, doc_or_span_level, mention_of_specifics, explanation
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        return None, None, None, None
    except KeyError as e:
        print(f"Missing key in API response: {e}")
        return None, None, None, None
    except ValueError as e:
        print(f"Invalid classification values: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None, None, None, None

def fetch_based_on(
        query: str,
        single_or_multiple_case: str | None,
        doc_or_span_level: str | None,
        relevant_pdf_names: List[str]
        ) -> List[Dict[str, Any]]:
    '''
    Returns chunks retrieved from pinecone depending on the scenarios
    '''
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("court-judgments")
    
    # Initialize embeddings model
    model = 'all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Generate query embedding
    query_embedding = embeddings.embed_query(query)
    
    results: List[Dict[str, Any]] = []
    
    # Case 1: relevant_pdf_names is not empty
    if relevant_pdf_names:
        if doc_or_span_level == 'doc':
            # Fetch all records where pdf_filename is in relevant_pdf_names
            for pdf_name in relevant_pdf_names:
                response = index.query(
                    vector=query_embedding,
                    filter={"pdf_filename": pdf_name},
                    top_k=10,
                    include_metadata=True
                )
                # Type-safe access to response
                matches = getattr(response, 'matches', [])
                if matches:
                    for match in matches:
                        metadata = getattr(match, 'metadata', {}) or {}
                        results.append({
                            'case_numbers': metadata.get('case_numbers', []),
                            'party_names': metadata.get('party_names', []),
                            'page_number': metadata.get('page_number', []),
                            'text': metadata.get('text', '')
                        })
        
        elif doc_or_span_level == 'span':
            # Fetch top 5 chunks based on similarity from relevant PDFs
            pdf_filter = {"pdf_filename": {"$in": relevant_pdf_names}}
            response = index.query(
                vector=query_embedding,
                filter=pdf_filter,
                top_k=10,
                include_metadata=True
            )
            # Type-safe access to response
            matches = getattr(response, 'matches', [])
            if matches:
                for match in matches:
                    metadata = getattr(match, 'metadata', {}) or {}
                    results.append({
                        'case_numbers': metadata.get('case_numbers', []),
                        'party_names': metadata.get('party_names', []),
                        'page_number': metadata.get('page_number', []),
                        'text': metadata.get('text', '')
                    })
    
    # Case 2: relevant_pdf_names is empty and single_or_multiple_case = 'Multiple'
    elif single_or_multiple_case == 'Multiple':
        if doc_or_span_level == 'doc':
            # Fetch top 10 records based on similarity
            response = index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            
            # Extract unique pdf_filenames
            unique_pdf_names = set()
            matches = getattr(response, 'matches', [])
            if matches:
                for match in matches:
                    metadata = getattr(match, 'metadata', {}) or {}
                    pdf_filename = metadata.get('pdf_filename')
                    if pdf_filename:
                        unique_pdf_names.add(pdf_filename)
            
            # Fetch all records from these PDFs
            for pdf_name in unique_pdf_names:
                pdf_response = index.query(
                    vector=query_embedding,
                    filter={"pdf_filename": pdf_name},
                    top_k=10,  # Large number to get all chunks
                    include_metadata=True
                )
                pdf_matches = getattr(pdf_response, 'matches', [])
                if pdf_matches:
                    for match in pdf_matches:
                        metadata = getattr(match, 'metadata', {}) or {}
                        results.append({
                            'case_numbers': metadata.get('case_numbers', []),
                            'party_names': metadata.get('party_names', []),
                            'page_number': metadata.get('page_number', []),
                            'text': metadata.get('text', '')
                        })
        
        elif doc_or_span_level == 'span':
            # Fetch top 10 chunks based on similarity
            response = index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            matches = getattr(response, 'matches', [])
            if matches:
                for match in matches:
                    metadata = getattr(match, 'metadata', {}) or {}
                    results.append({
                        'case_numbers': metadata.get('case_numbers', []),
                        'party_names': metadata.get('party_names', []),
                        'page_number': metadata.get('page_number', []),
                        'text': metadata.get('text', '')
                    })
    
    # Case 3: relevant_pdf_names is empty and single_or_multiple_case = 'Single'
    elif single_or_multiple_case == 'Single':
        if doc_or_span_level == 'doc':
            # Fetch top 1 record based on similarity
            response = index.query(
                vector=query_embedding,
                top_k=1,
                include_metadata=True
            )
            
            matches = getattr(response, 'matches', [])
            if matches:
                # Get pdf_filename of the top match
                top_match = matches[0]
                metadata = getattr(top_match, 'metadata', {}) or {}
                pdf_filename = metadata.get('pdf_filename')
                
                # Fetch all records from this PDF
                if pdf_filename:
                    pdf_response = index.query(
                        vector=query_embedding,
                        filter={"pdf_filename": pdf_filename},
                        top_k=10,
                        include_metadata=True
                    )
                    pdf_matches = getattr(pdf_response, 'matches', [])
                    if pdf_matches:
                        for match in pdf_matches:
                            match_metadata = getattr(match, 'metadata', {}) or {}
                            results.append({
                                'case_numbers': match_metadata.get('case_numbers', []),
                                'party_names': metadata.get('party_names', []),
                                'page_number': match_metadata.get('page_number', []),
                                'text': match_metadata.get('text', '')
                            })
        
        elif doc_or_span_level == 'span':
            # Fetch top 5 records based on similarity
            response = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            matches = getattr(response, 'matches', [])
            if matches:
                for match in matches:
                    metadata = getattr(match, 'metadata', {}) or {}
                    results.append({
                        'case_numbers': metadata.get('case_numbers', []),
                        'party_names': metadata.get('party_names', []),
                        'page_number': metadata.get('page_number', []),
                        'text': metadata.get('text', '')
                    })
    
    return results

def main():
    st.title('Ask question on judgements passed by Delhi High Court.')
    query = st.text_input('Enter your query: ')

    if not query:
        st.markdown("---")
        st.info("Don't know what to ask? Here are some sample queries:")
        st.write('- Give me the summary of shyam indus power vs delhi cgst commissioner case.')
        st.write('- What evidence did the judge accept in LPA/731/2023?')

    if query:
        with st.spinner("Understanding your question..."):
            single_or_multiple_case, doc_or_span_level, mention_of_specifics, explanation = classify_query_type(query)               

        # st.write(f'mention_of_specifics: {mention_of_specifics}')

        if mention_of_specifics == 'Yes':
            with st.spinner("Identifying relevent judgements for your query..."):
                relevant_pdf_names = identify_pdf_names_from_party_or_case(query)
                # st.write('relevant_pdf_names')
                # st.write(relevant_pdf_names)
        else:
            relevant_pdf_names = []

        with st.spinner("Fetching relevant text snippets..."):
            final_responsechunks = fetch_based_on(
                query,
                single_or_multiple_case,
                doc_or_span_level,
                relevant_pdf_names
                )
        
        with st.spinner("Generating a response..."):
            final_response = generate_final_response(query, final_responsechunks)

        st.write(final_response)

if __name__ == '__main__':
    main()